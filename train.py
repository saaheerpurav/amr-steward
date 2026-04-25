"""GRPO training script for AMR-Steward.

Colab-friendly entrypoint for TRL + Unsloth training.

Example:
    python train.py --stage 1 --samples 64 --model-name Qwen/Qwen2.5-0.5B-Instruct

Recommended launch in Colab / multi-GPU environments:
    accelerate launch train.py --stage 1 --samples 128
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset

try:
    from unsloth import FastLanguageModel

    USE_UNSLOTH = True
except ImportError:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    USE_UNSLOTH = False
    print("Unsloth not available; falling back to vanilla transformers.")

from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).parent))

from data import eucast_parser as eucast
from env import AMREnvironment
from env.models import PatientCase
from env.reward import (
    _load_drug_properties,
    _load_idsa,
    compute_total_reward,
    parse_prescription_from_text,
    parse_tool_calls_from_text,
)


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_SEQ_LEN = 2048
LORA_R = 16
LORA_ALPHA = 32
DEFAULT_OUTPUT_DIR = "checkpoints/amr-grpo"

# Curriculum: (num_samples, level) tuples
CURRICULUM = [
    (512, 1),
    (512, 2),
    (256, 3),
]

SYSTEM_PROMPT = """You are an antimicrobial stewardship AI.
Your job is to prescribe the narrowest-spectrum antibiotic that is:
1. Microbiologically active
2. Guideline concordant
3. Correctly dosed for the patient's renal function

Use the available investigation tools before you commit when needed.

TOOLS:
  INVESTIGATE: {"tool": "interpret_resistance", "arg": "<drug>"}
  INVESTIGATE: {"tool": "check_guideline", "arg": "<syndrome>"}
  INVESTIGATE: {"tool": "assess_patient_factors"}

FINAL ANSWER FORMAT:
  COMMIT: {"drug": "...", "dose": "...", "duration": "...", "justification": "..."}

Do not output anything after the COMMIT line.
"""

_IDSA = _load_idsa()
_DRUG_PROPS = _load_drug_properties()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AMR-Steward with GRPO.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3],
        help="Train only a single curriculum stage.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="Override sample count for the selected stage or for all stages.",
    )
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--max-completion-length", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--disable-unsloth",
        action="store_true",
        help="Force transformers path even if Unsloth is installed.",
    )
    parser.add_argument(
        "--save-merged",
        action="store_true",
        help="Attempt to save a merged model when supported by the runtime.",
    )
    return parser.parse_args()


def _render_prompt(obs) -> str:
    parts = [f"SYSTEM:\n{SYSTEM_PROMPT}", f"USER:\n{obs.patient_text}"]
    if obs.tool_results:
        parts.append("INVESTIGATION RESULTS:\n" + "\n---\n".join(obs.tool_results))
    if obs.world_model_rankings:
        parts.append(obs.world_model_rankings)
    parts.append(f"INVESTIGATION BUDGET REMAINING: {obs.budget_remaining}")
    if obs.budget_remaining == 0:
        parts.append("YOU MUST COMMIT NOW.")
    parts.append("ASSISTANT:")
    return "\n\n".join(parts)


def build_dataset(level: int, num_samples: int) -> Dataset:
    """Build a prompt dataset with the exact patient case embedded as metadata."""
    env = AMREnvironment()
    rows: list[dict[str, Any]] = []
    for idx in range(num_samples):
        obs = env.reset(curriculum_level=level)
        patient = env.current_patient
        rows.append(
            {
                "prompt": _render_prompt(obs),
                "patient_json": json.dumps(asdict(patient)),
                "curriculum_level": level,
                "case_id": f"level{level}-case{idx}",
            }
        )
    return Dataset.from_list(rows)


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        chunks: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, str):
                    chunks.append(content)
            elif isinstance(item, str):
                chunks.append(item)
        return "\n".join(chunks)
    return str(completion)


def _patient_from_json(payload: str) -> PatientCase:
    return PatientCase(**json.loads(payload))


def make_reward_fn():
    """Return a GRPO-compatible reward function tied to the prompted patient case."""

    def reward_fn(
        prompts,
        completions,
        patient_json,
        curriculum_level=None,
        case_id=None,
        log_metric=None,
        log_extra=None,
        **kwargs,
    ) -> list[float]:
        rewards: list[float] = []
        breakdowns: list[dict[str, float]] = []

        for completion, patient_payload in zip(completions, patient_json):
            completion_text = _completion_to_text(completion)
            prescription = parse_prescription_from_text(completion_text)
            if prescription is None:
                rewards.append(0.0)
                breakdowns.append(
                    {
                        "R1_activity": 0.0,
                        "R2_guideline": 0.0,
                        "R3_stewardship": 0.0,
                        "R4_dose": 0.0,
                        "R5_reasoning": 0.0,
                        "total": 0.0,
                    }
                )
                continue

            patient = _patient_from_json(patient_payload)
            tool_calls = parse_tool_calls_from_text(completion_text)
            total, breakdown = compute_total_reward(
                prescription=prescription,
                patient=patient,
                tool_call_history=tool_calls,
                eucast=eucast,
                idsa=_IDSA,
                drug_properties=_DRUG_PROPS,
            )
            rewards.append(float(total))
            breakdowns.append(breakdown)

        if log_metric and breakdowns:
            for metric in (
                "R1_activity",
                "R2_guideline",
                "R3_stewardship",
                "R4_dose",
                "R5_reasoning",
                "total",
            ):
                log_metric(metric, sum(b[metric] for b in breakdowns) / len(breakdowns))

        if log_extra and case_id is not None:
            log_extra("case_id", list(case_id))
            log_extra("parsed_commit", ["yes" if r > 0 else "no" for r in rewards])

        return rewards

    return reward_fn


def load_model(model_name: str, disable_unsloth: bool):
    use_unsloth = USE_UNSLOTH and not disable_unsloth

    if use_unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=MAX_SEQ_LEN,
            dtype=None,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_R,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=LORA_ALPHA,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
    else:
        from peft import LoraConfig, get_peft_model

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        lora_cfg = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(model, lora_cfg)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer


def make_stage_plan(args: argparse.Namespace) -> list[tuple[int, int]]:
    if args.stage is not None:
        samples = args.samples if args.samples is not None else dict(CURRICULUM)[args.stage]
        return [(samples, args.stage)]

    if args.samples is not None:
        return [(args.samples, level) for _, level in CURRICULUM]

    return CURRICULUM


def train_stage(
    model,
    tokenizer,
    args: argparse.Namespace,
    level: int,
    num_samples: int,
    stage_index: int,
):
    print(f"\n=== Stage {stage_index}: level={level}, samples={num_samples} ===")
    dataset = build_dataset(level, num_samples)
    stage_output_dir = f"{args.output_dir}/stage{stage_index}"

    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    config = GRPOConfig(
        output_dir=stage_output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=bf16_supported,
        fp16=torch.cuda.is_available() and not bf16_supported,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        report_to="none",
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        temperature=args.temperature,
        log_completions=True,
        use_vllm=False,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=make_reward_fn(),
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    history_path = Path(stage_output_dir) / "log_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")
    print(f"Saved training log history to {history_path}")
    return trainer


def save_final_artifacts(trainer, tokenizer, args: argparse.Namespace):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    if args.save_merged and USE_UNSLOTH and hasattr(trainer.model, "save_pretrained_merged"):
        merged_dir = output_dir / "merged_16bit"
        trainer.model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
        print(f"Saved merged model to {merged_dir}")


def main():
    args = parse_args()
    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model(args.model_name, args.disable_unsloth)

    trainer = None
    for stage_index, (num_samples, level) in enumerate(make_stage_plan(args), start=1):
        trainer = train_stage(model, tokenizer, args, level, num_samples, stage_index)

    if trainer is None:
        raise RuntimeError("No training stages were scheduled.")

    print(f"\nSaving final artifacts to {args.output_dir}")
    save_final_artifacts(trainer, tokenizer, args)
    print("Done.")


if __name__ == "__main__":
    main()
