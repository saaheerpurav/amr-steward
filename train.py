"""GRPO training script for AMR-Steward.
Uses TRL GRPOTrainer with Unsloth-accelerated Llama-3-8B-Instruct.
Run: python train.py
"""

import re
import sys
import torch
from pathlib import Path
from datasets import Dataset

try:
    from unsloth import FastLanguageModel
    USE_UNSLOTH = True
except ImportError:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    USE_UNSLOTH = False
    print("Unsloth not available — falling back to vanilla transformers.")

from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).parent))
from env.environment import AMREnvironment, _get_world_model
from env.reward import (
    parse_prescription_from_text,
    compute_total_reward,
    _load_idsa,
    _load_drug_properties,
)
from data import eucast_parser as eucast

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME   = "unsloth/Meta-Llama-3.1-8B-Instruct"
MAX_SEQ_LEN  = 2048
LORA_R       = 16
LORA_ALPHA   = 32
OUTPUT_DIR   = "checkpoints/amr-grpo"

# Curriculum: (num_samples, level) tuples — train each stage then move on
CURRICULUM = [
    (512, 1),   # level 1: susceptible only, normal renal
    (512, 2),   # level 2: add resistant, mild renal impairment
    (256, 3),   # level 3: MDR, severe impairment, allergies
]

SYSTEM_PROMPT = """\
You are an antimicrobial stewardship AI. Prescribe the narrowest-spectrum antibiotic \
that is microbiologically active, follows IDSA guidelines, and is correctly dosed for \
the patient's renal function.

You have a limited investigation budget. Use tools before committing.

TOOLS:
  interpret_resistance <drug>  — EUCAST MIC classification for a drug in the antibiogram
  check_guideline <syndrome>   — IDSA first-line recommendation for this organism
  assess_patient_factors       — renal function tier and allergy constraints

TOOL FORMAT:
  INVESTIGATE: {"tool": "interpret_resistance", "arg": "meropenem"}

COMMIT FORMAT:
  COMMIT: {"drug": "ceftazidime-avibactam", "dose": "2.5g IV q8h", "duration": "14 days", \
"justification": "CRE bacteremia, susceptible by MIC, IDSA first-line"}

Think step by step. Investigate before committing. Use the narrowest active agent.\
"""

# Pre-load data so the reward function is fast
_IDSA = _load_idsa()
_DRUG_PROPS = _load_drug_properties()


# ── Prompt helpers ────────────────────────────────────────────────────────────

def _obs_to_text(obs) -> str:
    """Convert an AMRObservation to a plain text user message."""
    text = obs.patient_text
    if obs.tool_results:
        text += "\n\nINVESTIGATION RESULTS:\n" + "\n---\n".join(obs.tool_results)
    if obs.world_model_rankings:
        text += "\n\n" + obs.world_model_rankings
    text += f"\n\nINVESTIGATION BUDGET REMAINING: {obs.budget_remaining}"
    if obs.budget_remaining == 0:
        text += " — YOU MUST COMMIT NOW."
    return text


def _make_messages(obs) -> list[dict]:
    return [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": _obs_to_text(obs)},
    ]


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_dataset(level: int, num_samples: int) -> Dataset:
    """Build a prompt dataset for GRPOTrainer: each row is a chat-format prompt."""
    env = AMREnvironment()
    rows = []
    for _ in range(num_samples):
        obs = env.reset(curriculum_level=level)
        rows.append({"prompt": _make_messages(obs), "_level": level})
    return Dataset.from_list(rows)


# ── Reward function ───────────────────────────────────────────────────────────

def make_reward_fn(level: int):
    """Return a GRPOTrainer-compatible reward function for a given curriculum level."""

    def reward_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
        rewards = []
        env = AMREnvironment()
        for completion in completions:
            prescription = parse_prescription_from_text(completion)
            if prescription is None:
                rewards.append(0.0)
                continue

            # Reconstruct patient from a fresh episode to get a stable patient for scoring.
            # In practice the patient is embedded in the prompt; we use a fresh one as an
            # approximation since GRPOTrainer doesn't expose the original env state.
            obs = env.reset(curriculum_level=level)
            patient = env.current_patient

            tool_calls = re.findall(r"INVESTIGATE[:\s]+([^\n]+)", completion, re.IGNORECASE)
            total, _ = compute_total_reward(
                prescription, patient, tool_calls,
                eucast, idsa=_IDSA, drug_properties=_DRUG_PROPS,
            )
            rewards.append(float(total))
        return rewards

    return reward_fn


# ── Training loop ─────────────────────────────────────────────────────────────

def load_model():
    if USE_UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LEN,
            dtype=None,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_R,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=LORA_ALPHA,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
    else:
        from peft import LoraConfig, get_peft_model
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        lora_cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, bias="none",
                              task_type="CAUSAL_LM")
        model = get_peft_model(model, lora_cfg)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def train_stage(model, tokenizer, level: int, num_samples: int, stage: int):
    print(f"\n=== Stage {stage}: curriculum level {level}, {num_samples} samples ===")
    dataset = build_dataset(level, num_samples)

    config = GRPOConfig(
        output_dir=f"{OUTPUT_DIR}/stage{stage}",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        save_steps=200,
        report_to="none",
        max_completion_length=512,
        num_generations=4,
        temperature=0.8,
        use_vllm=False,
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=make_reward_fn(level),
        args=config,
        train_dataset=dataset,
    )
    trainer.train()
    return trainer


def main():
    print(f"Loading model: {MODEL_NAME}")
    model, tokenizer = load_model()

    for stage_idx, (num_samples, level) in enumerate(CURRICULUM, start=1):
        trainer = train_stage(model, tokenizer, level, num_samples, stage_idx)

    print(f"\nSaving final model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
