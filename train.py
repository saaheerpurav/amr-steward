"""GRPO training script for AMR-Steward.
Uses TRL GRPOTrainer with Unsloth-accelerated Llama-3-8B.
Run: python train.py
"""

import os
import re
import sys
import torch
from dataclasses import dataclass, field
from typing import Optional

from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

# Unsloth patching must happen before any model loads
try:
    from unsloth import FastLanguageModel
    USE_UNSLOTH = True
except ImportError:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    USE_UNSLOTH = False
    print("Unsloth not available — falling back to vanilla transformers.")

from env.environment import AMREnvironment
from env.reward import parse_prescription_from_text, compute_total_reward
from data import eucast_parser as eucast

# ── Config ──────────────────────────────────────────────────────────────────

MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct"
MAX_SEQ_LEN = 2048
NUM_EPISODES = 512          # total rollout episodes per training run
CURRICULUM_SCHEDULE = {     # episode → curriculum level
    0:   1,
    128: 2,
    320: 3,
}
LORA_R = 16
LORA_ALPHA = 32
OUTPUT_DIR = "checkpoints/amr-grpo"

SYSTEM_PROMPT = """You are an antimicrobial stewardship AI. Your goal is to prescribe the narrowest-spectrum antibiotic that is microbiologically active against the pathogen, following IDSA guidelines, and correctly dosed for the patient's renal function.

You have a limited investigation budget. Use tools strategically before committing to a prescription.

AVAILABLE TOOLS:
- interpret_resistance <drug>  — look up MIC and EUCAST classification for a drug in the antibiogram
- check_guideline <syndrome>   — retrieve IDSA first-line recommendation for this patient's organism
- assess_patient_factors       — review renal function and allergy constraints

TOOL FORMAT:
INVESTIGATE: {"tool": "interpret_resistance", "arg": "meropenem"}

COMMIT FORMAT (when ready to prescribe):
COMMIT: {"drug": "ceftazidime-avibactam", "dose": "2.5g IV q8h", "duration": "14 days", "justification": "CRE bacteremia, susceptible by MIC, IDSA first-line"}

Think step by step. Investigate before committing. Use the narrowest active agent."""


# ── Prompt builder ───────────────────────────────────────────────────────────

def build_prompt(patient_text: str, tool_results: list[str], budget: int) -> str:
    obs = patient_text
    if tool_results:
        obs += "\n\nINVESTIGATION RESULTS:\n" + "\n---\n".join(tool_results)
    obs += f"\n\nINVESTIGATION BUDGET REMAINING: {budget}"
    if budget == 0:
        obs += " — YOU MUST COMMIT NOW."
    return obs


def make_messages(patient_text: str, tool_results: list[str], budget: int) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(patient_text, tool_results, budget)},
    ]


# ── Rollout collector ────────────────────────────────────────────────────────

def collect_rollout(model, tokenizer, env: AMREnvironment, level: int) -> dict:
    """Run one full episode. Returns (prompt_str, completion_str, reward)."""
    obs = env.reset(curriculum_level=level)
    all_tool_results = []
    full_completion = ""

    while not obs.done:
        messages = make_messages(obs.patient_text, obs.tool_results, obs.budget_remaining)
        if USE_UNSLOTH:
            input_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)
        else:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

        with torch.no_grad():
            out = model.generate(
                input_ids, max_new_tokens=256, do_sample=True,
                temperature=0.8, pad_token_id=tokenizer.eos_token_id,
            )
        generated = out[0][input_ids.shape[-1]:]
        response_text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        full_completion += response_text + "\n"

        # Parse COMMIT
        prescription = parse_prescription_from_text(response_text)
        if prescription:
            from env.models import AMRAction
            action = AMRAction(action_type="COMMIT", prescription=prescription)
            obs, reward, done = env.step(action)
            break

        # Parse INVESTIGATE
        inv_match = re.search(
            r'INVESTIGATE[:\s]+\{.*?"tool"\s*:\s*"([^"]+)".*?"arg"\s*:\s*"([^"]+)".*?\}',
            response_text, re.DOTALL | re.IGNORECASE
        )
        if inv_match:
            tool, arg = inv_match.group(1), inv_match.group(2)
        else:
            # Fallback: try simpler pattern
            inv_match2 = re.search(r'INVESTIGATE[:\s]+(\w+)[:\s]+([\w\-]+)', response_text, re.IGNORECASE)
            if inv_match2:
                tool, arg = inv_match2.group(1), inv_match2.group(2)
            else:
                # No valid action — force commit with empty prescription
                from env.models import AMRAction
                action = AMRAction(action_type="COMMIT", prescription={})
                obs, reward, done = env.step(action)
                break

        from env.models import AMRAction
        action = AMRAction(action_type="INVESTIGATE", tool_name=tool, tool_arg=arg)
        obs, reward, done = env.step(action)

    prompt_text = tokenizer.apply_chat_template(
        make_messages(obs.patient_text, [], BUDGET_BY_LEVEL_REF[level]),
        tokenize=False, add_generation_prompt=True,
    ) if USE_UNSLOTH else ""

    return {
        "prompt": build_prompt(obs.patient_text, [], BUDGET_BY_LEVEL_REF[level]),
        "completion": full_completion,
        "reward": reward,
    }


BUDGET_BY_LEVEL_REF = {1: 5, 2: 4, 3: 3}


# ── Reward function for GRPOTrainer ─────────────────────────────────────────

def make_reward_fn(env_ref: AMREnvironment):
    """Returns a reward function compatible with TRL GRPOTrainer."""

    def reward_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
        rewards = []
        for completion in completions:
            prescription = parse_prescription_from_text(completion)
            if prescription is None:
                rewards.append(0.0)
                continue
            if env_ref.current_patient is None:
                rewards.append(0.0)
                continue

            tool_calls = re.findall(r"INVESTIGATE[:\s]+([^\n]+)", completion, re.IGNORECASE)
            total, _ = compute_total_reward(
                prescription, env_ref.current_patient, tool_calls, eucast
            )
            rewards.append(total)
        return rewards

    return reward_fn


# ── Dataset builder ──────────────────────────────────────────────────────────

def build_dataset(env: AMREnvironment, num_samples: int = 256, level: int = 1) -> Dataset:
    """Build a prompt dataset for GRPOTrainer by generating patient observations."""
    prompts = []
    for _ in range(num_samples):
        obs = env.reset(curriculum_level=level)
        prompt_text = build_prompt(obs.patient_text, [], obs.budget_remaining)
        messages = make_messages(obs.patient_text, [], obs.budget_remaining)
        prompts.append({"prompt": messages})
    return Dataset.from_list(prompts)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading model: {MODEL_NAME}")

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

    env = AMREnvironment()

    # Start at level 1
    current_level = 1
    dataset = build_dataset(env, num_samples=256, level=current_level)

    grpo_config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        save_steps=100,
        report_to="none",
        max_completion_length=512,
        num_generations=4,
        temperature=0.8,
        use_vllm=False,
    )

    reward_fn = make_reward_fn(env)

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
    )

    print("Starting GRPO training...")
    trainer.train()

    print(f"Saving to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
