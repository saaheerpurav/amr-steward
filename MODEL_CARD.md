---
base_model: Qwen/Qwen3-4B
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
tags:
- base_model:adapter:Qwen/Qwen3-4B
- grpo
- lora
- transformers
- trl
- reinforcement-learning
- medical
- antimicrobial-resistance
- clinical-decision-support
- openenv
---

# AMR-Steward — Antibiotic Prescribing Agent

**Qwen3-4B + LoRA trained with multi-head GRPO** to prescribe the correct antibiotic for drug-resistant bacterial infections. Reward is fully verifiable: seven pure-function components against EUCAST v16.0 breakpoints and IDSA 2022/2023 clinical guidelines. **No LLM-as-judge anywhere.**

Trained inside the [AMR-Steward OpenEnv environment](https://huggingface.co/spaces/divyanshb06/amrsteward) — built for the Meta PyTorch OpenEnv Hackathon, April 2026.

| | |
|---|---|
| **Base model** | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) |
| **Fine-tuning** | LoRA (r=16, α=32, targets: q/k/v/o projections) |
| **Algorithm** | Multi-head GRPO (TRL + Unsloth, bf16) |
| **Hardware** | A10G GPU — HuggingFace Spaces |
| **Live demo** | [divyanshb06-amrsteward.hf.space/demo](https://divyanshb06-amrsteward.hf.space/demo) |
| **Environment** | [github.com/saaheerpurav/amr-steward](https://github.com/saaheerpurav/amr-steward) |
| **Writeup** | [BLOG.md](https://github.com/saaheerpurav/amr-steward/blob/main/BLOG.md) |

---

## Training Results

Three curriculum stages — susceptible organisms → MDR + severe renal failure + allergy constraints:

| Stage | Organisms | Budget | Steps | Start → Final | Peak | Mean |
|-------|-----------|--------|-------|--------------|------|------|
| **1 — Susceptible** | *K. pneumoniae*, *E. coli*, *S. aureus* (susceptible) | 5 tools | 128 | 0.54 → **0.90** | **0.923** | 0.840 |
| **2 — Resistant/MDR** | + ESBL, MRSA, VRE | 4 tools | 64 | 0.86 → **0.84** | 0.840 | 0.790 |
| **3 — MDR + Renal + Allergies** | + CRE, XDR Pseudomonas, VISA | 3 tools | 32 | 0.81 → **0.88** | **0.988** | 0.707 |

Random baseline: **~0.07**. Trained model: **12× better** on Stage 1, **10× better** on Stage 3.

Reward holds **above 0.70** even at Stage 3 — MDR organisms, CrCl 8, penicillin allergy, 3-tool budget.

![Reward curves — all 3 curriculum stages](reward_curves.png)

![Training summary — improvement vs random baseline](training_summary.png)

---

## What This Model Does

The agent receives a clinical patient case and must investigate, then prescribe:

```
Patient: 67F, ICU, K. pneumoniae bacteremia, meropenem MIC=8.0, CrCl=35, no allergies

Agent investigates:
  → interpret_resistance("meropenem")       → "MIC 8.0 → EUCAST: Resistant"
  → check_guideline("bacteremia")           → "IDSA: CRE K. pneumoniae → ceftazidime-avibactam"
  → assess_patient_factors()               → "CrCl 35: reduce to 1.25g IV q8h"

Agent prescribes:
  → ceftazidime-avibactam 1.25g IV q8h, 14 days
  → reward: 0.92
```

Without training (broad-empiric): prescribes meropenem → **reward ~0.11** (resistant organism, drug has zero effect).

---

## JEPA World Model

The training environment includes a **JEPA (Joint Embedding Predictive Architecture) world model** — the first application of Meta AI's I-JEPA pattern ([Assran et al., CVPR 2023](https://arxiv.org/abs/2301.08243)) inside a clinical RL environment.

The world model (≈50K params) predicts in latent space how each tool call would change the agent's known clinical state. It uses an **EMA-stabilised target encoder** (τ=0.99) — the critical anti-collapse mechanism from the original I-JEPA:

```
context_encoder(s_before) + tool → predictor → pred_repr
target_encoder(s_after)                      → tgt_repr   [EMA, stop-gradient]
Loss = MSE(pred_repr, tgt_repr)
```

Three training signals from JEPA: observation hints (ranked tool suggestions), JEPA-weighted reward shaping (0.5×–1.5× bonus multiplier), latent consistency bonus.

---

## Reward Design

All components are pure functions — deterministic, RLVR-verifiable, zero subjectivity:

| Component | What it measures | Range |
|-----------|-----------------|-------|
| **R0** Allergy gate | Prescribing an allergen → total = 0.0, episode ends | {0, 1} |
| **R1** Microbiologic activity | EUCAST MIC classification vs prescribed drug | {0, 1} |
| **R2** Guideline concordance | IDSA first-line=1.0, alternative=0.5, other=0.0 | {0, 0.5, 1} |
| **R3** Stewardship (gated on R1) | Narrowest active spectrum; zero if drug doesn't work | [0, 1] |
| **R4** Dose correctness | Matches renal-tier adjusted dose | [0, 1] |
| **R5** Tool efficiency | (unique tool types / budget spent) × (remaining / total) | [0, 1] |
| **R6** Format | Clean single COMMIT line | [0, 1] |

**Quality ratio (RLVR oracle):**
```python
process_score = 0.40·R1 + 0.25·R2 + 0.15·R3 + 0.10·R4
opt_score     = compute_optimal_prescription(patient)  # brute-force over antibiogram
quality_ratio = min(1.0, process_score / opt_score)   # 1.0 iff agent found optimal drug
total         = 0.90·quality_ratio + 0.10·R5
```

**Multi-head GRPO**: three independent reward functions (format R6, process R5, terminal quality_ratio) give the trainer separate gradient channels at three timescales — fast format feedback, per-step investigation signal, sparse terminal quality.

---

## Validation

### Published Clinical Cases — 3/3 match expert recommendations

| Case | Citation | Expert Prescription | Quality |
|------|----------|---------------------|---------|
| CRE bacteremia, post-renal-transplant | Tamma et al. *Clin Infect Dis.* 2023 | Ceftazidime-avibactam 1.25g IV q8h | **1.000** |
| MSSA bacteremia | Maraolo et al. *Open Forum Infect Dis.* 2018 | Cefazolin 2g IV q8h | **1.000** |
| VRE on hemodialysis | Britt et al. *Clin Infect Dis.* 2015 | Daptomycin 8mg/kg post-HD | **0.939** |

### Adversarial Stress Test — 10/10 pass

| Policy | Pass rate (quality_ratio ≥ 0.85) |
|--------|----------------------------------|
| Broad-empiric (always meropenem) | 0 / 10 |
| Random (seed=42) | 2 / 10 |
| EUCAST-only (no IDSA) | 7 / 10 |
| **Trained model** | **10 / 10** |

---

## Usage

This is a PEFT LoRA adapter — load on top of Qwen3-4B:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B")
model = PeftModel.from_pretrained(base_model, "saaheerpurav/amr-steward-model")
tokenizer = AutoTokenizer.from_pretrained("saaheerpurav/amr-steward-model")
```

To use inside the AMR-Steward environment (recommended):

```bash
git clone https://github.com/saaheerpurav/amr-steward
pip install -r requirements.txt
uvicorn app:app --port 7860
# Then POST /reset + POST /step via the REST API
```

Try the live demo: [divyanshb06-amrsteward.hf.space/demo](https://divyanshb06-amrsteward.hf.space/demo)

---

## Scope and Limitations

- **Not approved for clinical use.** Research artefact only.
- Covers the five WHO critical-priority pathogens: *K. pneumoniae*, *E. coli*, *P. aeruginosa*, *S. aureus*, *Enterococcus* spp.
- Single-organism, single-drug episodes — no polymicrobial cases or combination therapy.
- Trained on synthetic patient cases, not real EHR data.
- Vancomycin dosing is renal-tier-based, not AUC/MIC-guided therapeutic drug monitoring.

---

## Training Infrastructure

| | |
|---|---|
| **GPU** | NVIDIA A10G (24 GB) via HuggingFace Spaces |
| **Precision** | bf16 |
| **LoRA rank** | r=16, α=32 |
| **GRPO generations** | 4 per step |
| **Max completion length** | 768 tokens |
| **Stage 1 steps** | 128 |
| **Stage 2 steps** | 64 |
| **Stage 3 steps** | 32 |
| **Framework** | TRL 0.17+ · Unsloth · HuggingFace Transformers |

---

## Citation

```bibtex
@misc{amr-steward-2026,
  title  = {AMR-Steward: RLVR Training Environment for Clinical Antimicrobial Stewardship},
  author = {Saaheer Purav and Divyansh Bhatia and Palak},
  year   = {2026},
  url    = {https://github.com/saaheerpurav/amr-steward}
}
```

---

*Built at Meta PyTorch OpenEnv Hackathon India, April 2026. Not approved for clinical use.*
