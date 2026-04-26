---
title: AMR-Steward
emoji: 🦠
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# AMR-Steward

**RL environment for clinical antimicrobial stewardship.** Trains an LLM to prescribe the right antibiotic for drug-resistant bacterial infections — verified against EUCAST breakpoints and IDSA guidelines. No LLM judges.

**Stack:** OpenEnv · TRL GRPOTrainer · Unsloth · HuggingFace Spaces

---

## The Problem

Antimicrobial resistance (AMR) kills **1.27 million people per year** and is projected to surpass cancer as a leading cause of death by 2050. A central driver is inappropriate antibiotic prescribing: wrong drug, wrong dose, or a broad-spectrum agent used when a narrow one would work. Antibiotic stewardship programs exist to fix this, but they are expensive, understaffed, and unavailable in most of the world.

AMR-Steward asks: can an LLM learn to prescribe correctly — not by memorizing guidelines, but by reasoning through resistance data, patient factors, and clinical evidence the way a trained physician would?

---

## How the Environment Works

Each episode is a clinical decision:

1. **Reset** — a synthetic patient is sampled (organism, resistance phenotype, renal function, allergies, antibiogram).
2. **Investigate** — the agent calls up to N tools to gather information before committing.
3. **Commit** — the agent prescribes a drug, dose, and duration.
4. **Reward** — five components evaluate the prescription (see below).

The agent must learn *when* to investigate (budget is limited) and *what* to prescribe given imperfect information.

```
env.reset(curriculum_level=1)
→ AMRObservation(patient_text, tool_results=[], budget_remaining=5)

env.step(INVESTIGATE: interpret_resistance("meropenem"))
→ "meropenem MIC=8.0 mg/L → EUCAST: Resistant"

env.step(INVESTIGATE: check_guideline("bacteremia"))
→ "IDSA: K. pneumoniae (CRE) + bacteremia → first-line: ceftazidime-avibactam"

env.step(COMMIT: {drug: "ceftazidime-avibactam", dose: "2.5g IV q8h", duration: "14 days"})
→ reward: 0.92
```

### Available Tools

| Tool | What it does |
|------|-------------|
| `interpret_resistance(drug)` | Looks up MIC from the antibiogram, classifies via EUCAST (S/I/R) |
| `check_guideline(syndrome)` | Returns IDSA first-line recommendation for this organism + syndrome |
| `assess_patient_factors()` | Returns renal dose adjustments and allergy flags for all antibiogram drugs |

---

## Reward Functions

All components are pure functions — no LLM judge. The terminal reward is RLVR-verifiable.

| Component | Role | What it measures |
|-----------|------|-----------------|
| **R0** Allergy safety | Hard gate | Prescribing a drug the patient is allergic to → total = 0.0 immediately |
| **R1** Microbiological activity | Oracle input | Does the drug cover this organism? (EUCAST MIC lookup) |
| **R2** Guideline concordance | Oracle input | Is this the IDSA-recommended agent? (1.0 = first-line, 0.5 = alternative) |
| **R3** Stewardship | Oracle input | Is this the *narrowest* effective drug? Penalizes unnecessary broad-spectrum use |
| **R4** Dose correctness | Oracle input | Is the dose appropriate for this patient's renal function? |
| **R5** Tool efficiency | Process signal | `(unique_tool_types / budget_spent) × (budget_remaining / budget_total)` — counted from `AMRState.tool_history` (structured `{tool, arg}` log), no text parsing |
| **R6** Output format | Format signal | Clean single COMMIT line (1.0 for ≤3 lines, decays 0.05/line after) |

**Quality ratio** (RLVR oracle): for each patient, `compute_optimal_prescription()` brute-forces all antibiogram drugs to find the maximum achievable process score. The agent is then scored relative to that optimum:

```
process_score = 0.40·R1 + 0.25·R2 + 0.15·R3 + 0.10·R4
quality_ratio = min(1.0, process_score / opt_score)   ← 1.0 iff agent found optimal prescription
total         = 0.90·quality_ratio + 0.10·R5
```

**Multi-head GRPO**: three independent reward functions give the trainer separate gradient channels — format (R6, fast feedback), process (R5 tool efficiency, dense), terminal (quality_ratio, sparse). Each provides a different learning signal at a different timescale.

---

## JEPA World Model

AMR-Steward uses a **Joint Embedding Predictive Architecture (JEPA)** world model to guide the agent's investigation strategy.

Before committing, the world model predicts which tool call would provide the most information given the current known state. This is encoded as an *information gain score* for each available tool, appended to the observation:

```
PREDICTED INFORMATION GAIN:
- check_guideline_UTI: 0.1287
- assess_patient_factors: 0.0614
- interpret_resistance_ceftriaxone: 0.0599
- interpret_resistance_meropenem: 0.0388
```

The world model is pre-trained on synthetic (state, tool, next-state) triples generated from 500 seeded episodes drawn from the same patient distribution used in RL training. The predictor is trained to map `(context_encoder(s_before), tool) → target_encoder(s_after)` using MSE against an EMA-stabilised target encoder (I-JEPA pattern). At inference, information gain for tool `t` is measured as `||predictor(context(s), t) − target_encoder(s)|| / √d_repr` — the L2 distance in target-encoder space, so the training objective and the serving metric are computed in the same embedding geometry.

**Architecture:**
- Context encoder: 64 → 256 → 128 (ReLU MLP)
- Predictor: 128 + 16 (tool one-hot) → 256 → 128 (ReLU MLP)
- Target encoder: EMA copy of context encoder (decay = 0.99)
- State vector: 64-dim handcrafted features (organism, phenotype, site, CrCl, allergy flags, tool-called flags, antibiogram presence)

**Information gain** is measured as `||pred_next − target(s)|| / √repr_dim` — both `pred_next` and the anchor `target(s)` live in target-encoder space, matching the training objective. Tools expected to shift the state representation further in that space are ranked higher.

Weights are pre-trained locally (`jepa_pretrain.py`, seeded for reproducibility) and committed as `jepa_weights.pt`. The environment auto-loads them at startup.

**Dense shaping**: INVESTIGATE steps earn `+0.04` reward for each unique `(tool, argument)` pair called, hard-capped at `+0.20` total so the terminal quality_ratio always dominates. This prevents the agent from learning to commit blindly while still providing gradient signal during the investigation phase.

---

## Curriculum

Training proceeds in three stages:

| Stage | Organisms | Renal function | Budget | Achieved reward |
|-------|-----------|---------------|--------|----------------|
| 1 | Susceptible only | Normal | 5 tools | 0.55 → **0.84** |
| 2 | + Resistant (ESBL, MRSA, VRE) | Mild–moderate impairment | 4 tools | 0.40 → **0.79** |
| 3 | + MDR (CRE, XDR Pseudomonas, VISA) | Severe impairment + allergies | 3 tools | **0.71** (stable) |

---

## Results

GRPO training on `Qwen/Qwen3-4B` + LoRA (r=16) across three curriculum stages (A10G GPU via HF Spaces):

| Stage | Cases | Peak Reward | Final Reward |
|-------|-------|-------------|--------------|
| 1 — Susceptible | 128 | **0.840** | 0.840 |
| 2 — Resistant / MDR | 64 | **0.790** | 0.790 |
| 3 — MDR + Renal + Allergies | 32 | **0.707** | 0.707 |

Reward holds consistently above 0.70 even as case complexity scales from susceptible organisms to MDR + severe renal failure + allergy constraints.

**Training curves across all 3 stages:**

![Reward curves across curriculum stages](reward_curves.png)

A perfect prescription (correct drug, first-line IDSA, narrowest spectrum, correct renal dose, full investigation) scores **1.0** (`quality_ratio = 1.0`). Random prescribing scores ~0.05–0.10. The trained model consistently scores **0.71–0.84** across all stages.

**Improvement: +0.65–0.79 over random baseline** (0.05–0.10 → 0.71–0.84).

---

## Using the Environment

```python
from env.environment import AMREnvironment
from env.models import AMRAction

env = AMREnvironment()
obs = env.reset(curriculum_level=1)
print(obs.patient_text)

# Investigate
action = AMRAction(
    action_type="INVESTIGATE",
    tool_name="interpret_resistance",
    tool_arg="meropenem",
)
obs = env.step(action)
print(obs.tool_results[-1])

# Commit
action = AMRAction(
    action_type="COMMIT",
    prescription={
        "drug": "ceftriaxone",
        "dose": "2g IV q24h",
        "duration": "14 days",
        "justification": "Susceptible K. pneumoniae bacteremia. Narrowest active agent.",
    },
)
obs = env.step(action)
print(f"Reward: {obs.reward}")
print(obs.metadata["reward_breakdown"])
```

### REST API (OpenEnv)

```bash
# Start episode
POST /reset   {"curriculum_level": 1}
→ {"observation": {...}, "reward": null, "done": false}

# Take a step
POST /step    {"action": {"action_type": "INVESTIGATE", "tool_name": "interpret_resistance", "tool_arg": "meropenem"}}
POST /step    {"action": {"action_type": "COMMIT", "prescription": {"drug": "...", "dose": "...", "duration": "...", "justification": "..."}}}
→ {"observation": {...}, "reward": 0.92, "done": true}

GET  /state   → full episode state
GET  /health  → 200 OK
```

See [`demo.py`](demo.py) for a complete worked example comparing an untrained broad-spectrum guess against a trained IDSA-first-line prescription.

---

## Data Sources

- **IDSA Guidelines**: IDSA Clinical Practice Guidelines 2022/2023 (bacteremia, UTI, pneumonia, intra-abdominal infection)
- **EUCAST Breakpoints**: EUCAST Clinical Breakpoints v16.0 (2026)
- **Drug Properties**: Standard prescribing references (renal adjustments, allergy flags)
- **Patient Cases**: Synthetically generated from realistic clinical distributions

---

## Links

| Resource | URL |
|----------|-----|
| Live Environment (HF Space) | https://divyanshb06-amrsteward.hf.space |
| Trained Model (HF Hub) | https://huggingface.co/saaheerpurav/amr-steward-model |
| Training Notebook (Colab) | https://colab.research.google.com/github/saaheerpurav/amr-steward/blob/main/AMR_Steward.ipynb |

---

## Judging Criteria

| Criterion | Weight | Evidence |
|---|---|---|
| **Environment Innovation** | 40% | Clinical AMR domain — zero prior RL environments exist for antibiotic stewardship. JEPA-inspired world model (Joint Embedding Predictive Architecture, Meta AI) pre-trained on synthetic (state, tool, next-state) triples from 500 seeded episodes guides investigation strategy. Quality-ratio oracle brute-forces the optimal prescription at reset time, giving a patient-specific reward ceiling with zero variance. R0 hard allergy gate, R3 stewardship gated on R1 — three independent anti-hacking layers. |
| **Storytelling** | 30% | 1.27 million people die from antimicrobial resistance per year — more than HIV or malaria. The before/after is visceral: untrained model prescribes meropenem to a carbapenem-resistant organism (reward ~0.10, ineffective treatment); trained model investigates resistance, checks IDSA guidelines, adjusts for renal function, prescribes ceftazidime-avibactam at the correct renal dose (reward 0.84). Wrong drug → patient dies. Right drug → patient lives. |
| **Showing Improvement** | 20% | GRPO training on Qwen3-4B across three curriculum stages (A10G GPU). Stage 1: 0.55 → **0.84**. Stage 2: 0.40 → **0.79**. Stage 3: **0.71** (stable under max complexity). Reward holds above 0.70 as case complexity increases from susceptible organisms to MDR + renal failure + allergy constraints. Training curves in `reward_curves.png`. |
| **Reward & Training Pipeline** | 10% | Multi-head GRPO: three independent reward functions (format R6, tool efficiency R5, terminal quality_ratio) give the trainer separate gradient channels at different timescales. Dense shaping (+0.04/unique tool call, capped +0.20) provides per-step signal without dominating the terminal reward. Seven reward components (R0–R6), all pure functions — no LLM judge anywhere in the pipeline. R5 computed from a structured tool-call log, not text heuristics. |

---

## Team

Built at a 24-hour hackathon, April 2026.

| Person | Role |
|--------|------|
| Saaheer | ML/RL — JEPA world model, reward functions, GRPO training |
| Bhatia | Backend — OpenEnv environment, FastAPI, HuggingFace deployment |
| Palak | Data — Medical data tables, patient cases, content |

---

*No real patient data was used. All patient cases are synthetically generated.*
