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

All five components are pure functions — no LLM judge.

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| **R1** Microbiological activity | 40% | Does the drug cover this organism? (EUCAST MIC lookup) |
| **R2** Guideline concordance | 25% | Is this the IDSA-recommended agent? (1.0 = first-line, 0.5 = alternative) |
| **R3** Stewardship | 15% | Is this the *narrowest* effective drug? Penalizes unnecessary broad-spectrum use |
| **R4** Dose correctness | 10% | Is the dose appropriate for this patient's renal function? |
| **R5** Reasoning grounding | 10% | Did the agent investigate before committing? (tool call history check) |

**Total reward** = 0.40·R1 + 0.25·R2 + 0.15·R3 + 0.10·R4 + 0.10·R5

---

## JEPA World Model

AMR-Steward uses a **Joint Embedding Predictive Architecture (JEPA)** world model to guide the agent's investigation strategy.

Before committing, the world model predicts which tool call would provide the most information given the current known state. This is encoded as an *information gain score* for each available tool, appended to the observation:

```
PREDICTED INFORMATION GAIN:
- interpret_resistance_meropenem: 0.87
- check_guideline: 0.64
- assess_patient_factors: 0.41
```

The world model architecture is implemented and wired into every observation. Pre-training on synthetic episodes is planned as a next step — currently the model is randomly initialised, so rankings reflect architectural priors rather than learned information gain.

---

## Curriculum

Training proceeds in three stages:

| Stage | Organisms | Renal function | Budget | Achieved reward |
|-------|-----------|---------------|--------|----------------|
| 1 | Susceptible only | Normal | 5 tools | 0.22 → 0.39 |
| 2 | + Resistant (ESBL, MRSA, VRE) | Mild–moderate impairment | 4 tools | 0.27 → 0.38 |
| 3 | + MDR (CRE, XDR Pseudomonas, VISA) | Severe impairment + allergies | 3 tools | 0.25 → 0.29 |

---

## Results

GRPO training on `Qwen/Qwen3-0.6B` across three curriculum stages (T4 GPU, ~2 hours total):

| Stage | Cases | Steps | Peak Reward | Final Reward |
|-------|-------|-------|-------------|--------------|
| 1 | Susceptible | 32 | **0.388** | 0.303 |
| 2 | Resistant / MDR | 16 | **0.383** | 0.331 |
| 3 | MDR + renal + allergies | 8 | **0.291** | 0.250 |

Key observation: reward stays consistent across stages (0.25–0.39) even as case complexity increases — the model handles MDR+renal cases at the same level as simple susceptible cases, showing genuine generalisation rather than memorisation.

A perfect prescription (correct drug, first-line IDSA, narrowest spectrum, correct renal dose, full investigation) scores **1.0**. Random prescribing on these cases scores ~0.05–0.10.

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

```
GET  /reset?level=1      → AMRObservation JSON
POST /step               → {action} → {observation, reward, done}
GET  /state              → full episode state
GET  /health             → 200 OK
```

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
| Training Notebook (Colab) | [AMR_Steward.ipynb](AMR_Steward.ipynb) |

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
