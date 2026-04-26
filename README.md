---
title: AMR-Steward
emoji: 🦠
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# 🦠 AMR-Steward: An RL Environment for Clinical Antimicrobial Stewardship

> **TL;DR:** AMR-Steward is an OpenEnv reinforcement learning environment that trains an LLM to prescribe antibiotics correctly for drug-resistant infections. We bypassed the "LLM-as-a-judge" trap entirely by building a **fully deterministic, verifiable reward stack (RLVR)** based on EUCAST clinical breakpoints and IDSA guidelines. The headline innovation? We deployed **Meta's I-JEPA architecture** as a self-supervised world model *inside* the environment to rank tool calls by predicted information gain in latent space. The trained model improves from a ~0.07 random baseline to **0.84–0.90** across a 3-stage curriculum (Stage 1 peak 0.923, Stage 3 peak 0.988 — 12× over random) and passes **10/10** adversarial stress tests. 

### 🔗 Quick Links
- **Live Environment (HF Space):** [divyanshb06-amrsteward.hf.space](https://divyanshb06-amrsteward.hf.space)
- **Interactive Demo:** [/demo](https://divyanshb06-amrsteward.hf.space/demo) — Clinical clue cards with JEPA info-gain bars
- **Trained Model (HF Hub):** [saaheerpurav/amr-steward-model](https://huggingface.co/saaheerpurav/amr-steward-model)
- **Training Notebook (Colab):** [AMR_Steward.ipynb](https://colab.research.google.com/github/saaheerpurav/amr-steward/blob/main/AMR_Steward.ipynb)
- **Technical Blog:** [BLOG.md](BLOG.md)
- **Source Code:** [GitHub Repository](https://github.com/saaheerpurav/amr-steward)
- **Docs:** [Architecture](docs/Architecture.md) | [Reward Spec](docs/Reward-spec.md) | [Failure Analysis](docs/Failure-Analysis.md) | [Clinical Validation Matrix](docs/Clinical-Validation-Matrix.md)

---

## 📈 The Results: 12× Better Than Random

We trained Qwen3-4B + LoRA using GRPO across three curriculum stages. As case complexity scaled from simple susceptible organisms to MDR infections with severe renal failure and penicillin allergies, our agent maintained a reward above 0.70. Random baseline: ~0.07. **Our trained model reaches 0.84–0.90 — 12× better than random on Stage 1, with reward holding above 0.70 even on the hardest Stage 3 cases.**

![Training summary — improvement over random baseline](training_summary.png)

![Reward curves across curriculum stages](reward_curves.png)

---

## 📖 The Story: Why This Matters

Antimicrobial resistance (AMR) is a silent pandemic. It kills **1.27 million people per year**—more than HIV or malaria. A central driver of this crisis is inappropriate antibiotic prescribing: using the wrong drug, the wrong dose, or a broad-spectrum "nuke" when a targeted "sniper rifle" would have worked.

Antibiotic stewardship programs exist to fix this, but they require highly specialized, expensive human experts. 

**AMR-Steward asks a fundamental question:** Can an LLM learn to prescribe correctly—not by memorizing static text guidelines, but by actively reasoning through resistance data, patient factors, and clinical evidence the way an infectious disease physician would?

---

## Hackathon Validation Checklist

Every item below is explicitly addressed and verifiable from this repo.

| # | Validation Criterion | Evidence |
|---|---|---|
| 1 | Public, cloneable HF Space (logged-out browser, no 404) | [divyanshb06-amrsteward.hf.space](https://divyanshb06-amrsteward.hf.space) returns 200 OK with public OpenEnv landing page; OpenAPI docs at `/docs`, health probe at `/health` |
| 2 | Valid OpenEnv structure: `Environment` base class | [`env/environment.py`](env/environment.py) — `class AMREnvironment(Environment)` from `openenv.core.env_server` |
| 3 | Gym-style `reset` / `step` / `state` | [`env/environment.py`](env/environment.py) — all three present, Pydantic-typed, returns `AMRObservation` (subclass of `openenv.core.env_server.Observation`) |
| 4 | Parseable `openenv.yaml` | [`openenv.yaml`](openenv.yaml) — valid YAML with `name`, `action_space`, `observation_space`, `reward_range`, `curriculum_levels` |
| 5 | Training evidence committed as `.png` files (loss + reward curves) | [`reward_curves.png`](reward_curves.png) (3-stage GRPO reward) + [`training_summary.png`](training_summary.png) (improvement vs random baseline) — both embedded inline in the Results section |
| 6 | Runnable training script (Python or Colab notebook) | [`train.py`](train.py) (Python, ~600 lines) + [`AMR_Steward.ipynb`](AMR_Steward.ipynb) ([Colab](https://colab.research.google.com/github/saaheerpurav/amr-steward/blob/main/AMR_Steward.ipynb)) — both end-to-end reproducible on A10G |
| 7 | README links every deliverable, plots embedded inline | This Quick Links table + Results section embeds both PNGs via relative paths (works on GitHub *and* HF Space) |
| 8 | Writeup linked from README | [`BLOG.md`](BLOG.md) — 10-section, ~2000 words |
| 9 | Unit + integration tests pass | `pytest test_env.py test_jepa_integration.py` — **21/21 tests pass** (8 env tests, 13 JEPA integration tests); covers reset/step/budget/reward, JEPA info-gain bounds, dense cap enforcement, EMA world model loading |
| 10 | Reproducible evaluation | `python eval.py` (baseline benchmarks) + `python eval_published_cases.py` (3 published cases) + `python eval_adversarial.py --seed 42` (10 adversarial cases) — all run on CPU in <60 seconds, no GPU required |

---

## Clinical Validation Against Published Case Literature

Three real cases from peer-reviewed literature, encoded as `PatientCase` objects and run through the live environment. The RLVR oracle scores the published expert recommendation independently — no hand-tuning.

| Case | Patient | Published Recommendation | Citation | AMR-Steward Output | R1 | R2 | Quality |
|------|---------|--------------------------|----------|--------------------|----|----|---------|
| **CRE Bacteremia** | 67M post-renal-transplant, K. pneumoniae, CrCl 40 | Ceftazidime-avibactam (IDSA preferred for KPC-CRE bacteremia, renal-adjusted) | Tamma PD et al. *Clin Infect Dis.* 2023;76(7):1228–1270. [PMC9890506](https://pubmed.ncbi.nlm.nih.gov/36462428/) | `ceftazidime-avibactam 1.25g IV q8h` | ✅ 1.0 | ✅ 1.0 | **1.000** |
| **MSSA Bacteremia** | 58M, S. aureus bacteremia, CrCl 65 | Cefazolin (IDSA first-line for MSSA bacteremia; non-inferior to nafcillin) | Maraolo AE et al. *Open Forum Infect Dis.* 2018;5(3):ofy042. [doi:10.1093/ofid/ofy042](https://doi.org/10.1093/ofid/ofy042) | `cefazolin 2g IV q8h` | ✅ 1.0 | ✅ 1.0 | **1.000** |
| **VRE on Hemodialysis** | 72F on HD, E. faecium VRE, CrCl 8 | High-dose daptomycin ≥8 mg/kg post-HD (superior microbiologic clearance vs linezolid in VRE BSI) | Britt NS et al. *Clin Infect Dis.* 2015;61(6):871–878. [PMC4551011](https://pubmed.ncbi.nlm.nih.gov/26021990/) | `daptomycin 8mg/kg IV post-HD` | ✅ 1.0 | ⚠ 0.5 | **0.939** |

> R2 = 0.5 on Case 3 is correct behaviour: IDSA formally lists linezolid as first-line for VRE; daptomycin is an evidence-supported alternative. The environment adheres to the guideline hierarchy, which is the point.

> **Reproduce:** `python eval_published_cases.py` — injects each case via `POST /reset`, runs the investigation sequence, commits the published recommendation, and scores it.

---

## Adversarial Stress Test (10 hand-crafted hard cases)

These cases are not in any training set; each is engineered to break a specific
baseline failure mode. MIC values are set to be unambiguous against EUCAST v16.0
breakpoints. *Trained* column links to the live HuggingFace Space where you can
inject any case and observe the model's prescription in real time.

**Pass threshold**: quality\_ratio >= 0.85 (near-optimal IDSA-concordant prescription).

**Note on R5**: Baselines make zero tool calls so R5=0. The trained model must beat
baselines on *both* R0-R4 (correct prescription) *and* R5 (systematic investigation).

| ID | Scenario | Best Drug | Broad-Empiric | Random (seed=42) | EUCAST-Only | Trained |
|----|----------|-----------|---------------|-----------------|-------------|---------|
| **A1** | VSE bacteremia + penicillin allergy | `vancomycin` | FAIL (0.00) | FAIL (0.00) | SUBOPT (0.78) | PASS (0.88) |
| **A2** | CRE K. pneumoniae bacteremia | `ceftazidime-avibactam` | FAIL (0.11) | FAIL (0.11) | PASS (1.00) | PASS (0.94) |
| **A3** | Susceptible E. coli UTI -- stewardship trap | `ceftriaxone` | SUBOPT (0.61) | SUBOPT (0.61) | SUBOPT (0.76) | PASS (0.96) |
| **A4** | MRSA pneumonia | `vancomycin` | FAIL (0.11) | PASS (1.00) | PASS (1.00) | PASS (0.92) |
| **A5** | CRE bacteremia + moderate-severe renal impairment (CrCl 25) | `ceftazidime-avibactam` | FAIL (0.11) | PASS (1.00) | PASS (1.00) | PASS (0.95) |
| **A6** | MDR Enterococcus bacteremia + dialysis (CrCl 8) | `daptomycin` | FAIL (0.11) | FAIL (0.11) | PASS (1.00) | PASS (0.91) |
| **A7** | XDR P. aeruginosa pneumonia -- last-line agent | `cefiderocol` | FAIL (0.11) | SUBOPT (0.76) | PASS (1.00) | PASS (0.97) |
| **A8** | MSSA bacteremia -- stewardship: cefazolin vs vancomycin | `cefazolin` | FAIL (0.11) | SUBOPT (0.64) | SUBOPT (0.81) | PASS (0.93) |
| **A9** | ESBL E. coli bacteremia -- carbapenem stewardship | `ertapenem` | SUBOPT (0.82) | FAIL (0.06) | PASS (1.00) | PASS (0.95) |
| **A10** | MDR E. coli CRE intra-abdominal infection | `ceftazidime-avibactam` | FAIL (0.11) | FAIL (0.11) | PASS (1.00) | PASS (0.96) |

> **Summary**: Broad-empiric 0/10 pass. Random(42) 2/10 pass. EUCAST-only 7/10 pass. Trained model: 10/10 pass.

> **Reproduce**: `python eval_adversarial.py --seed 42` — runs in under 10 seconds on CPU, no GPU required.

> **Why each case fails or passes:** [docs/Failure-Analysis.md](docs/Failure-Analysis.md) — per-case root cause analysis with R0–R5 breakdowns and actionable insights.

---

## Clinical Validation Matrix

Every reward component is traceable to a specific published clinical standard. See [docs/Clinical-Validation-Matrix.md](docs/Clinical-Validation-Matrix.md) for the full table of:

**Clinical Guideline → Env Requirement → Reward Component → Test Case(s)**

Quick summary:

| Reward Component | Clinical Standard | Verified In |
|---|---|---|
| R0 Allergy hard gate | Drug allergy avoidance; beta-lactam cross-reactivity | A1, `test_env.py` |
| R1 Microbiological activity | EUCAST Clinical Breakpoints v16.0 (2026) | A2–A10, P1–P3 |
| R2 Guideline concordance | IDSA Clinical Practice Guidelines 2022/2023 | All cases |
| R3 Stewardship | IDSA stewardship principles; WHO AMR Action Plan | A3, A9 |
| R4 Dose correctness | FDA labeling; pharmacokinetic references | A5, A6, P1–P3 |
| R5 Tool efficiency | Systematic investigation vs. empiric prescribing | All cases |

---

## The Problem

Antimicrobial resistance (AMR) kills **1.27 million people per year** and is projected to surpass cancer as a leading cause of death by 2050. A central driver is inappropriate antibiotic prescribing: wrong drug, wrong dose, or a broad-spectrum agent used when a narrow one would work. Antibiotic stewardship programs exist to fix this, but they are expensive, understaffed, and unavailable in most of the world.

AMR-Steward asks: can an LLM learn to prescribe correctly — not by memorizing guidelines, but by reasoning through resistance data, patient factors, and clinical evidence the way a trained physician would?

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

The agent is guided by an **I-JEPA world model** deployed inside the RL environment to rank tool calls by predicted information gain in latent space. See [docs/Architecture.md](docs/Architecture.md) for full architectural details.

---

## Curriculum

Training proceeds in three stages:

| Stage | Organisms | Renal function | Budget | Achieved reward |
|-------|-----------|---------------|--------|----------------|
| 1 | Susceptible only | Normal | 5 tools | 0.54 → **0.90** (peak 0.923, mean 0.84) |
| 2 | + Resistant (ESBL, MRSA, VRE) | Mild–moderate impairment | 4 tools | 0.86 → **0.84** (terminal mean 0.79) |
| 3 | + MDR (CRE, XDR Pseudomonas, VISA) | Severe impairment + allergies | 3 tools | 0.81 → **0.88** (peak 0.988, mean 0.71) |



## Results

GRPO training on `Qwen/Qwen3-4B` + LoRA (r=16) across three curriculum stages (A10G GPU via HF Spaces):

| Stage | Cases | Peak Reward | Final Reward | Mean |
|-------|-------|-------------|--------------|------|
| 1 — Susceptible | 128 | **0.923** | 0.900 | 0.840 |
| 2 — Resistant / MDR | 64 | **0.840** | 0.840 | 0.790 |
| 3 — MDR + Renal + Allergies | 32 | **0.988** | 0.880 | 0.707 |

Reward holds consistently above 0.70 even as case complexity scales from susceptible organisms to MDR + severe renal failure + allergy constraints.

**Training curves across all 3 stages:**

![Reward curves across curriculum stages](reward_curves.png)

**Baseline comparison and curriculum generalisation:**

![Training summary — improvement over random baseline](training_summary.png)

A perfect prescription (correct drug, first-line IDSA, narrowest spectrum, correct renal dose, full investigation) scores **1.0** (`quality_ratio = 1.0`). Random baseline: ~0.07. The trained model reaches **0.84–0.90** — **12× better than random** on Stage 1, reward holds above 0.70 on the hardest Stage 3 cases (MDR + severe renal failure + allergy constraints).

---

## Tests

```bash
pytest test_env.py test_jepa_integration.py -v
# 21 passed in ~5s (CPU, no GPU required)
```

| File | Tests | What's covered |
|------|-------|---------------|
| [`test_env.py`](test_env.py) | 8 | Reset, tool calls, correct/wrong prescription rewards, budget exhaustion, invalid action handling, state property, app import |
| [`test_jepa_integration.py`](test_jepa_integration.py) | 13 | JEPA info-gain bounds, dense reward cap, EMA world model loading, repeated-tool no-bonus rule, latent consistency bonus bounds, full episode accumulation, correct prescription reward with JEPA active |

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

**Why these pathogens specifically:** The environment covers the five bacteria designated as *critical priority* by the WHO Global Priority Pathogens List — *K. pneumoniae*, *E. coli*, *P. aeruginosa*, *S. aureus*, and *Enterococcus*. These five account for the overwhelming majority of drug-resistant infection deaths globally and are the primary targets of antibiotic stewardship programs worldwide. Scope is intentionally narrow and medically verified rather than broad and approximate — every breakpoint and guideline entry in the environment is traceable to a published EUCAST or IDSA source.



## Team

Built at a 24-hour hackathon, April 2026.

| Person | Role |
|--------|------|
| Saaheer | ML/RL — JEPA world model, reward functions, GRPO training |
| Divyansh | Backend — OpenEnv environment, FastAPI, HuggingFace deployment |
| Palak | Data — Medical data tables, patient cases, content |

---

*No real patient data was used. All patient cases are synthetically generated.*
