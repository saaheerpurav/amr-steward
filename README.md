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

> **TL;DR:** AMR-Steward is an OpenEnv reinforcement learning environment that trains an LLM to prescribe antibiotics correctly for drug-resistant infections. We bypassed the "LLM-as-a-judge" trap entirely by building a **fully deterministic, verifiable reward stack (RLVR)** based on EUCAST clinical breakpoints and IDSA guidelines. The headline innovation? We deployed **Meta's I-JEPA architecture** as a self-supervised world model *inside* the environment to rank tool calls by predicted information gain in latent space. The trained model reaches **0.84–0.90** across a 3-stage curriculum (Stage 1 peak 0.923, Stage 3 peak 0.988), passes **10/10** adversarial stress tests, and scores **10/10 vs 0/10 for broad-empiric prescribing** — the policy of defaulting to the broadest available antibiotic regardless of resistance data. 

### 🔗 Quick Links
- **Live Environment (HF Space):** [divyanshb06-amrsteward.hf.space](https://divyanshb06-amrsteward.hf.space)
- **Interactive Demo:** [/demo](https://divyanshb06-amrsteward.hf.space/demo) — Clinical clue cards with JEPA info-gain bars
- **Trained Model (HF Hub):** [saaheerpurav/amr-steward-model](https://huggingface.co/saaheerpurav/amr-steward-model)
- **Training Notebook (Colab):** [AMR_Steward.ipynb](https://colab.research.google.com/github/saaheerpurav/amr-steward/blob/main/AMR_Steward.ipynb)
- **Technical Blog:** [BLOG.md](BLOG.md)
- **Source Code:** [GitHub Repository](https://github.com/saaheerpurav/amr-steward)
- **Docs:** [Architecture](docs/Architecture.md) | [Reward Spec](docs/Reward-spec.md) | [Failure Analysis](docs/Failure-Analysis.md) | [Clinical Validation Matrix](docs/Clinical-Validation-Matrix.md)

---

## 📈 The Results: 10/10 Where Broad-Empiric Scores 0/10

We trained Qwen3-4B + LoRA using GRPO across three curriculum stages. The broad-empiric baseline — prescribing the broadest available antibiotic regardless of resistance data — passes **0 of 10** adversarial cases. The trained model passes **10 of 10**. On the hardest Stage 3 cases (MDR + severe renal failure + allergy constraints), broad-empiric scores 0.21. **The trained model scores 0.71–0.88.**

![Training summary — mean reward per stage vs random baseline](training_summary.png)

![Reward curves across curriculum stages — Stage 1: peak 0.923, Stage 2: peak shown as multi-head reward sum (3 GRPO heads combined; quality_ratio terminal mean 0.79), Stage 3: peak 0.988](reward_curves.png)

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

AMR is projected to surpass cancer as a leading cause of death by 2050. A central driver is inappropriate antibiotic prescribing: wrong drug, wrong dose, or a broad-spectrum agent used when a narrow one would work. Stewardship programs exist to fix this, but they are expensive, understaffed, and unavailable in most of the world.

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

## Reward Hacking Defenses

Four independent mechanisms prevent agents from gaming the reward signal:

| Vector | Defense | How it works |
|--------|---------|-------------|
| **Allergy bypass** | R0 hard gate | Any prescription the patient is allergic to → `total = 0.0, done = True` immediately. No partial credit. Checked before any other component. |
| **Dense reward farming** | Investigation cap | INVESTIGATE steps earn `+0.04` per novel `(tool, argument)` pair, **hard-capped at `+0.20` total per episode** (`DENSE_CAP`). An agent that only calls tools and never commits cannot exceed 0.20 — well below any meaningful terminal reward. |
| **Repeated tool calls** | `_called_tools` deduplication | `AMREnvironment._called_tools` tracks every `(tool_name, tool_arg)` pair seen. Calling the same tool with the same argument a second time earns **zero** dense bonus. Prevents reward farming via repetition. |
| **Stewardship gaming** | R3 gated on R1 | R3 (narrowest effective drug) only fires if R1 ≥ threshold — the drug must actually cover the organism. Prescribing a useless narrow-spectrum drug to game the stewardship score returns R3 = 0. |

The patient-specific `quality_ratio` oracle (`compute_optimal_prescription()`) brute-forces the ceiling at reset time — so the terminal signal is relative to what is *actually achievable* for this patient, not a fixed threshold that could be gamed with an easy case.

---

## JEPA World Model — Latent-Space Guidance System

AMR-Steward applies **Meta AI's Joint Embedding Predictive Architecture (JEPA)** — specifically the **I-JEPA pattern** with an EMA-stabilised target encoder — as a self-supervised world model for clinical state prediction. **To our knowledge this is the first JEPA-based world model deployed inside a clinical-domain RL environment**: the same SSL objective Meta uses for vision representation learning ([Assran et al., CVPR 2023](https://arxiv.org/abs/2301.08243)) is applied here to clinical `(state, tool, next_state)` prediction.

**What JEPA is — and is not**: JEPA does not fine-tune the LLM's weights. It is a **latent-space guidance system** — a compact self-supervised model (≈50K params) that acts as a *learned prior* over tool utility. It predicts, in embedding space, how much each available tool call would change the agent's known clinical state, flowing into training through three mechanisms: observation hints, JEPA-weighted reward shaping, and a latent consistency bonus.

**Training objective** (faithful to I-JEPA):

```
context_encoder(s_before) + tool → predictor → pred_repr
target_encoder(s_after)                      → tgt_repr   (EMA, stop-gradient)
Loss = MSE(pred_repr, tgt_repr)
```

**The EMA-stabilised target encoder** (τ = 0.99) is the critical anti-collapse mechanism: by updating the target encoder only as a slow EMA of the context encoder — never via backprop — the SSL objective remains non-trivial. Without EMA, the model collapses all representations to a constant, making information-gain scores meaningless. This is the "secret sauce" of JEPA that Meta engineers specifically look for.

**Honest scope**: ~50K parameters total — appropriate for a 64-dim handcrafted clinical state vector, not vision-scale. The contribution is *correct application of I-JEPA's SSL pattern to a new domain*, not a new architecture.

**Architecture:**
- `context_encoder`: 64-dim clinical state → 256 → 128-dim repr (ReLU MLP)
- `predictor`: 128-dim repr + 16-dim tool one-hot → 256 → 128-dim repr (ReLU MLP)
- `target_encoder`: EMA copy of `context_encoder` (τ = 0.99, frozen at inference)
- Pre-trained on 500 seeded synthetic episodes in [`jepa_pretrain.py`](jepa_pretrain.py) (~30s on CPU)
- Weights committed as [`jepa_weights.pt`](jepa_weights.pt) — env auto-loads on startup

**Three ways JEPA influences agent training:**

1. **Observation prior** — JEPA-ranked top-K tool predictions appended to every observation.
2. **JEPA-weighted reward shaping** — INVESTIGATE bonuses scaled by predicted info-gain (0.5×–1.5× multiplier). Agents that pick the world model's top-ranked tool earn a larger bonus; picking the lowest-ranked earns a reduced one.
3. **Latent consistency bonus** — actual state delta in target-encoder space (`‖tgt(s_after) − tgt(s_before)‖₂`) added as a curiosity signal after each INVESTIGATE step.

**Critical correctness detail** (the bug we did *not* ship):

```
gain(tool t) = ‖predictor(context(s), t) − target_encoder(s)‖ / √d_repr
```

Both `predictor(context(s), t)` and the anchor `target_encoder(s)` live in target-encoder space — so the training objective and the serving metric are computed in the same embedding geometry. This avoids the most common JEPA-deployment failure mode (anchoring against `context_encoder(s)` at serve time, which mismatches the SSL training loss).

**Inference output appended to every observation:**

```
PREDICTED INFORMATION GAIN:
- check_guideline_UTI: 0.1287
- assess_patient_factors: 0.0614
- interpret_resistance_ceftriaxone: 0.0599
- interpret_resistance_meropenem: 0.0388
```

**State vector** (64 dims, handcrafted): organism one-hot, resistance phenotype, infection site, normalised CrCl, allergy flags, tool-called flags, antibiogram presence indicators. Pretraining script [`jepa_pretrain.py`](jepa_pretrain.py) is fully seeded for reproducibility.

**Dense shaping** (JEPA-weighted): INVESTIGATE steps earn a base `+0.04` per novel `(tool, argument)` pair, scaled by the JEPA information-gain score and hard-capped at `+0.20` total so the terminal quality_ratio always dominates.

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

A perfect prescription (correct drug, first-line IDSA, narrowest spectrum, correct renal dose, full investigation) scores **1.0** (`quality_ratio = 1.0`). Broad-empiric prescribing scores **0.21** on the hardest cases. The trained model reaches **0.84–0.90** and holds above 0.70 even at Stage 3 (MDR + severe renal failure + allergy constraints).

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



## Judging Criteria

| Criterion | Weight | Evidence (with file references) |
|---|---|---|
| **Environment Innovation** | 40% | **First JEPA-based world model deployed inside a clinical-domain RL environment**: applies Meta AI's Joint Embedding Predictive Architecture (I-JEPA pattern, EMA-stabilised target encoder) to clinical `(state, tool, next_state)` prediction — see [`env/world_model.py`](env/world_model.py) and [`jepa_pretrain.py`](jepa_pretrain.py). Every observation served to the LLM contains JEPA-ranked tool calls by predicted information gain, computed in target-encoder space (matches the SSL training objective). Clinical AMR domain itself has zero prior RL environments. Quality-ratio oracle ([`env/reward.py`](env/reward.py) `compute_optimal_prescription`) brute-forces the optimal prescription at reset time, giving a patient-specific reward ceiling with zero variance. R0 hard allergy gate, R3 gated on R1, R5 diversity term — three independent anti-hacking layers. |
| **Storytelling** | 30% | 1.27 million deaths per year from antimicrobial resistance — more than HIV or malaria. Before/after is visceral: untrained model prescribes meropenem to a carbapenem-resistant organism (reward ~0.10, ineffective treatment); trained model investigates resistance, checks IDSA guidelines, adjusts for renal function, prescribes ceftazidime-avibactam at the correct renal dose (reward 0.84). Wrong drug → patient dies. Right drug → patient lives. Full narrative in [`BLOG.md`](BLOG.md). |
| **Showing Improvement** | 20% | GRPO training on Qwen3-4B + LoRA across three curriculum stages (A10G via HF Spaces). Stage 1: 0.54 → **0.90** (peak 0.923, mean 0.84). Stage 2: terminal mean **0.79**. Stage 3: 0.81 → **0.88** (peak 0.988, mean 0.71). Broad-empiric baseline: **0/10** adversarial cases passed, 0.21 on hardest cases. **Trained model: 10/10 adversarial cases passed.** Reward holds above 0.70 as complexity scales from susceptible organisms to MDR + renal failure + allergies. Training curves: [`reward_curves.png`](reward_curves.png), [`training_summary.png`](training_summary.png). Validated against published literature ([`eval_published_cases.py`](eval_published_cases.py)) and 10 adversarial cases ([`eval_adversarial.py`](eval_adversarial.py)) — see Clinical Validation and Adversarial Stress Test sections. |
| **Reward & Training Pipeline** | 10% | Multi-head GRPO: three independent reward functions (format R6, tool efficiency R5, terminal quality_ratio) give the trainer separate gradient channels at three timescales — see [`train.py`](train.py). Dense shaping (+0.04/unique tool call, capped +0.20) provides per-step signal without dominating the terminal reward. Seven reward components (R0–R6) in [`env/reward.py`](env/reward.py), all pure functions — no LLM judge anywhere in the pipeline. R5 computed from a structured `AMRState.tool_history` log ([`env/models.py`](env/models.py)), not text heuristics. |

---

## Team

Built at a 24-hour hackathon, April 2026.

| Person | Role |
|--------|------|
| Saaheer | ML/RL — JEPA world model, reward functions, GRPO training |
| Divyansh | Backend — OpenEnv environment, FastAPI, HuggingFace deployment |
| Palak | Data — Medical data tables, patient cases, content |

---

*No real patient data was used. All patient cases are synthetically generated.*
