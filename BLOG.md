# AMR-Steward — Teaching an LLM to Prescribe Antibiotics Correctly

*OpenEnv Hackathon submission writeup, April 2026.*

> Live environment: [divyanshb06-amrsteward.hf.space](https://divyanshb06-amrsteward.hf.space) · Trained model: [saaheerpurav/amr-steward-model](https://huggingface.co/saaheerpurav/amr-steward-model) · Code: [github.com/saaheerpurav/amr-steward](https://github.com/saaheerpurav/amr-steward) · Training notebook: [Colab](https://colab.research.google.com/github/saaheerpurav/amr-steward/blob/main/AMR_Steward.ipynb)

---

## TL;DR

We built an OpenEnv RL environment that teaches an LLM (Qwen3-4B + LoRA, GRPO) to prescribe the correct antibiotic for drug-resistant bacterial infections. **The agent is guided by a JEPA world model — the first time, to our knowledge, that Meta's Joint Embedding Predictive Architecture (I-JEPA pattern) has been deployed inside a clinical-domain RL environment** to rank tool calls by predicted information gain in embedding space. The JEPA world model's predictions are now visible in the [interactive demo](https://divyanshb06-amrsteward.hf.space/demo) as "Predicted Information Gain" bars on each clue card. Every reward is computed from EUCAST clinical breakpoints and IDSA guideline tables — no LLM-as-judge anywhere. The trained model reaches **0.84–0.90** across three curriculum stages (Stage 1 peak: 0.923, Stage 3 peak: 0.988), passes **3/3 published clinical cases** + **10/10 hand-crafted adversarial cases**, and scores **10/10 where broad-empiric prescribing scores 0/10** — the failure mode that kills patients in the real world. Full failure mode analysis: [docs/Failure-Analysis.md](docs/Failure-Analysis.md).

---

## 1. The Problem

Antimicrobial resistance (AMR) kills **1.27 million people per year**—more than HIV or malaria. A massive driver of this is **inappropriate antibiotic prescribing**: wrong drug, wrong dose, or dropping a broad-spectrum "nuke" when a targeted agent would work.

Antibiotic stewardship programs exist to fix this, but require scarce human experts. We asked: **can an LLM learn to prescribe correctly by reasoning through resistance data and clinical evidence like a physician?**

This is a perfect fit for **RL with verified rewards (RLVR)**. We can deterministically verify if a prescription covered the bacteria, followed IDSA guidelines, used the narrowest spectrum, and dosed correctly. No subjective judge needed.

---

## 2. Why an OpenEnv Fit

OpenEnv gave us four critical things out of the box:

1. **Structured `Action` + `Observation`** via Pydantic—no messy string-parsing.
2. **`Environment` base class**—drops directly into TRL's `GRPOTrainer`.
3. **HTTP server + HF Spaces**—same code runs locally for training and over HTTP for the live demo.
4. **State persistence**—multiple concurrent training episodes without trampling state.

---

## 3. Reward Design — The RLVR Stack

We built seven independent reward components, all pure functions:

| | Component | What it measures | Range |
|---|---|---|---|
| R0 | Allergy safety | Hard gate — drug allergy → total = 0.0 | {0.0, 1.0} |
| R1 | Microbiologic activity | EUCAST classification of MIC vs the prescribed drug | {0.0, 1.0} |
| R2 | Guideline concordance | IDSA first-line=1.0, alternative=0.5, else 0.0 | {0.0, 0.5, 1.0} |
| R3 | Stewardship | Narrowest active drug given antibiogram + allergies | [0, 1] |
| R4 | Dose correctness | Matches renal-tier dose from drug_properties.json | [0, 1] |
| R5 | Tool efficiency | (unique_tool_types / spent) × (remaining / total) | [0, 1] |
| R6 | Output format | Single COMMIT line, ≤3 lines reasoning | [0, 1] |

**The Quality Ratio:**

```python
process_score = 0.40·R1 + 0.25·R2 + 0.15·R3 + 0.10·R4
opt_score     = compute_optimal_prescription(patient)   # brute-force over antibiogram
quality_ratio = min(1.0, process_score / opt_score)     # ∈ [0, 1]
total         = 0.90·quality_ratio + 0.10·R5
```

Our oracle (`compute_optimal_prescription`) calculates the maximum score this *specific* patient could possibly achieve. The reward ceiling is patient-specific, making the reward truly **RLVR-verifiable**.

**Anti-hacking layers**:
- R0 is a hard gate—allergy violation zeros the entire reward.
- R3 is gated on R1—no stewardship credit for inactive drugs.
- R5 penalizes repeated calls to the same tool.

---

## 4. Multi-Head GRPO — Three Gradient Channels

Single-reward GRPO is brittle on long-horizon tasks because the terminal reward arrives once, late, and noisily. We pass *three independent reward functions* into `GRPOTrainer.reward_funcs`:

1. **Format head (R6)** — fast feedback. Decays 0.05 per extra line beyond 3, so the model learns to be concise within ~50 steps.
2. **Process head (R5 + dense shaping)** — per-step signal during the investigation phase. Each unique `(tool, argument)` pair earns +0.04, capped at +0.20 to prevent reward hacking.
3. **Terminal head (quality_ratio)** — the RLVR oracle score. Sparse but verifiable.

Each head provides a different learning signal at a different timescale. The format head converges first (clean output by step ~50). The process head shapes investigation behaviour next (the model learns to call all three tool types before committing). The terminal head dominates training rewards once R1+R2 alignment kicks in.

---

## 5. JEPA World Model — Latent-Space Guidance System

This is the headline ML contribution of the project.

AMR-Steward applies **Meta AI's Joint Embedding Predictive Architecture (JEPA)** — specifically the **I-JEPA pattern** — as a self-supervised world model to act as a *learned prior* over tool utility. **To our knowledge, this is the first JEPA-based world model deployed inside a clinical RL environment.**

**What JEPA is**: It is a compact self-supervised model (≈50K params) that predicts, in embedding space, how each available tool call will change the known clinical state. These predictions flow into the RL training loop via observation hints, reward shaping, and latent consistency bonuses.

**Training objective** (faithful to I-JEPA):

```
ctx_repr  = context_encoder(s_before)         
pred_repr = predictor(concat(ctx_repr, tool)) 
tgt_repr  = target_encoder(s_after)           # EMA-stabilised target (stop-gradient)

Loss = MSE(pred_repr, tgt_repr)               
```

**The EMA target encoder**: The `target_encoder` is updated only as a slow EMA of `context_encoder` (decay τ = 0.99) — never via backpropagation. Without this, the model collapses all representations to a constant. EMA stabilization is the "secret sauce" of JEPA.

**At inference**, the world model ranks tools by predicted information gain in target-encoder space:

```
gain(tool t) = ‖predictor(context_encoder(s), t) − target_encoder(s)‖₂ / √d_repr
```

Both operands live in **target-encoder space** — matching the SSL training geometry exactly.

**Three ways JEPA acts as a learned prior during RL training:**

1. **Observation prior** — Top-K tools appended to observations.
2. **Reward shaping** — INVESTIGATE step bonuses are scaled (0.5×–1.5×) by predicted information-gain. The agent is *rewarded* for making the right investigation step.
3. **Latent state consistency** — A curiosity bonus rewards tool calls that genuinely surprise the model (high actual state delta in target-encoder space).

---

## 6. Curriculum & Training Results

Three stages on Qwen3-4B + LoRA r=16 (A10G GPU via HF Spaces):

| Stage | Cases | Organisms | Renal | Budget | Result |
|-------|-------|-----------|-------|--------|--------|
| 1 | 128 | Susceptible only | Normal | 5 tools | 0.54 → **0.90** (peak 0.923, mean 0.84) |
| 2 | 64 | + ESBL, MRSA, VRE | Mild–moderate | 4 tools | 0.86 → **0.84** (terminal mean 0.79) |
| 3 | 32 | + CRE, XDR, VISA | Severe + allergies | 3 tools | 0.81 → **0.88** (peak 0.988, mean 0.71) |

The reward holds **above 0.70 across all three stages** even as case complexity scales from "susceptible E. coli, normal renal function, no allergies" to "MDR Pseudomonas, CrCl 25, penicillin allergy, 3-tool budget".

![Reward curves across all three curriculum stages](reward_curves.png)

![Training summary — improvement over random baseline](training_summary.png)

---

## 7. Validation — The Killer Slides

Two complementary evaluation suites prove the env is well-calibrated and the model is clinically credible.

### 7.1 Published clinical cases (3 cases, peer-reviewed literature)

We took three real cases from peer-reviewed papers, encoded them as `PatientCase` objects, and ran the env's reward stack against the *expert published recommendation*:

| Case | Citation | Expert prescription | Quality |
|------|----------|---------------------|---------|
| CRE bacteremia, post-renal-transplant | Tamma PD et al. *Clin Infect Dis.* 2023 ([PMC9890506](https://pubmed.ncbi.nlm.nih.gov/36462428/)) | Ceftazidime-avibactam 1.25g IV q8h | **1.000** |
| MSSA bacteremia | Maraolo AE et al. *Open Forum Infect Dis.* 2018 ([doi](https://doi.org/10.1093/ofid/ofy042)) | Cefazolin 2g IV q8h | **1.000** |
| VRE on hemodialysis | Britt NS et al. *Clin Infect Dis.* 2015 ([PMC4551011](https://pubmed.ncbi.nlm.nih.gov/26021990/)) | Daptomycin 8mg/kg post-HD | **0.939** |

The 0.939 on Case 3 is *correct behaviour*: IDSA formally lists linezolid as first-line for VRE; daptomycin is an evidence-supported alternative (R2=0.5). The env adheres to the guideline hierarchy, which is the point.

Reproduce: `python eval_published_cases.py`.

### 7.2 Adversarial stress test (10 hand-crafted hard cases)

Each case is engineered to break a specific baseline policy in a predictable way. MIC values are set to be unambiguous against EUCAST v16.0 breakpoints.

| Policy | Pass rate (quality_ratio ≥ 0.85) |
|--------|----------------------------------|
| Broad-empiric (always meropenem) | **0/10** |
| Random (seed=42) | **2/10** |
| EUCAST-only (antibiogram + allergy aware, no IDSA) | **7/10** |
| **Trained model** | **10/10** |

Broad-empiric fails 0/10 because meropenem doesn't cover MRSA, VRE, or Enterococcus, has no breakpoint for several organism+drug pairs, and over-broadens stewardship for susceptible organisms. EUCAST-only passes 7/10 — it gets resistance + allergies right but lacks IDSA guideline knowledge to break ties (e.g. cefazolin vs oxacillin for MSSA).

The trained model must beat EUCAST-only by *also* nailing R5 (systematic investigation), not just R0–R4. See the live HF Space to test it on any of the 10 cases.

Reproduce: `python eval_adversarial.py --seed 42` (under 10 seconds on CPU).

### 7.3 Failure Mode Analysis — Why Each Case Fails

Understanding *why* a policy fails is more useful than knowing *that* it fails. Here's the root cause for the most interesting failures from `adversarial_results.json`:

| Case | Policy | Root Cause | Key Component |
|------|--------|-----------|---------------|
| **A1** VSE + penicillin allergy | Broad-empiric (0.00) | Meropenem's beta-lactam allergy flag matches documented penicillin allergy → R0 fires, total = 0 | R0 hard gate |
| **A3** E.coli UTI, stewardship trap | Broad-empiric (0.55) | Meropenem *works* (R1=1.0) but IDSA first-line for susceptible E.coli UTI is ceftriaxone → R2=0, R3=0.1 | R2 guideline |
| **A8** MSSA bacteremia | Broad-empiric (0.10) | Meropenem has no EUCAST breakpoint for S. aureus → classified UNKNOWN → R1=0 | R1 activity |
| **A6** VRE on dialysis | Random (0.10) | Random picks vancomycin, which is the standard drug for Enterococcus — but VRE MIC=32, fully resistant → R1=0 | R1 activity |
| **A9** ESBL bacteremia | Random (0.05) | Random picks ceftriaxone; ESBL hydrolyzes all 3rd-gen cephalosporins → R1=0 | R1 activity |

**The pattern**: 4 of 5 failures are R1=0 — the drug is inactive. This is the most catastrophic failure mode (wrong drug = patient dies). The environment is correctly sensitive to this. An agent that systematically calls `interpret_resistance()` before committing will catch all of these.

Full per-case analysis with R0–R5 breakdowns: [docs/Failure-Analysis.md](docs/Failure-Analysis.md).

---

## 8. What We Got Right

- **Pure-function rewards** — every component is a deterministic lookup. No LLM-as-judge means no judge instability and no reward gaming via politeness or verbosity.
- **Patient-specific reward ceiling** — `compute_optimal_prescription` brute-forces the optimum at episode start, so quality_ratio is a true `[0, 1]` regardless of how easy or hard the patient is.
- **Two layers of validation** — published cases (clinical credibility) + adversarial cases (env calibration) cover both "does it match real expert decisions?" and "does it differentiate good from bad prescribing?".
- **Multi-head GRPO** — three independent gradient channels at three timescales avoided the "stuck at chance for the first 100 steps" failure mode of single-reward GRPO.
- **JEPA architecture consistency** — anchoring against `target_encoder(s)` rather than `context_encoder(s)` at inference matches the training objective. Easy bug to make; we caught it during the Tier-S review pass.

## 9. What We'd Add Given More Time

- **Polymicrobial cases** — currently single-organism. Real ICU patients often have 2–3 pathogens.
- **Combination therapy** — env scores single-drug prescriptions. Endocarditis and severe MDR cases need combos.
- **Allergy nuance** — current R0 fires on substring match (`penicillin` → `penicillin_cross_reactivity_low_risk`). A graded R0 with cross-reactivity weights would be more clinically realistic.
- **Vancomycin AUC/MIC dosing** — currently treated as a fixed dose tier; real-world dosing requires therapeutic drug monitoring math.

---

## 10. Reproducing the Submission

```bash
# 1. Clone
git clone https://github.com/saaheerpurav/amr-steward
cd amr-steward

# 2. Install
pip install -r requirements.txt

# 3. Run baseline + adversarial eval (no GPU, ~30 seconds total)
python eval.py
python eval_published_cases.py
python eval_adversarial.py --seed 42

# 4. Spin up the env locally
uvicorn app:app --port 7860
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"curriculum_level": 1}'

# 5. Re-train (GPU required; A10G recommended)
# Open AMR_Steward.ipynb in Colab and run all cells.
```

---

*AMR-Steward is a research artefact and is not approved for clinical use.*
