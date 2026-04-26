# AMR-Steward — Teaching an LLM to Prescribe Antibiotics Correctly

*OpenEnv Hackathon submission writeup, April 2026.*

> Live environment: [divyanshb06-amrsteward.hf.space](https://divyanshb06-amrsteward.hf.space) · Trained model: [saaheerpurav/amr-steward-model](https://huggingface.co/saaheerpurav/amr-steward-model) · Code: [github.com/saaheerpurav/amr-steward](https://github.com/saaheerpurav/amr-steward) · Training notebook: [Colab](https://colab.research.google.com/github/saaheerpurav/amr-steward/blob/main/AMR_Steward.ipynb)

---

## TL;DR

We built an OpenEnv RL environment that teaches an LLM (Qwen3-4B + LoRA, GRPO) to prescribe the correct antibiotic for drug-resistant bacterial infections. **The agent is guided by a JEPA world model — the first time, to our knowledge, that Meta's Joint Embedding Predictive Architecture (I-JEPA pattern) has been deployed inside a clinical-domain RL environment** to rank tool calls by predicted information gain in embedding space. Every reward is computed from EUCAST clinical breakpoints and IDSA guideline tables — no LLM-as-judge anywhere. The trained model improves from a random-baseline reward of ~0.05 to **0.71–0.84** across three curriculum stages, and the env passes **3/3 published clinical cases** + **9/10 hand-crafted adversarial cases** designed to break specific baseline failure modes.

---

## 1. The Problem

Antimicrobial resistance kills **1.27 million people per year** — more than HIV or malaria combined — and the [WHO projects](https://www.who.int/news-room/fact-sheets/detail/antimicrobial-resistance) it could surpass cancer as a leading cause of death by 2050. The single biggest preventable driver is **inappropriate antibiotic prescribing**: wrong drug, wrong dose, or a broad-spectrum agent used when a narrow one would have worked.

Antibiotic stewardship programs exist to fix this, but they are expensive, understaffed, and unavailable in most of the world. So we asked: **can an LLM learn to prescribe correctly — not by memorising guidelines, but by reasoning through resistance data, patient factors, and clinical evidence the way a trained physician would?**

This is a perfect fit for **RL with verified rewards (RLVR)**. Unlike "is this poem good?", the question "did this prescription cover the bacteria, follow IDSA guidelines, use the narrowest active spectrum, and dose appropriately for renal function?" can be answered with deterministic lookup tables. No subjective judge needed.

---

## 2. Why an OpenEnv Fit

OpenEnv (Meta-PyTorch's Gymnasium-style RL framework) gave us four things that mattered:

1. **Structured `Action` + `Observation`** via Pydantic — no string-parsing the model output. The agent emits `{"action_type": "INVESTIGATE", "tool_name": "interpret_resistance", "tool_arg": "meropenem"}` and we validate it before stepping the env.
2. **`Environment` base class with `reset` / `step` / `state`** — drops directly into TRL's `GRPOTrainer` and any future Gymnasium-compatible algorithm.
3. **HTTP server + HuggingFace Spaces deployment** — the same env code runs locally for training and over HTTP for the live demo. Judges can `curl` it without cloning.
4. **State persistence via `SUPPORTS_CONCURRENT_SESSIONS`** — multiple agents can hit the env simultaneously without trampling each other's episodes.

The env is structured as:

```
env/
  models.py        AMRAction, AMRObservation, AMRState (Pydantic + dataclasses)
  environment.py   AMREnvironment(openenv.core.env_server.Environment)
  reward.py        R0-R6 pure-function reward components
  world_model.py   JEPA world model for tool-call ranking
data/
  eucast.csv               EUCAST v16.0 clinical breakpoints
  idsa_guidelines.json     IDSA 2022/2023 first-line + alternatives
  drug_properties.json     Renal adjustments, allergy flags, spectrum
```

---

## 3. Reward Design — The RLVR Stack

Seven independent reward components, all pure functions:

| | Component | What it measures | Range |
|---|---|---|---|
| R0 | Allergy safety | Hard gate — drug allergy → total = 0.0 | {0.0, 1.0} |
| R1 | Microbiologic activity | EUCAST classification of MIC vs the prescribed drug | {0.0, 1.0} |
| R2 | Guideline concordance | IDSA first-line=1.0, alternative=0.5, else 0.0 | {0.0, 0.5, 1.0} |
| R3 | Stewardship | Narrowest active drug given antibiogram + allergies | [0, 1] |
| R4 | Dose correctness | Matches renal-tier dose from drug_properties.json | [0, 1] |
| R5 | Tool efficiency | (unique_tool_types / spent) × (remaining / total) | [0, 1] |
| R6 | Output format | Single COMMIT line, ≤3 lines reasoning | [0, 1] |

The clever bit is **the quality ratio**:

```python
process_score = 0.40·R1 + 0.25·R2 + 0.15·R3 + 0.10·R4
opt_score     = compute_optimal_prescription(patient)   # brute-force over antibiogram
quality_ratio = min(1.0, process_score / opt_score)     # ∈ [0, 1]
total         = 0.90·quality_ratio + 0.10·R5
```

`compute_optimal_prescription` iterates the entire antibiogram, builds the IDSA-recommended dose at the patient's CrCl tier, and returns the maximum process score this *specific* patient could possibly achieve. So the reward ceiling is patient-specific (not a global constant), and `quality_ratio = 1.0` if and only if the agent found the IDSA-first-line drug at the correct dose. This is what makes the reward **RLVR-verifiable** rather than just a fixed-target heuristic.

**Three independent anti-hacking layers**:
- R0 is a hard gate — allergy violation zeros the entire reward, regardless of how brilliant R1–R5 are.
- R3 is gated on R1 — you can't get stewardship credit for prescribing a narrow drug that doesn't work.
- R5's diversity term penalises repeated calls to the same tool, even with different arguments.

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

AMR-Steward applies **Meta AI's Joint Embedding Predictive Architecture (JEPA)** — specifically the **I-JEPA pattern** with an EMA-stabilised target encoder — as a self-supervised world model that acts as a *learned prior* over tool utility. **To our knowledge this is the first JEPA-based world model deployed inside a clinical-domain RL environment.** The same SSL objective Meta uses for vision representation learning ([Assran et al., CVPR 2023](https://arxiv.org/abs/2301.08243)) is applied here to clinical `(state, tool, next_state)` prediction.

**What JEPA is — and is not**: JEPA does **not** fine-tune the LLM's LoRA weights. It is a **latent-space guidance system** — a compact self-supervised model (≈50K params) that predicts, in embedding space, how each available tool call would change the agent's known clinical state. These predictions flow into the RL training loop through three concrete mechanisms: observation hints, JEPA-weighted reward shaping, and a latent consistency bonus.

**Honest scope**: ~50K parameters total — appropriate for a 64-dim handcrafted clinical state vector, not vision-scale. The contribution is *correct application of I-JEPA's SSL pattern to a new domain*, not a new architecture.

We pretrain on synthetic `(state, tool, next_state)` triples drawn from 500 seeded episodes from the same patient distribution as RL training.

**Training objective** (faithful to I-JEPA):

```
ctx_repr  = context_encoder(s_before)         # 64-dim state → 128-dim repr
pred_repr = predictor(concat(ctx_repr, tool)) # predict next-state representation
tgt_repr  = target_encoder(s_after)           # EMA-stabilised target (stop-gradient)

Loss = MSE(pred_repr, tgt_repr)               # trained on synthetic rollouts
```

**The EMA target encoder — the anti-collapse mechanism**: The `target_encoder` is updated only as a slow EMA of `context_encoder` (decay τ = 0.99) — never via backpropagation:

```
θ_target ← τ · θ_target + (1 − τ) · θ_context
```

Without this, the model trivially minimises the JEPA loss by collapsing all representations to a constant (`pred_repr ≈ tgt_repr ≈ 0`). EMA stabilisation keeps the target distribution slowly but continuously moving, forcing the predictor to learn genuinely informative representations. This is what Meta engineers specifically look for in a correct JEPA implementation — and a common place where ports fail.

**At inference**, the world model ranks every available tool by predicted information gain in target-encoder space:

```
gain(tool t) = ‖predictor(context_encoder(s), t) − target_encoder(s)‖₂ / √d_repr
```

Both operands live in **target-encoder space** — matching the SSL training geometry exactly. Anchoring against `context_encoder(s)` at serve time (the common mistake) would mismatch the training objective and produce uncalibrated scores.

**Three ways JEPA acts as a learned prior during RL training:**

1. **Observation prior** — The top-K ranked tools by predicted gain are appended to every observation served to the LLM, providing a data-driven signal for investigation order without hard constraints.

2. **JEPA-weighted reward shaping** — INVESTIGATE step bonuses are scaled by the predicted information-gain score (0.5×–1.5× multiplier on the `+0.04` dense bonus). Agents that pick the world model's highest-predicted tool earn a larger bonus; picking the lowest-gain tool earns a reduced one. This directly ties the reward signal to JEPA's latent-space predictions — the agent is not merely *reading* JEPA's suggestions, it is *rewarded* for following them.

3. **Latent state consistency** — After each INVESTIGATE step, the actual state delta in target-encoder space is measured: `‖target_encoder(s_after) − target_encoder(s_before)‖₂`. A small curiosity bonus proportional to this delta rewards tool calls that revealed genuinely surprising (high-information) state changes, even if they weren't JEPA's top prediction.

---

## 6. Curriculum & Training Results

Three stages on Qwen3-4B + LoRA r=16 (A10G GPU via HF Spaces):

| Stage | Cases | Organisms | Renal | Budget | Result |
|-------|-------|-----------|-------|--------|--------|
| 1 | 128 | Susceptible only | Normal | 5 tools | 0.55 → **0.84** |
| 2 | 64 | + ESBL, MRSA, VRE | Mild–moderate | 4 tools | 0.40 → **0.79** |
| 3 | 32 | + CRE, XDR, VISA | Severe + allergies | 3 tools | **0.71** stable |

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
| Oracle (IDSA first-line at correct dose) | **9/10** |

Broad-empiric fails 0/10 because meropenem doesn't cover MRSA, VRE, or Enterococcus, has no breakpoint for several organism+drug pairs, and over-broadens stewardship for susceptible organisms. EUCAST-only passes 7/10 — it gets resistance + allergies right but lacks IDSA guideline knowledge to break ties (e.g. cefazolin vs oxacillin for MSSA).

The trained model must beat EUCAST-only by *also* nailing R5 (systematic investigation), not just R0–R4. See the live HF Space to test it on any of the 10 cases.

Reproduce: `python eval_adversarial.py --seed 42` (under 10 seconds on CPU).

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

*No real patient data was used. All `PatientCase` objects are synthetically generated from realistic clinical distributions. AMR-Steward is a research artefact and is not approved for clinical use.*
