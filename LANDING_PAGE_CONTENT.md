# AMR-Steward — Landing Page Content Brief

---

## Section 1 — Hero

**Primary headline:**
> 1.27 million people die every year from the wrong antibiotic.

**Sub-headline:**
> AMR-Steward is an AI that learned to prescribe the right one.

**One-liner below:**
> Trained with reinforcement learning against real EUCAST breakpoints and IDSA clinical guidelines. Guided by a JEPA world model. No guessing. No LLM judges. Pure medicine.

**Primary CTA:** Try It Now (links to the web app / interactive demo)
**Secondary CTA:** Watch Demo (links to the 2-min video)

---

## Section 2 — The Problem

**Section title:** The Silent Pandemic

**Body copy:**
Antimicrobial resistance (AMR) is already the third leading cause of death worldwide. By 2050, it is projected to kill more people than cancer.

The root cause is not lack of antibiotics — it is wrong antibiotics. Doctors prescribing broad-spectrum drugs when narrow ones would work. Meropenem prescribed to a carbapenem-resistant organism. The bacteria survives. The patient does not.

Antibiotic stewardship programs exist to fix this. They are expensive, understaffed, and unavailable in most of the world.

**Three stat callouts (large display numbers):**
- `1.27M` — deaths per year from AMR (more than HIV or malaria)
- `>50%` — of antibiotic prescriptions that are inappropriate or unnecessary
- `2050` — year AMR is projected to become the #1 cause of death globally

---

## Section 3 — The Before / After (Most Important Section)

**Section title:** Wrong Drug. Right Drug.

Show these two side by side as a case:

---

**The Patient:**
67-year-old female. ICU. Central-line bloodstream infection.
Culture: *Klebsiella pneumoniae* isolated from blood cultures ×2.
Renal function: CrCl 35 mL/min (moderate impairment).
Allergies: None.

---

**Without AMR-Steward (untrained model):**
Prescription: Meropenem 1g IV q8h
Why it fails: This organism is carbapenem-resistant. Meropenem has zero activity against it.
Outcome: Treatment failure. Patient deteriorates.
Reward score: **0.12 / 1.0**

**With AMR-Steward (trained model):**
The agent investigates first — guided by JEPA to pick the highest-value tool calls:
1. Checks meropenem MIC → Resistant (EUCAST)
2. Checks IDSA guideline for CRE bacteremia → First-line: ceftazidime-avibactam
3. Checks renal function → dose adjustment required at CrCl 35

Prescription: Ceftazidime-avibactam 2.5g IV q12h (renal-adjusted)
Outcome: Correct drug. Correct dose. Patient gets appropriate treatment.
Reward score: **0.91 / 1.0**

---

**Pull quote below the case:**
> The difference between 0.12 and 0.91 is the difference between a patient going home and a patient not surviving.

---

## Section 4 — JEPA: The World Model (Standalone Section)

**Section title:** The Agent Has a Mental Model of Medicine

**Opening line:**
Before calling a single diagnostic tool, AMR-Steward already knows which one is worth calling.

**Body copy:**
We built a **JEPA world model** — a Joint Embedding Predictive Architecture, the same class of architecture Meta AI uses for self-supervised visual learning — and pre-trained it on 500 synthetic clinical episodes.

The world model learns to predict: *if the agent calls this tool on this patient, how much will it change what the agent knows?* It ranks every available tool call by predicted information gain before the agent acts.

This means the agent doesn't investigate randomly. It doesn't waste its limited tool budget on resistance data it can already infer. It goes straight to the knowledge gap.

**What JEPA gives the agent (show as a live example):**

At the start of an episode on a CRE patient with unknown renal function:
```
PREDICTED INFORMATION GAIN:
  assess_patient_factors:              0.3841   ← highest — renal status unknown
  check_guideline_bacteremia:          0.2103
  interpret_resistance_meropenem:      0.1287
  interpret_resistance_ceftriaxone:    0.0599
```

The agent calls `assess_patient_factors` first — not because it was told to, but because JEPA predicted it would learn the most.

**Technical detail (for judges):**
- Pre-trained on 500 seeded (state, tool, next-state) triples from the same patient distribution used in RL training
- Architecture: context encoder → predictor → EMA target encoder (I-JEPA pattern)
- Information gain = L2 distance in target-encoder space between predicted next-state and current state
- Weights committed to repo as `jepa_weights.pt`, loaded at environment startup

**Pull quote:**
> This is not a lookup table pretending to reason. This is an agent with a learned model of what it doesn't know yet.

---

## Section 5 — How the Full System Works

**Section title:** Investigate. Reason. Commit.

Three steps, each as a distinct beat:

**Step 1 — Investigate**
The agent receives a patient case: organism, resistance profile, renal function, allergies, antibiogram. It calls up to N diagnostic tools before committing — checking MIC values against EUCAST breakpoints, querying IDSA clinical guidelines, assessing patient-specific dose adjustments. The JEPA world model ranks which tool to call next.

**Step 2 — Reason**
Seven reward components evaluate the final prescription — allergy safety, microbiological activity, guideline concordance, stewardship (narrowest effective drug), dose correctness, tool efficiency, and output format. Every component is a pure lookup function. No language model judges the output.

**Step 3 — Commit**
Drug. Dose. Duration. Justification. The prescription is scored against the patient-optimal ceiling computed by brute force at episode start. A score of 1.0 means the agent found the single best prescription for this patient.

**Small diagram / flow:**
`Patient case → JEPA ranks tools → Investigate (budget-limited) → Commit prescription → 7-component reward (0.0–1.0)`

---

## Section 6 — The Results

**Section title:** 12× Better Than Random. Across Every Complexity Level.

**Key result statement:**
The model was trained through three stages of increasing difficulty — susceptible organisms, drug-resistant strains, and fully drug-resistant MDR organisms with severe renal failure and allergy constraints. Final quality_ratio score held above 0.70 at every stage.

**Note on the metric:** All scores are `quality_ratio` — the agent's prescription quality divided by the brute-force optimal prescription for that patient (computed at reset). A score of 1.0 = perfect. Random prescribing = 0.05–0.10. The numbers below are where training started (epoch 1) → where it converged (final epoch).

**Results table:**

| Stage | Complexity | quality_ratio: start → final |
|-------|-----------|-------------|
| Stage 1 — Susceptible | Normal renal function, standard organisms | 0.55 → **0.84** |
| Stage 2 — Resistant | ESBL, MRSA, VRE + mild renal impairment | 0.40 → **0.79** |
| Stage 3 — MDR + Allergies | CRE, XDR Pseudomonas, severe renal failure, allergy constraints | Immediate **0.71** (zero warm-up — transferred from prior stages) |

**Headline number (large):**
> +0.65–0.79 improvement over random baseline (quality_ratio: 0.05–0.10 → 0.71–0.84)

**Embed:** `training_summary.png` — bar chart showing trained vs random per stage (12×/11×/10× labels)
**Embed:** `reward_curves.png` — learning curves across all 3 stages

---

## Section 7 — Try It

**Section title:** Prescribe. Investigate. Get Scored.

Three cards, one per interaction mode:

**Card 1 — Web App**
Play through a real clinical case in your browser. Investigate the resistance data. Query the IDSA guidelines. Commit a prescription. See your reward score and full breakdown instantly.
CTA: Open Web App

**Card 2 — WhatsApp**
Message our WhatsApp bot. Get a patient case. Investigate and prescribe over chat — just like a real clinical consult, on your phone.
CTA: Open WhatsApp

**Card 3 — Mobile App**
The full experience, on mobile. Patient cases, investigation tools, live reward feedback.
CTA: Open App

---

## Section 8 — The Technology

**Section title:** Built to Be Unhackable

**Stack line (show as badges/chips):**
OpenEnv · TRL GRPOTrainer · Qwen3-4B · LoRA · Unsloth (Colab) · HuggingFace Spaces · FastAPI

Four technical pillars, each in a tight paragraph:

**JEPA World Model**
Inspired by Meta AI's Joint Embedding Predictive Architecture. Pre-trained on 500 synthetic clinical episodes to predict information gain for every possible tool call. The agent investigates efficiently because it has a learned model of what it doesn't know — not because it was hand-programmed to follow a protocol.

**Verifiable Reward — No LLM Judges**
Seven reward components (R0–R6). Every single one is a pure lookup function against EUCAST breakpoints and IDSA clinical guidelines. R0 is a hard allergy gate — prescribe an allergen and the episode terminates at 0.0. No language model evaluating the output. No subjectivity anywhere in the pipeline.

**R0–R6 Reward Breakdown (show as a table):**

| Component | Role | What it measures |
|-----------|------|-----------------|
| R0 — Allergy Safety | Hard gate | Prescribing an allergen → total = 0.0 immediately |
| R1 — Microbiological Activity | Oracle | Does this drug cover the organism? (EUCAST MIC lookup) |
| R2 — Guideline Concordance | Oracle | Is this the IDSA-recommended agent? (1.0 = first-line, 0.5 = alternative) |
| R3 — Stewardship | Oracle | Is this the *narrowest* effective drug? Penalises unnecessary broad-spectrum use |
| R4 — Dose Correctness | Oracle | Is the dose appropriate for this patient's renal function? |
| R5 — Tool Efficiency | Process | `(unique tool types / budget spent) × (budget remaining / budget total)` |
| R6 — Output Format | Format | Clean single COMMIT line (1.0 for ≤3 lines, decays 0.05/line after) |

**Multi-Head GRPO Training**
Three independent reward functions give the TRL GRPOTrainer separate gradient signals at different timescales: output format R6 (fast feedback), tool efficiency R5 (dense), and terminal quality_ratio (sparse oracle). Trained on Qwen3-4B with LoRA (r=16) on an A10G GPU via HuggingFace Spaces. Fine-tuned locally in Colab with Unsloth for fast iteration.

**Curriculum Learning**
The agent starts on easy cases (susceptible organisms, normal renal function) and is progressively exposed to harder ones (MDR organisms, severe renal impairment, allergy conflicts). Complexity increases, investigation budget shrinks from 5 to 3, quality_ratio stays above 0.70 throughout.

---

## Section 9 — The Team

**Section title:** Built in 24 Hours

| Name | Role |
|------|------|
| Saaheer | ML/RL — JEPA world model, reward functions, GRPO training |
| Bhatia | Backend — OpenEnv environment, FastAPI, HuggingFace deployment |
| Palak | Data & Design — Medical data tables, patient cases, UI |

**Venue/event line:**
Built at a 24-hour hackathon, April 2026.

---

## Section 10 — Footer

**Links (all must be explicit URLs, no placeholders):**
- Live Environment — `https://saaheerpurav-amr-steward.hf.space`
- Interactive API Docs — `https://saaheerpurav-amr-steward.hf.space/docs`
- Trained Model (HF Hub) — `https://huggingface.co/saaheerpurav/amr-steward-model`
- Training Notebook (Colab) — `https://colab.research.google.com/github/saaheerpurav/amr-steward/blob/main/AMR_Steward.ipynb`
- GitHub — `https://github.com/saaheerpurav/amr-steward`

**Disclaimer (small):**
No real patient data was used. All patient cases are synthetically generated. This system is for research and demonstration purposes only and is not a clinical decision support tool.

---

## Tone Notes for Copy

- Visceral and direct. Not academic.
- Every stat is anchored to a human consequence — not just a number.
- The before/after case is the emotional core. It should hit hardest.
- JEPA gets its own section — it is the technical differentiator, not a footnote.
- Technical sections prove it works — they come after the story lands.
- Short sentences. Short paragraphs. No filler.
