# AMR-Steward — Session Changes

Changes made during the April 26 2026 hackathon session.

---

## Reward & Training Pipeline

### RLVR-Verifiable Terminal Reward (`env/reward.py`)
- Added `compute_optimal_prescription(patient)` — brute-forces all antibiogram drugs to find the maximum achievable process score (R1·0.4 + R2·0.25 + R3·0.15 + R4·0.1). This is the RLVR oracle that makes the terminal reward zero-variance and externally verifiable.
- `compute_total_reward` now computes `quality_ratio = min(1.0, agent_score / opt_score)` and `total = 0.90·quality_ratio + 0.10·R5`. Comparable to the MILP `clip(opt/agent, 0, 1)` pattern from operations research RL environments.
- `budget_remaining` and `budget_total` passed through to enable the structured R5.
- `R0_allergy_safety` hard gate added — allergy conflict immediately zeroes all reward components regardless of prescription quality.

### R5 Rewrite — Tool Efficiency (`env/reward.py`)
- `R5_tool_efficiency(unique_tool_types, budget_spent, budget_remaining, budget_total)` replaces keyword heuristics.
- Formula: `(unique_types / budget_spent) × (budget_remaining / budget_total)` — rewards diverse investigation with budget left. No text parsing, no keywords.
- `_infer_tool_type(text)` helper handles both env result strings and training JSON call lines in one function.

### Capped Dense Shaping (`env/environment.py`)
- INVESTIGATE steps earn `+0.04` for each novel tool type called, hard-capped at `+0.20` total per episode (`DENSE_CAP`).
- `_called_tools` set tracks tool+arg pairs to prevent farming dense reward from repeated identical calls.
- Cap keeps terminal signal dominant: a perfect prescription terminal (~0.80+) always exceeds the dense ceiling.

### Multi-Head GRPO (`train.py`)
- Three independent reward functions passed to `GRPOTrainer`:
  - **Head 1 (format)**: R6 format compliance × 0.05 — fast feedback on output structure.
  - **Head 2 (process)**: R5 tool efficiency — gradient signal during the investigation phase.
  - **Head 3 (terminal)**: cumulative env reward via `_score_completion_with_env` — multi-turn signal.

### Multi-Turn Env Rollout (`train.py`)
- `_score_completion_with_env(text, patient_payload, level)` replays all INVESTIGATE and COMMIT actions from a model completion through a fresh `AMREnvironment` instance using the exact patient case baked into the prompt.
- Returns cumulative reward: dense shaping from INVESTIGATE steps + terminal COMMIT reward (quality_ratio).
- Training Head 3 now fires the full env pipeline — budget enforcement, allergy gate, RLVR oracle — not just static text parsing.
- `_parse_investigate_action(line)` extracts `(tool_name, tool_arg)` from JSON INVESTIGATE payloads.
- `reset()` accepts an optional `patient: PatientCase` parameter so the training rollout can seed the env with the exact patient used to build the prompt.

---

## Bug Fixes

| Bug | Fix |
|-----|-----|
| `dict(CURRICULUM)[args.stage]` KeyError in `train.py` | Fixed to `CURRICULUM[args.stage - 1][0]` |
| `reward_range: [0.0, 1.0]` violated by `-0.1` budget penalty | Fixed to `[-0.1, 1.0]` in `openenv.yaml` |
| R2 / `check_guideline` key mismatch | Both now use identical candidate key resolution logic |
| `SUPPORTS_CONCURRENT_SESSIONS = False` (regression) | Restored to `True` |
| `last_reward_breakdown: Dict[str, float]` rejected error strings | Fixed to `Dict[str, Any]` in `env/models.py` |

---

## Evaluation Framework (`eval.py`)

New standalone evaluation script — no GPU required, runs in ~10 seconds.

Three deterministic baselines, each making zero tool calls:

| Policy | Description |
|--------|-------------|
| **Broad empiric** | Always meropenem — the lazy broad-spectrum default |
| **Random** | Uniform random drug from the antibiogram |
| **EUCAST-only** | Narrowest EUCAST-susceptible drug with correct renal dose — best achievable without IDSA guideline lookup |

Each baseline is scored by the same RLVR oracle (`compute_total_reward`) with full budget unused (R5=0). Results across all three curriculum levels show the learning headroom available to the RL agent.

Key numbers (n=100 per level, seed=42):

| Policy | Level 1 | Level 2 | Level 3 |
|--------|---------|---------|---------|
| Broad empiric (meropenem) | 0.466 | 0.351 | 0.209 |
| Random (antibiogram) | 0.750 | 0.567 | 0.544 |
| EUCAST-only (oracle access) | **0.992** | **0.904** | **0.804** |
| Trained Qwen3-0.6B (GRPO) | ~0.337 est | ~0.368 est | ~0.278 est |

The EUCAST-only policy represents an **oracle expert system** with direct API access to the same data tables used by the reward function — it is the ceiling. The LLM agent must discover this through natural language investigation, which is the actual RL challenge.

Automatically calls `viz.py` to generate comparison charts when matplotlib is available.

---

## Visualization (`viz.py`)

Extended with two new chart functions:

- **`plot_eval_comparison(eval_results_path)`** — grouped bar chart comparing quality_ratio across all baseline policies and curriculum levels → `assets/eval_comparison.png`
- **`plot_component_breakdown(eval_results_path, level)`** — horizontal bar chart of R1–R4 component means per policy → `assets/component_breakdown_level1.png`

New CLI flags:
```
python viz.py --eval eval_results.json              # eval comparison charts
python viz.py --eval eval_results.json --breakdown-level 2
```

Training curve plots (`plot_stages`, `plot_combined`) unchanged — silently skipped when checkpoint dir is absent.

---

## Generated Assets

| File | Description |
|------|-------------|
| `assets/eval_comparison.png` | Quality ratio bar chart by policy × level |
| `assets/component_breakdown_level1.png` | R1–R4 component breakdown at Level 1 |
| `eval_results.json` | Full eval output: per-policy per-level aggregated metrics |
