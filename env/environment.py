"""
env/environment.py — AMR-Steward OpenEnv environment.
Owns: Bhatia

Implements the three OpenEnv lifecycle methods:
  reset()  → AMRObservation
  step()   → (AMRObservation, reward, done)
  state()  → dict
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from .models import AMRAction, AMRObservation, PatientCase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BUDGET_BY_LEVEL: dict[int, int] = {1: 5, 2: 4, 3: 3}

# Resolve the repo-root data directory regardless of CWD.
_HERE = Path(__file__).parent.parent          # project root
DATA_DIR = _HERE / "data"


# ---------------------------------------------------------------------------
# Lazy data loaders (loaded once, shared across episodes)
# ---------------------------------------------------------------------------

def _load_json(filename: str) -> dict:
    path = DATA_DIR / filename
    if not path.exists():
        logger.warning("Data file not found: %s — using empty dict", path)
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_idsa: dict | None = None
_drug_properties: dict | None = None
_eucast: Any | None = None


def _get_idsa() -> dict:
    global _idsa
    if _idsa is None:
        raw = _load_json("idsa_guidelines.json")
        # Strip internal comment keys that start with "_"
        _idsa = {k: v for k, v in raw.items() if not k.startswith("_")}
    return _idsa


def _get_drug_properties() -> dict:
    global _drug_properties
    if _drug_properties is None:
        raw = _load_json("drug_properties.json")
        _drug_properties = {k: v for k, v in raw.items() if not k.startswith("_")}
    return _drug_properties


def _get_eucast():
    """Return the EUCAST parser module (lazy import to avoid circular deps)."""
    global _eucast
    if _eucast is None:
        import sys
        sys.path.insert(0, str(_HERE))
        from data.eucast_parser import classify_mic, is_susceptible
        _eucast = type("EucastParser", (), {
            "classify_mic": staticmethod(classify_mic),
            "is_susceptible": staticmethod(is_susceptible),
        })
    return _eucast


# ---------------------------------------------------------------------------
# Tool functions (Task B4)
# ---------------------------------------------------------------------------

def interpret_resistance(drug: str, patient: PatientCase, eucast) -> str:
    """
    Classify a drug's MIC from the patient's antibiogram using EUCAST.
    Returns a plain English result string.
    """
    mic = patient.antibiogram.get(drug)
    if mic is None:
        return (
            f"No MIC data available for {drug} in this antibiogram. "
            f"Available drugs: {', '.join(patient.antibiogram.keys()) or 'none'}."
        )
    classification = eucast.classify_mic(patient.organism, drug, mic)
    labels = {"S": "Susceptible", "I": "Intermediate", "R": "Resistant", "UNKNOWN": "UNKNOWN (no breakpoint)"}
    label = labels.get(classification, classification)
    return (
        f"{drug.capitalize()} MIC = {mic} mg/L -> EUCAST classification: {label} "
        f"(organism: {patient.organism})."
    )


def check_guideline(syndrome: str, patient: PatientCase, idsa: dict) -> str:
    """
    Look up the IDSA guideline recommendation for this syndrome + organism phenotype combo.
    Returns a plain English result string.
    """
    syndrome_data = idsa.get(syndrome)
    if syndrome_data is None:
        available = ", ".join(idsa.keys())
        return f"No IDSA data found for syndrome '{syndrome}'. Available syndromes: {available}."

    # Build organism key: try exact match, then phenotype-enriched keys.
    organism = patient.organism
    phenotype = patient.phenotype

    # Map patient organism + phenotype to IDSA key variants.
    candidate_keys = [
        organism,
        f"{organism} ({phenotype})",
        # Common clinical shorthand aliases
        f"{organism} (susceptible)",
        f"{organism} (MSSA)" if organism == "S. aureus" and phenotype == "susceptible" else None,
        f"{organism} (MRSA)" if organism == "S. aureus" and phenotype in ("resistant", "MDR") else None,
        f"{organism} (ESBL)" if phenotype == "resistant" else None,
        f"{organism} (CRE)" if phenotype in ("resistant", "MDR") else None,
        f"{organism} (VSE)" if organism == "Enterococcus" and phenotype == "susceptible" else None,
        f"{organism} (VRE)" if organism == "Enterococcus" and phenotype in ("resistant", "MDR") else None,
    ]
    candidate_keys = [k for k in candidate_keys if k is not None]

    rec = None
    matched_key = None
    for key in candidate_keys:
        if key in syndrome_data:
            rec = syndrome_data[key]
            matched_key = key
            break

    if rec is None:
        available_keys = ", ".join(syndrome_data.keys())
        return (
            f"No specific IDSA recommendation found for {organism} ({phenotype}) + {syndrome}. "
            f"Available entries for {syndrome}: {available_keys}."
        )

    alts = ", ".join(rec.get("alternatives", [])) or "none listed"
    return (
        f"IDSA recommendation for {matched_key} + {syndrome}:\n"
        f"  First-line: {rec['first_line']} — {rec.get('dose', 'dose not specified')} "
        f"for {rec.get('duration', 'duration not specified')}.\n"
        f"  Alternatives: {alts}.\n"
        f"  Notes: {rec.get('notes', 'none')}."
    )


def assess_patient_factors(patient: PatientCase, drug_properties: dict) -> str:
    """
    Check renal adjustment rules and allergy flags for all antibiogram drugs.
    Returns a plain English summary of relevant patient factor constraints.
    """
    crcl = patient.creatinine_clearance
    allergies = patient.allergies

    # Determine renal tier label
    if crcl >= 50:
        renal_tier = "CrCl_above_50"
        renal_label = "normal / mild impairment"
    elif crcl >= 30:
        renal_tier = "CrCl_30_50"
        renal_label = "moderate impairment"
    elif crcl >= 10:
        renal_tier = "CrCl_10_30"
        renal_label = "severe impairment"
    else:
        renal_tier = "CrCl_under_10"
        renal_label = "kidney failure / dialysis-range"

    lines = [
        f"Renal function: CrCl {crcl} mL/min ({renal_label}).",
        f"Allergies reported: {', '.join(allergies) if allergies else 'none'}.",
        "Renal dosing alerts for drugs in this antibiogram:",
    ]

    drugs_with_data = []
    for drug in patient.antibiogram:
        props = drug_properties.get(drug)
        if props is None:
            continue
        # Find the right renal-tier dose
        adj = props.get("renal_adjustments", {})
        # Try to find the appropriate tier — fall back to the closest bracket
        dose_at_tier = (
            adj.get(renal_tier)
            or adj.get("CrCl_above_50")
            or next(iter(adj.values()), "not specified")
        )

        # Allergy conflict check
        drug_allergy_flags: list[str] = props.get("allergy_flags", [])
        allergy_conflicts = [
            flag for flag in drug_allergy_flags
            if any(a.lower() in flag.lower() for a in allergies)
        ]

        line = f"  • {drug}: {dose_at_tier}"
        if allergy_conflicts:
            line += f" ⚠ ALLERGY FLAG: {', '.join(allergy_conflicts)}"
        drugs_with_data.append(line)

    if drugs_with_data:
        lines.extend(drugs_with_data)
    else:
        lines.append("  (No dosing data found for antibiogram drugs in drug_properties.json)")

    # Add broad allergy warning
    if allergies:
        lines.append(
            f"\nNote: Patient has documented allergy to {', '.join(allergies)}. "
            "Verify allergy history and cross-reactivity risk before prescribing."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Environment class (Task B3)
# ---------------------------------------------------------------------------

class AMREnvironment:
    """
    OpenEnv-compatible RL environment for antibiotic prescribing decisions.

    Episode lifecycle:
      1. reset(curriculum_level) → initial AMRObservation
      2. step(AMRAction) → (AMRObservation, reward, done)   [repeat]
      3. state() → full episode state dict (for logging / debugging)
    """

    def __init__(self) -> None:
        self.current_patient: PatientCase | None = None
        self.tool_results: list[str] = []
        self.budget_remaining: int = 5
        self.done: bool = False
        self.curriculum_level: int = 1
        self.episode_step: int = 0
        self._last_reward_breakdown: dict[str, float] | None = None

    # ------------------------------------------------------------------
    # Public OpenEnv API
    # ------------------------------------------------------------------

    def reset(self, curriculum_level: int = 1) -> AMRObservation:
        """Start a fresh episode. Sample a new patient from the generator."""
        if curriculum_level not in BUDGET_BY_LEVEL:
            raise ValueError(f"curriculum_level must be 1, 2, or 3. Got: {curriculum_level}")

        self.curriculum_level = curriculum_level
        self.budget_remaining = BUDGET_BY_LEVEL[curriculum_level]
        self.tool_results = []
        self.done = False
        self.episode_step = 0
        self._last_reward_breakdown = None

        self.current_patient = self._sample_patient(curriculum_level)
        logger.info(
            "Episode reset | level=%d | patient=%d%s %s | organism=%s | phenotype=%s | budget=%d",
            curriculum_level,
            self.current_patient.age,
            self.current_patient.sex,
            self.current_patient.infection_site,
            self.current_patient.organism,
            self.current_patient.phenotype,
            self.budget_remaining,
        )
        return self._build_observation()

    def step(self, action: AMRAction) -> tuple[AMRObservation, float, bool]:
        """
        Apply an action.

        Returns:
            (observation, reward, done)

        Raises:
            ValueError: if episode is already done or action is invalid.
        """
        if self.done:
            raise ValueError("Episode is already done. Call reset() first.")
        if self.current_patient is None:
            raise ValueError("No active episode. Call reset() first.")

        self.episode_step += 1
        reward = 0.0

        if action.action_type == "INVESTIGATE":
            reward, done = self._handle_investigate(action)
            self.done = done

        elif action.action_type == "COMMIT":
            reward, done = self._handle_commit(action)
            self.done = done

        else:
            raise ValueError(
                f"Unknown action_type '{action.action_type}'. Must be 'INVESTIGATE' or 'COMMIT'."
            )

        obs = self._build_observation()
        logger.info(
            "Step %d | action=%s | reward=%.4f | done=%s | budget=%d",
            self.episode_step,
            action.action_type,
            reward,
            self.done,
            self.budget_remaining,
        )
        return obs, reward, self.done

    def state(self) -> dict[str, Any]:
        """Return full serialized episode state. Useful for debugging and logging."""
        return {
            "patient": self.current_patient.__dict__ if self.current_patient else None,
            "tool_results": list(self.tool_results),
            "budget_remaining": self.budget_remaining,
            "done": self.done,
            "curriculum_level": self.curriculum_level,
            "episode_step": self.episode_step,
            "last_reward_breakdown": self._last_reward_breakdown,
        }

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_investigate(self, action: AMRAction) -> tuple[float, bool]:
        """Execute a tool call, append result, decrement budget."""
        result = self._execute_tool(action.tool_name, action.tool_arg)
        self.tool_results.append(result)
        self.budget_remaining -= 1

        logger.info(
            "Tool call | tool=%s | arg=%s | result_preview=%s",
            action.tool_name,
            action.tool_arg,
            result[:120].replace("\n", " "),
        )

        if self.budget_remaining <= 0:
            logger.warning("Budget exhausted without COMMIT — penalizing episode.")
            penalty = -0.1
            return penalty, True

        return 0.0, False

    def _handle_commit(self, action: AMRAction) -> tuple[float, bool]:
        """Compute reward from the committed prescription."""
        if not action.prescription:
            logger.warning("COMMIT action missing prescription — returning 0 reward.")
            return 0.0, True

        try:
            from env.reward import compute_total_reward  # Saaheer's file
            total, breakdown = compute_total_reward(
                prescription=action.prescription,
                patient=self.current_patient,
                tool_call_history=self.tool_results,
                eucast=_get_eucast(),
                idsa=_get_idsa(),
                drug_properties=_get_drug_properties(),
            )
            self._last_reward_breakdown = breakdown
            logger.info(
                "COMMIT | drug=%s | total_reward=%.4f | breakdown=%s",
                action.prescription.get("drug", "?"),
                total,
                breakdown,
            )
            return total, True
        except Exception as exc:
            # Graceful fallback so training never hard-crashes on reward errors.
            logger.error("Reward computation error: %s — returning 0.0 reward.", exc)
            self._last_reward_breakdown = {"error": str(exc)}
            return 0.0, True

    # ------------------------------------------------------------------
    # Tool dispatch (Task B4)
    # ------------------------------------------------------------------

    def _execute_tool(self, tool_name: str | None, tool_arg: str | None) -> str:
        """Route a tool call to the appropriate tool function."""
        if not tool_name:
            return "Error: tool_name is required for INVESTIGATE actions."

        patient = self.current_patient
        eucast = _get_eucast()
        idsa = _get_idsa()
        drug_properties = _get_drug_properties()

        if tool_name == "interpret_resistance":
            if not tool_arg:
                return "Error: tool_arg (drug name) is required for interpret_resistance."
            return interpret_resistance(drug=tool_arg, patient=patient, eucast=eucast)

        elif tool_name == "check_guideline":
            syndrome = tool_arg or patient.infection_site
            return check_guideline(syndrome=syndrome, patient=patient, idsa=idsa)

        elif tool_name == "assess_patient_factors":
            return assess_patient_factors(patient=patient, drug_properties=drug_properties)

        else:
            available = "interpret_resistance | check_guideline | assess_patient_factors"
            return f"Unknown tool '{tool_name}'. Available tools: {available}."

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(self) -> AMRObservation:
        """Construct the AMRObservation the agent sees after each action."""
        p = self.current_patient
        from data.patient_generator import patient_to_text  # type: ignore[import]
        try:
            patient_text = patient_to_text(p)
        except Exception:
            # Fallback text if patient_to_text is unavailable
            allergy_str = ", ".join(p.allergies) if p.allergies else "None reported"
            patient_text = (
                f"Patient: {p.age}-year-old {p.sex}.\n"
                f"Infection site: {p.infection_site}.\n"
                f"Culture result: {p.organism} isolated.\n"
                f"Renal function: CrCl {p.creatinine_clearance} mL/min.\n"
                f"Allergies: {allergy_str}.\n"
                f"Available antibiogram data: {list(p.antibiogram.keys())}.\n"
            )

        return AMRObservation(
            patient_text=patient_text,
            tool_results=list(self.tool_results),
            budget_remaining=self.budget_remaining,
            world_model_rankings="",   # Saaheer's enrich_observation() fills this in
            done=self.done,
        )

    # ------------------------------------------------------------------
    # Patient sampling
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_patient(curriculum_level: int) -> PatientCase:
        """Sample a patient from the generator. Falls back to a hardcoded demo case."""
        try:
            import sys
            repo_root = str(Path(__file__).parent.parent)
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            from data.patient_generator import generate_patient  # type: ignore[import]
            return generate_patient(curriculum_level)
        except Exception as exc:
            logger.warning(
                "patient_generator unavailable (%s) — using demo patient (67F CRE bacteremia).", exc
            )
            # Canonical demo patient from PROJECT_MASTER.md §10
            return PatientCase(
                age=67, sex="F",
                infection_site="bacteremia",
                organism="K. pneumoniae",
                creatinine_clearance=35.0,
                allergies=[],
                antibiogram={"meropenem": 8.0, "ceftazidime-avibactam": 1.0, "colistin": 1.0},
                phenotype="resistant",
                curriculum_level=curriculum_level,
            )
