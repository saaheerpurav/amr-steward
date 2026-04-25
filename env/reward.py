# SAAHEER OWNS THIS FILE
# All 5 reward components as pure functions. No LLM judges.
# TODO Saaheer: implement each function below

from .models import PatientCase


def R1_microbiological_activity(prescription: dict, patient: PatientCase, eucast) -> float:
    """Does the prescribed drug actually work against this bacteria?
    Uses EUCAST MIC breakpoint lookup.
    Returns 1.0 if drug is Susceptible, 0.0 if Resistant or Intermediate."""
    # TODO Saaheer
    return 0.0


def R2_guideline_concordance(prescription: dict, patient: PatientCase, idsa: dict) -> float:
    """Is the prescribed drug the IDSA-recommended agent for this syndrome + organism?
    Returns 1.0 if first-line match, 0.5 if acceptable alternative, 0.0 otherwise."""
    # TODO Saaheer
    return 0.0


def R3_stewardship(prescription: dict, patient: PatientCase, eucast, r1_score: float) -> float:
    """Is this the narrowest-spectrum drug that still works?
    CONDITIONAL on R1 — returns 0.0 if R1 failed.
    Returns 1.0 if narrowest effective option chosen."""
    # TODO Saaheer
    if r1_score == 0.0:
        return 0.0
    return 0.0


def R4_dose_correctness(prescription: dict, patient: PatientCase, drug_properties: dict) -> float:
    """Is the dose correct for this patient's renal function and weight?
    Looks up adjustment rules in drug_properties.json.
    Returns 1.0 if correct, 0.5 if minor error, 0.0 if major error."""
    # TODO Saaheer
    return 0.0


def R5_reasoning_grounding(tool_call_history: list[str], prescription: dict) -> float:
    """Did the agent actually investigate before committing?
    Regex check: must have called interpret_resistance at minimum.
    Returns 1.0 if sufficient investigation, 0.0 if blind guess."""
    # TODO Saaheer
    if not tool_call_history:
        return 0.0
    return 0.0


def compute_total_reward(
    prescription: dict,
    patient: PatientCase,
    tool_call_history: list[str],
    eucast,
    idsa: dict,
    drug_properties: dict,
) -> tuple[float, dict]:
    """Compute weighted total reward. Returns (total, breakdown dict)."""
    r1 = R1_microbiological_activity(prescription, patient, eucast)
    r2 = R2_guideline_concordance(prescription, patient, idsa)
    r3 = R3_stewardship(prescription, patient, eucast, r1)
    r4 = R4_dose_correctness(prescription, patient, drug_properties)
    r5 = R5_reasoning_grounding(tool_call_history, prescription)

    total = 0.40*r1 + 0.25*r2 + 0.15*r3 + 0.10*r4 + 0.10*r5

    breakdown = {"R1_activity": r1, "R2_guideline": r2, "R3_stewardship": r3,
                 "R4_dose": r4, "R5_reasoning": r5, "total": total}
    return total, breakdown
