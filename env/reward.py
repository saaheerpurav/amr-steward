import re
import json
import os
from .models import PatientCase

# Spectrum score — lower = narrower spectrum (more targeted = better stewardship)
SPECTRUM_SCORE = {
    "nitrofurantoin": 1, "trimethoprim-sulfamethoxazole": 1, "ampicillin": 1,
    "cefazolin": 2, "ceftriaxone": 2, "oxacillin": 2, "nafcillin": 2,
    "vancomycin": 3, "daptomycin": 3, "ampicillin-sulbactam": 3,
    "piperacillin-tazobactam": 4, "ertapenem": 4, "cefepime": 4, "linezolid": 4,
    "meropenem": 5, "imipenem": 5,
    "ceftazidime-avibactam": 6, "meropenem-vaborbactam": 6,
    "colistin": 7, "tigecycline": 7,
}


def _load_drug_properties() -> dict:
    path = os.path.join(os.path.dirname(__file__), "..", "data", "drug_properties.json")
    with open(path) as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _load_idsa() -> dict:
    path = os.path.join(os.path.dirname(__file__), "..", "data", "idsa_guidelines.json")
    with open(path) as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


_DRUG_PROPS = None
_IDSA = None


def _get_drug_props():
    global _DRUG_PROPS
    if _DRUG_PROPS is None:
        _DRUG_PROPS = _load_drug_properties()
    return _DRUG_PROPS


def _get_idsa():
    global _IDSA
    if _IDSA is None:
        _IDSA = _load_idsa()
    return _IDSA


def _normalize_drug(drug: str) -> str:
    return drug.strip().lower().replace(" ", "-")


def _organism_to_idsa_key(organism: str, phenotype: str) -> str:
    """Map patient organism + phenotype to IDSA JSON key."""
    org_map = {
        "k. pneumoniae": "K. pneumoniae",
        "e. coli": "E. coli",
        "p. aeruginosa": "P. aeruginosa",
        "s. aureus": "S. aureus",
        "enterococcus": "Enterococcus",
    }
    phenotype_map = {
        "susceptible": "susceptible",
        "resistant": "resistant",
        "mdr": "MDR",
    }
    org = org_map.get(organism.lower().strip(), organism)
    pheno_label = phenotype_map.get(phenotype.lower(), phenotype)

    # Special cases
    if org == "K. pneumoniae" and pheno_label == "resistant":
        return "K. pneumoniae (CRE)"
    if org == "K. pneumoniae" and pheno_label == "susceptible":
        return "K. pneumoniae (susceptible)"
    if org == "K. pneumoniae" and pheno_label == "MDR":
        return "K. pneumoniae (CRE)"
    if org == "S. aureus" and pheno_label == "resistant":
        return "S. aureus (MRSA)"
    if org == "S. aureus" and pheno_label == "susceptible":
        return "S. aureus (MSSA)"
    if org == "Enterococcus" and pheno_label == "resistant":
        return "Enterococcus (VRE)"
    if org == "Enterococcus" and pheno_label == "susceptible":
        return "Enterococcus (VSE)"
    if org == "E. coli" and pheno_label == "resistant":
        return "E. coli (ESBL)"
    if org == "E. coli" and pheno_label == "susceptible":
        return "E. coli (susceptible)"
    if org == "P. aeruginosa":
        return "P. aeruginosa (susceptible)"

    return f"{org} ({pheno_label})"


def R1_microbiological_activity(prescription: dict, patient: PatientCase, eucast) -> float:
    """R1: Does the prescribed drug work against this bacteria?
    Looks up the MIC from the antibiogram and classifies via EUCAST.
    Returns 1.0 if Susceptible, 0.0 if Resistant/Intermediate/Unknown."""
    drug = _normalize_drug(prescription.get("drug", ""))
    antibiogram = patient.antibiogram

    # Find drug in antibiogram (flexible matching)
    mic = None
    for abx, mic_val in antibiogram.items():
        if _normalize_drug(abx) == drug:
            mic = mic_val
            break

    if mic is None:
        # Drug not tested — cannot confirm activity
        return 0.0

    classification = eucast.classify_mic(patient.organism, drug, mic)
    return 1.0 if classification == "S" else 0.0


def R2_guideline_concordance(prescription: dict, patient: PatientCase, idsa: dict | None = None) -> float:
    """R2: Is the prescribed drug the IDSA-recommended agent?
    Returns 1.0 for first-line, 0.5 for listed alternative, 0.0 otherwise."""
    if idsa is None:
        idsa = _get_idsa()
    drug = _normalize_drug(prescription.get("drug", ""))
    syndrome = patient.infection_site
    idsa_key = _organism_to_idsa_key(patient.organism, patient.phenotype)

    syndrome_data = idsa.get(syndrome, {})
    org_data = syndrome_data.get(idsa_key, {})

    if not org_data or "first_line" not in org_data:
        return 0.0

    first_line = _normalize_drug(org_data.get("first_line", ""))
    alternatives = [_normalize_drug(a) for a in org_data.get("alternatives", [])]

    if drug == first_line:
        return 1.0
    elif drug in alternatives:
        return 0.5
    return 0.0


def R3_stewardship(prescription: dict, patient: PatientCase, eucast, r1_score: float) -> float:
    """R3: Is this the narrowest-spectrum drug that still works?
    CONDITIONAL on R1 — returns 0.0 if R1 failed (drug doesn't even work).
    Returns 1.0 if optimal stewardship, graded down for broader choices."""
    if r1_score == 0.0:
        return 0.0

    drug = _normalize_drug(prescription.get("drug", ""))
    prescribed_score = SPECTRUM_SCORE.get(drug, 5)

    # Find all susceptible drugs from the antibiogram
    susceptible_drugs = []
    for abx, mic in patient.antibiogram.items():
        abx_norm = _normalize_drug(abx)
        if eucast.classify_mic(patient.organism, abx_norm, mic) == "S":
            susceptible_drugs.append(abx_norm)

    if not susceptible_drugs:
        return 1.0  # Only option, full stewardship score

    # Check allergy constraints
    drug_props = _get_drug_props()
    valid_options = []
    for abx in susceptible_drugs:
        props = drug_props.get(abx, {})
        allergy_flags = props.get("allergy_flags", [])
        patient_allergies = [a.lower() for a in patient.allergies]
        has_conflict = any(
            any(pa in flag for pa in patient_allergies)
            for flag in allergy_flags
        )
        if not has_conflict:
            valid_options.append(abx)

    if not valid_options:
        return 1.0

    min_spectrum = min(SPECTRUM_SCORE.get(abx, 5) for abx in valid_options)

    if prescribed_score == min_spectrum:
        return 1.0
    elif prescribed_score == min_spectrum + 1:
        return 0.5
    else:
        return max(0.0, 1.0 - 0.3 * (prescribed_score - min_spectrum))


def R4_dose_correctness(prescription: dict, patient: PatientCase, drug_properties: dict | None = None) -> float:
    """R4: Is the dose correct for this patient's renal function?
    Returns 1.0 if correct, 0.5 if within one tier, 0.0 if wrong."""
    drug = _normalize_drug(prescription.get("drug", ""))
    prescribed_dose = prescription.get("dose", "").lower().strip()
    crcl = patient.creatinine_clearance
    drug_props = drug_properties if drug_properties is not None else _get_drug_props()

    props = drug_props.get(drug, {})
    if not props or "renal_adjustments" not in props:
        return 0.5  # No data — neutral score

    adjustments = props["renal_adjustments"]

    # Determine expected dose tier based on CrCl
    if crcl > 50:
        expected = adjustments.get("CrCl_above_50", "")
    elif crcl >= 30:
        expected = adjustments.get("CrCl_30_50", "")
    elif crcl >= 10:
        expected = adjustments.get("CrCl_10_30", "")
    else:
        expected = adjustments.get("CrCl_under_10", "")

    if not expected:
        return 0.5

    expected_norm = expected.lower().strip()

    # Exact match
    if prescribed_dose == expected_norm:
        return 1.0

    # Fuzzy match — extract dose number and compare
    def extract_dose_mg(dose_str: str) -> float | None:
        match = re.search(r"([\d.]+)\s*g", dose_str)
        if match:
            return float(match.group(1)) * 1000
        match = re.search(r"([\d.]+)\s*mg", dose_str)
        if match:
            return float(match.group(1))
        return None

    prescribed_mg = extract_dose_mg(prescribed_dose)
    expected_mg = extract_dose_mg(expected_norm)

    if prescribed_mg and expected_mg:
        ratio = prescribed_mg / expected_mg
        if 0.9 <= ratio <= 1.1:
            return 1.0
        elif 0.7 <= ratio <= 1.3:
            return 0.5

    return 0.0


def R5_reasoning_grounding(tool_call_history: list[str], prescription: dict | None = None) -> float:
    """R5: Did the agent actually investigate before committing?
    Rewards systematic investigation. Penalizes blind guessing.
    Returns 0.0-1.0 based on quality of investigation."""
    if not tool_call_history:
        return 0.0

    score = 0.0
    history_text = " ".join(tool_call_history).lower()

    if "interpret_resistance" in history_text or "mic" in history_text or "eucast" in history_text:
        score += 0.5

    if "guideline" in history_text or "idsa" in history_text or "check_guideline" in history_text:
        score += 0.3

    if "assess_patient" in history_text or "crcl" in history_text or "renal" in history_text:
        score += 0.2

    return min(score, 1.0)


def compute_total_reward(
    prescription: dict,
    patient: PatientCase,
    tool_call_history: list[str],
    eucast,
    idsa: dict | None = None,
    drug_properties: dict | None = None,
) -> tuple[float, dict]:
    """Compute weighted total reward. Returns (total, breakdown dict).
    Weights: R1=40%, R2=25%, R3=15%, R4=10%, R5=10%"""
    r1 = R1_microbiological_activity(prescription, patient, eucast)
    r2 = R2_guideline_concordance(prescription, patient, idsa)
    r3 = R3_stewardship(prescription, patient, eucast, r1)
    r4 = R4_dose_correctness(prescription, patient, drug_properties)
    r5 = R5_reasoning_grounding(tool_call_history, prescription)

    total = round(0.40*r1 + 0.25*r2 + 0.15*r3 + 0.10*r4 + 0.10*r5, 4)

    breakdown = {
        "R1_activity": r1,
        "R2_guideline": r2,
        "R3_stewardship": r3,
        "R4_dose": r4,
        "R5_reasoning": r5,
        "total": total,
    }
    return total, breakdown


def parse_prescription_from_text(model_output: str) -> dict | None:
    """Parse a COMMIT action from model-generated text.
    Looks for JSON after 'COMMIT:' keyword."""
    match = re.search(r"COMMIT:\s*(\{.*?\})", model_output, re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        # Try to extract drug name at minimum
        drug_match = re.search(r'"drug"\s*:\s*"([^"]+)"', model_output, re.IGNORECASE)
        dose_match = re.search(r'"dose"\s*:\s*"([^"]+)"', model_output, re.IGNORECASE)
        if drug_match:
            return {
                "drug": drug_match.group(1),
                "dose": dose_match.group(1) if dose_match else "",
                "duration": "",
                "justification": "",
            }
    return None


def parse_tool_calls_from_text(model_output: str) -> list[str]:
    """Extract INVESTIGATE calls from model-generated text."""
    return re.findall(r"INVESTIGATE[:\s]+([^\n]+)", model_output, re.IGNORECASE)
