# PALAK OWNS THIS FILE
# Generates synthetic patient cases for RL training episodes.
# TODO Palak: fill in the distributions to make cases more realistic.

import random
from sys import path
path.insert(0, ".")
from env.models import PatientCase


ORGANISMS = ["K. pneumoniae", "E. coli", "P. aeruginosa", "S. aureus", "Enterococcus"]
INFECTION_SITES = ["bacteremia", "UTI", "pneumonia", "intra-abdominal"]

# MIC value ranges per organism + phenotype (Palak: adjust to be more realistic)
MIC_RANGES = {
    ("K. pneumoniae", "susceptible"): {
        "meropenem": (0.06, 1.0), "ceftriaxone": (0.03, 0.5),
        "piperacillin-tazobactam": (2.0, 8.0), "ertapenem": (0.03, 0.5),
        "ceftazidime-avibactam": (0.12, 2.0), "colistin": (0.5, 1.0),
    },
    ("K. pneumoniae", "resistant"): {   # CRE / KPC
        "meropenem": (8.0, 32.0), "ceftriaxone": (32.0, 128.0),
        "ceftazidime-avibactam": (0.5, 2.0), "meropenem-vaborbactam": (0.5, 2.0),
        "colistin": (0.5, 2.0), "tigecycline": (0.5, 2.0),
    },
    ("K. pneumoniae", "MDR"): {         # pan-resistant / MBL
        "meropenem": (16.0, 64.0), "ceftriaxone": (64.0, 256.0),
        "ceftazidime-avibactam": (16.0, 64.0), "meropenem-vaborbactam": (8.0, 32.0),
        "colistin": (0.5, 4.0), "tigecycline": (1.0, 4.0),
    },
    ("E. coli", "susceptible"): {
        "ceftriaxone": (0.03, 0.25), "meropenem": (0.06, 0.5),
        "piperacillin-tazobactam": (1.0, 8.0), "ertapenem": (0.03, 0.25),
        "trimethoprim-sulfamethoxazole": (0.25, 1.0),
    },
    ("E. coli", "resistant"): {         # ESBL
        "ceftriaxone": (16.0, 64.0), "meropenem": (0.06, 0.5),
        "piperacillin-tazobactam": (8.0, 32.0), "ertapenem": (0.03, 0.25),
        "ceftazidime-avibactam": (0.25, 1.0),
    },
    ("P. aeruginosa", "susceptible"): {
        "piperacillin-tazobactam": (4.0, 16.0), "cefepime": (1.0, 8.0),
        "meropenem": (0.25, 2.0), "ceftazidime-avibactam": (0.5, 4.0),
        "colistin": (0.5, 1.0),
    },
    ("P. aeruginosa", "resistant"): {   # MDR Pseudomonas
        "piperacillin-tazobactam": (32.0, 128.0), "cefepime": (16.0, 64.0),
        "meropenem": (8.0, 32.0), "ceftazidime-avibactam": (4.0, 16.0),
        "colistin": (0.5, 2.0),
    },
    ("S. aureus", "susceptible"): {     # MSSA
        "vancomycin": (0.5, 1.0), "daptomycin": (0.25, 0.5),
        "oxacillin": (0.12, 0.5), "cefazolin": (0.25, 1.0),
    },
    ("S. aureus", "resistant"): {       # MRSA
        "vancomycin": (0.5, 2.0), "daptomycin": (0.5, 1.0),
        "linezolid": (1.0, 2.0), "oxacillin": (4.0, 32.0),
    },
    ("Enterococcus", "susceptible"): {  # VSE
        "ampicillin": (0.5, 2.0), "vancomycin": (0.5, 2.0),
        "linezolid": (1.0, 2.0), "daptomycin": (1.0, 2.0),
    },
    ("Enterococcus", "resistant"): {    # VRE
        "ampicillin": (32.0, 128.0), "vancomycin": (32.0, 512.0),
        "linezolid": (1.0, 2.0), "daptomycin": (2.0, 4.0),
    },
}

PHENOTYPE_BY_LEVEL = {
    1: ["susceptible"],
    2: ["susceptible", "resistant"],
    3: ["susceptible", "resistant", "MDR"],
}


def generate_patient(curriculum_level: int = 1) -> PatientCase:
    """Generate a random patient case appropriate for the curriculum level."""
    organism = random.choice(ORGANISMS)
    infection_site = random.choice(INFECTION_SITES)
    phenotype = random.choice(PHENOTYPE_BY_LEVEL[curriculum_level])

    # Renal function by level
    if curriculum_level == 1:
        crcl = random.uniform(60, 120)   # normal
    elif curriculum_level == 2:
        crcl = random.uniform(30, 60)    # mild-moderate impairment
    else:
        crcl = random.uniform(10, 45)    # moderate-severe impairment

    # Allergies (rare in level 1, possible in level 2+)
    allergies = []
    if curriculum_level >= 2 and random.random() < 0.25:
        allergies = [random.choice(["penicillin", "cephalosporin"])]

    # Generate antibiogram
    antibiogram = _generate_antibiogram(organism, phenotype)

    return PatientCase(
        age=random.randint(35, 85),
        sex=random.choice(["M", "F"]),
        infection_site=infection_site,
        organism=organism,
        creatinine_clearance=round(crcl, 1),
        allergies=allergies,
        antibiogram=antibiogram,
        phenotype=phenotype,
        curriculum_level=curriculum_level,
    )


def _generate_antibiogram(organism: str, phenotype: str) -> dict[str, float]:
    """Generate realistic MIC values for the organism + phenotype combo."""
    ranges = MIC_RANGES.get((organism, phenotype), {})
    antibiogram = {}
    for drug, (low, high) in ranges.items():
        antibiogram[drug] = round(random.uniform(low, high), 3)
    return antibiogram


def patient_to_text(patient: PatientCase) -> str:
    """Convert a PatientCase to the plain English observation text the LLM sees."""
    allergy_str = ", ".join(patient.allergies) if patient.allergies else "None reported"
    return (
        f"Patient: {patient.age}-year-old {patient.sex}.\n"
        f"Infection site: {patient.infection_site}.\n"
        f"Culture result: {patient.organism} isolated.\n"
        f"Renal function: CrCl {patient.creatinine_clearance} mL/min.\n"
        f"Allergies: {allergy_str}.\n"
        f"Available antibiogram data: {list(patient.antibiogram.keys())}.\n"
    )
