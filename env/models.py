from dataclasses import dataclass, field


@dataclass
class PatientCase:
    age: int
    sex: str
    infection_site: str          # "bacteremia" | "UTI" | "pneumonia" | "intra-abdominal"
    organism: str                # "K. pneumoniae" | "E. coli" | "P. aeruginosa" | "S. aureus" | "Enterococcus"
    creatinine_clearance: float  # mL/min — kidney function
    allergies: list[str]         # e.g. ["penicillin"]
    antibiogram: dict[str, float]  # drug → MIC value in mg/L
    phenotype: str               # "susceptible" | "resistant" | "MDR"
    curriculum_level: int        # 1 | 2 | 3


@dataclass
class AMRAction:
    action_type: str             # "INVESTIGATE" | "COMMIT"
    tool_name: str | None = None # "interpret_resistance" | "check_guideline" | "assess_patient_factors"
    tool_arg: str | None = None  # drug name or syndrome name
    prescription: dict | None = None  # {drug, dose, duration, justification}


@dataclass
class AMRObservation:
    patient_text: str            # plain English patient description
    tool_results: list[str] = field(default_factory=list)  # tool call results so far
    budget_remaining: int = 5
    world_model_rankings: str = ""  # JEPA information gain predictions
    done: bool = False
