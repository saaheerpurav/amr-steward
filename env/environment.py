# BHATIA OWNS THIS FILE
# Implements the OpenEnv environment: reset(), step(), state()

from .models import PatientCase, AMRAction, AMRObservation
from .reward import compute_total_reward, _get_idsa, _normalize_drug, _organism_to_idsa_key

BUDGET_BY_LEVEL = {1: 5, 2: 4, 3: 3}


class AMREnvironment:
    def __init__(self):
        self.current_patient: PatientCase | None = None
        self.tool_results: list[str] = []
        self.tool_call_history: list[str] = []
        self.budget_remaining: int = 5
        self.done: bool = False
        self.curriculum_level: int = 1

    def reset(self, curriculum_level: int = 1) -> AMRObservation:
        """Start a fresh episode. Sample a new patient."""
        from data.patient_generator import generate_patient
        self.curriculum_level = curriculum_level
        self.budget_remaining = BUDGET_BY_LEVEL[curriculum_level]
        self.tool_results = []
        self.tool_call_history = []
        self.done = False
        self.current_patient = generate_patient(curriculum_level)
        return self._build_observation()

    def step(self, action: AMRAction) -> tuple[AMRObservation, float, bool]:
        """Apply an action. Returns (observation, reward, done)."""
        if self.done:
            raise ValueError("Episode is done. Call reset() first.")

        from data import eucast_parser as eucast
        reward = 0.0

        if action.action_type == "INVESTIGATE":
            tool_tag = f"{action.tool_name}:{action.tool_arg}"
            self.tool_call_history.append(tool_tag)
            result = self._execute_tool(action.tool_name, action.tool_arg)
            self.tool_results.append(result)
            self.budget_remaining -= 1

            if self.budget_remaining <= 0:
                reward = -0.1
                self.done = True

        elif action.action_type == "COMMIT":
            prescription = action.prescription or {}
            total, breakdown = compute_total_reward(
                prescription, self.current_patient, self.tool_call_history, eucast
            )
            reward = total
            self.done = True

        obs = self._build_observation()
        return obs, reward, self.done

    def state(self) -> dict:
        """Return full serialized episode state."""
        return {
            "patient": self.current_patient.__dict__ if self.current_patient else None,
            "tool_results": self.tool_results,
            "budget_remaining": self.budget_remaining,
            "done": self.done,
            "curriculum_level": self.curriculum_level,
        }

    def _execute_tool(self, tool_name: str, tool_arg: str) -> str:
        """Execute a diagnostic tool call. Returns plain text result."""
        from data import eucast_parser as eucast
        p = self.current_patient

        if tool_name == "interpret_resistance":
            drug = _normalize_drug(tool_arg)
            mic = p.antibiogram.get(tool_arg) or p.antibiogram.get(drug)
            if mic is None:
                return f"No MIC data available for {tool_arg} in antibiogram."
            classification = eucast.classify_mic(p.organism, drug, mic)
            label = {"S": "Susceptible", "I": "Intermediate", "R": "Resistant"}.get(classification, "Unknown")
            return (
                f"Resistance interpretation for {tool_arg} against {p.organism}:\n"
                f"  MIC: {mic} mg/L -> {label} ({classification})\n"
                f"  EUCAST breakpoint applied."
            )

        elif tool_name == "check_guideline":
            idsa = _get_idsa()
            syndrome = tool_arg.lower() if tool_arg else p.infection_site
            idsa_key = _organism_to_idsa_key(p.organism, p.phenotype)
            syndrome_data = idsa.get(syndrome, {})
            org_data = syndrome_data.get(idsa_key, {})

            if not org_data:
                return f"No IDSA guideline found for {idsa_key} in {syndrome}."

            return (
                f"IDSA guideline — {syndrome} / {idsa_key}:\n"
                f"  First-line: {org_data.get('first_line', 'N/A')}\n"
                f"  Dose: {org_data.get('dose', 'N/A')}\n"
                f"  Duration: {org_data.get('duration', 'N/A')}\n"
                f"  Alternatives: {', '.join(org_data.get('alternatives', [])) or 'None listed'}\n"
                f"  Notes: {org_data.get('notes', '')}"
            )

        elif tool_name == "assess_patient_factors":
            crcl = p.creatinine_clearance
            if crcl > 60:
                renal_status = "Normal (CrCl >60 mL/min) — standard dosing applies."
            elif crcl >= 30:
                renal_status = f"Mild-moderate impairment (CrCl {crcl} mL/min) — adjust renally-cleared drugs."
            elif crcl >= 10:
                renal_status = f"Severe impairment (CrCl {crcl} mL/min) — significant dose reduction required."
            else:
                renal_status = f"Renal failure (CrCl {crcl} mL/min) — avoid nephrotoxic agents; dialysis dosing."

            allergy_str = ", ".join(p.allergies) if p.allergies else "None reported"
            return (
                f"Patient factors:\n"
                f"  CrCl: {crcl} mL/min — {renal_status}\n"
                f"  Allergies: {allergy_str}\n"
                f"  Age: {p.age} years, Sex: {p.sex}\n"
                f"  Phenotype: {p.phenotype}"
            )

        return f"Unknown tool: {tool_name}"

    def _build_observation(self) -> AMRObservation:
        from data.patient_generator import patient_to_text
        p = self.current_patient
        patient_text = patient_to_text(p)
        return AMRObservation(
            patient_text=patient_text,
            tool_results=list(self.tool_results),
            budget_remaining=self.budget_remaining,
            world_model_rankings="",
            done=self.done,
        )
