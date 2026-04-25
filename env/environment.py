# BHATIA OWNS THIS FILE
# Implements the OpenEnv environment: reset(), step(), state()

from .models import PatientCase, AMRAction, AMRObservation

# TODO Bhatia: import patient_generator and antibiogram_generator from data/
# TODO Bhatia: import reward functions from env/reward.py once Saaheer pushes them
# TODO Bhatia: import eucast_parser from data/ once Saaheer pushes it


BUDGET_BY_LEVEL = {1: 5, 2: 4, 3: 3}


class AMREnvironment:
    def __init__(self):
        self.current_patient: PatientCase | None = None
        self.tool_results: list[str] = []
        self.budget_remaining: int = 5
        self.done: bool = False
        self.curriculum_level: int = 1

    def reset(self, curriculum_level: int = 1) -> AMRObservation:
        """Start a fresh episode. Sample a new patient."""
        self.curriculum_level = curriculum_level
        self.budget_remaining = BUDGET_BY_LEVEL[curriculum_level]
        self.tool_results = []
        self.done = False

        # TODO Bhatia: replace with real patient generator
        # self.current_patient = generate_patient(curriculum_level)
        # For now stub:
        self.current_patient = PatientCase(
            age=67, sex="F", infection_site="bacteremia",
            organism="K. pneumoniae", creatinine_clearance=35.0,
            allergies=[], antibiogram={"meropenem": 8.0, "ceftazidime-avibactam": 1.0},
            phenotype="resistant", curriculum_level=curriculum_level
        )

        return self._build_observation()

    def step(self, action: AMRAction) -> tuple[AMRObservation, float, bool]:
        """Apply an action. Returns (observation, reward, done)."""
        if self.done:
            raise ValueError("Episode is done. Call reset() first.")

        reward = 0.0

        if action.action_type == "INVESTIGATE":
            result = self._execute_tool(action.tool_name, action.tool_arg)
            self.tool_results.append(result)
            self.budget_remaining -= 1

            if self.budget_remaining <= 0:
                # Budget exhausted without committing — small penalty
                reward = -0.1
                self.done = True

        elif action.action_type == "COMMIT":
            # TODO Bhatia: call reward functions from env/reward.py
            # reward = compute_total_reward(action.prescription, self.current_patient, self.tool_results)
            reward = 0.5  # stub until Saaheer's reward.py is ready
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
        # TODO Bhatia: wire up real tool functions
        if tool_name == "interpret_resistance":
            return f"[STUB] {tool_arg} resistance interpretation for {self.current_patient.organism}"
        elif tool_name == "check_guideline":
            return f"[STUB] IDSA guideline for {tool_arg}"
        elif tool_name == "assess_patient_factors":
            return f"[STUB] Patient factors: CrCl={self.current_patient.creatinine_clearance}, allergies={self.current_patient.allergies}"
        return f"[STUB] Unknown tool: {tool_name}"

    def _build_observation(self) -> AMRObservation:
        p = self.current_patient
        patient_text = (
            f"Patient: {p.age}-year-old {p.sex}, infection site: {p.infection_site}.\n"
            f"Culture: {p.organism} isolated.\n"
            f"Renal function: CrCl {p.creatinine_clearance} mL/min.\n"
            f"Allergies: {', '.join(p.allergies) if p.allergies else 'None reported'}.\n"
        )
        return AMRObservation(
            patient_text=patient_text,
            tool_results=list(self.tool_results),
            budget_remaining=self.budget_remaining,
            world_model_rankings="",  # Saaheer fills this in via enrich_observation()
            done=self.done,
        )
