"""
test_env.py — Integration smoke test for AMR-Steward environment.
Run: python test_env.py
"""
import sys
sys.path.insert(0, ".")

from env import AMREnvironment, AMRAction
from env.environment import PatientCase

PASS = "[PASS]"
FAIL = "[FAIL]"


def make_demo_env():
    """Return an env pre-loaded with the canonical 67F CRE bacteremia demo patient."""
    env = AMREnvironment()
    env.current_patient = PatientCase(
        age=67, sex="F", infection_site="bacteremia",
        organism="K. pneumoniae", creatinine_clearance=35.0,
        allergies=[],
        antibiogram={"meropenem": 8.0, "ceftazidime-avibactam": 1.0, "colistin": 1.0},
        phenotype="resistant", curriculum_level=1,
    )
    env.budget_remaining = 5
    env.done = False
    env.episode_step = 0
    env.tool_results = []
    return env


def test_reset():
    env = AMREnvironment()
    obs = env.reset(curriculum_level=1)
    assert obs.budget_remaining == 5, "budget should be 5 for level 1"
    assert not obs.done
    print(PASS + " reset() level=1 | budget=5")

    obs2 = env.reset(curriculum_level=3)
    assert obs2.budget_remaining == 3
    print(PASS + " reset() level=3 | budget=3")


def test_tools():
    env = make_demo_env()

    obs, _, done = env.step(AMRAction("INVESTIGATE", tool_name="interpret_resistance", tool_arg="meropenem"))
    assert not done
    assert "Resistant" in obs.tool_results[-1] or "MIC" in obs.tool_results[-1]
    assert obs.budget_remaining == 4
    print(PASS + " interpret_resistance meropenem | " + obs.tool_results[-1][:80].replace("\n", " "))

    obs, _, done = env.step(AMRAction("INVESTIGATE", tool_name="check_guideline", tool_arg="bacteremia"))
    assert not done
    assert "ceftazidime-avibactam" in obs.tool_results[-1].lower() or "IDSA" in obs.tool_results[-1]
    print(PASS + " check_guideline bacteremia | " + obs.tool_results[-1][:80].replace("\n", " "))

    obs, _, done = env.step(AMRAction("INVESTIGATE", tool_name="assess_patient_factors"))
    assert not done
    assert "CrCl" in obs.tool_results[-1] or "Renal" in obs.tool_results[-1]
    print(PASS + " assess_patient_factors | " + obs.tool_results[-1][:80].replace("\n", " "))


def test_correct_prescription():
    env = make_demo_env()
    # Investigate first (boosts R5)
    env.step(AMRAction("INVESTIGATE", tool_name="interpret_resistance", tool_arg="meropenem"))
    env.step(AMRAction("INVESTIGATE", tool_name="check_guideline", tool_arg="bacteremia"))
    env.step(AMRAction("INVESTIGATE", tool_name="assess_patient_factors"))

    # Correct drug: ceftazidime-avibactam. CrCl=35 -> 1.25g q8h
    obs, reward, done = env.step(AMRAction("COMMIT", prescription={
        "drug": "ceftazidime-avibactam",
        "dose": "1.25g IV q8h",
        "duration": "14 days",
        "justification": "CRE K. pneumoniae, IDSA first-line, renal-adjusted",
    }))
    assert done
    bd = env.state()["last_reward_breakdown"]
    assert bd["R1_activity"] == 1.0, "R1 should be 1.0 (ceftazidime-avibactam is susceptible for CRE)"
    assert bd["R2_guideline"] >= 0.5, "R2 should be >= 0.5 (IDSA first-line)"
    assert bd["R5_reasoning"] > 0.0, "R5 should be > 0 (tools were called)"
    assert reward >= 0.5, "Total reward for correct Rx should be >= 0.5"
    print(PASS + " CORRECT Rx reward=" + str(reward) +
          " | R1=" + str(bd["R1_activity"]) +
          " R2=" + str(bd["R2_guideline"]) +
          " R3=" + str(bd["R3_stewardship"]) +
          " R4=" + str(bd["R4_dose"]) +
          " R5=" + str(bd["R5_reasoning"]))


def test_wrong_prescription():
    env = make_demo_env()
    # No investigation (blind guess)
    obs, reward, done = env.step(AMRAction("COMMIT", prescription={
        "drug": "meropenem",
        "dose": "1g IV q8h",
        "duration": "14 days",
        "justification": "blind guess",
    }))
    assert done
    bd = env.state()["last_reward_breakdown"]
    assert bd["R1_activity"] == 0.0, "R1 must be 0 for a resistant drug"
    assert bd["R3_stewardship"] == 0.0, "R3 must be 0 when R1 is 0"
    assert bd["R5_reasoning"] == 0.0, "R5 must be 0 — no tools called"
    assert reward < 0.4, "Wrong Rx should have low reward"
    print(PASS + " WRONG Rx (meropenem/CRE) reward=" + str(reward) +
          " | R1=" + str(bd["R1_activity"]) + " (expect 0.0)")


def test_budget_exhaustion():
    env = make_demo_env()
    env.budget_remaining = 1
    obs, reward, done = env.step(AMRAction("INVESTIGATE", tool_name="assess_patient_factors"))
    assert done, "Budget exhausted -> done=True"
    assert reward < 0, "Budget exhaustion should give negative reward"
    print(PASS + " budget exhaustion | reward=" + str(reward) + " done=" + str(done))


def test_invalid_action():
    env = make_demo_env()
    try:
        env.step(AMRAction("INVALID_TYPE"))
        print(FAIL + " should have raised ValueError")
    except ValueError:
        print(PASS + " invalid action_type raises ValueError correctly")


def test_app_import():
    from app import app
    assert app.title == "AMR-Steward"
    print(PASS + " app.py FastAPI import OK | title=" + app.title)


def test_state():
    env = make_demo_env()
    env.step(AMRAction("INVESTIGATE", tool_name="interpret_resistance", tool_arg="meropenem"))
    s = env.state()
    assert s["episode_step"] == 1
    assert s["budget_remaining"] == 4
    assert len(s["tool_results"]) == 1
    print(PASS + " state() | step=" + str(s["episode_step"]) + " budget=" + str(s["budget_remaining"]))


if __name__ == "__main__":
    print("=" * 60)
    print("AMR-Steward Integration Tests")
    print("=" * 60)
    errors = []
    for fn in [test_reset, test_tools, test_correct_prescription,
               test_wrong_prescription, test_budget_exhaustion,
               test_invalid_action, test_state, test_app_import]:
        try:
            fn()
        except Exception as e:
            print(FAIL + " " + fn.__name__ + " => " + str(e))
            errors.append(fn.__name__)
    print("=" * 60)
    if errors:
        print("FAILED: " + ", ".join(errors))
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
