"""
app.py — AMR-Steward FastAPI server.
Owns: Bhatia

Exposes the standard OpenEnv endpoints:
  GET  /reset?level=1  → AMRObservation
  POST /step           → StepResponse
  GET  /state          → StateResponse
  GET  /health         → HealthResponse
  GET  /               → status page

Run locally:
  uvicorn app:app --reload --port 7860

HuggingFace Spaces:
  CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
"""

from __future__ import annotations

import logging
import textwrap
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from env import AMREnvironment, AMRAction
from env.models import (
    ObservationResponse,
    StateResponse,
    StepRequest,
    StepResponse,
    HealthResponse,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("amr_steward.app")

# ---------------------------------------------------------------------------
# App + environment singleton
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AMR-Steward environment booting …")
    # Warm up data loaders at startup (avoids first-request cold start)
    try:
        from env.environment import _get_idsa, _get_drug_properties, _get_eucast
        _get_idsa()
        _get_drug_properties()
        _get_eucast()
        logger.info("Data layers loaded successfully.")
    except Exception as exc:
        logger.warning("Data pre-load warning: %s", exc)
    yield
    logger.info("AMR-Steward shutting down.")


app = FastAPI(
    title="AMR-Steward",
    description=(
        "OpenEnv RL environment for clinical antimicrobial stewardship. "
        "Trains an LLM to prescribe the correct antibiotic for drug-resistant infections, "
        "verified against EUCAST breakpoints and IDSA guidelines."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# One environment instance per server process.
# For multi-worker deployments, each worker gets its own env (stateless per episode).
env = AMREnvironment()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obs_to_response(obs, reward_breakdown: dict[str, Any] | None = None) -> ObservationResponse:
    """Convert an AMRObservation dataclass → Pydantic response model."""
    return ObservationResponse(
        patient_text=obs.patient_text,
        tool_results=obs.tool_results,
        budget_remaining=obs.budget_remaining,
        world_model_rankings=obs.world_model_rankings,
        done=obs.done,
        observation_text=obs.to_text(),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/reset",
    response_model=ObservationResponse,
    summary="Start a new episode",
    tags=["OpenEnv"],
)
def reset(
    level: int = Query(
        default=1,
        ge=1,
        le=3,
        description="Curriculum level: 1 (easy/susceptible), 2 (moderate), 3 (complex/MDR).",
    )
) -> ObservationResponse:
    """
    Reset the environment and return the initial patient observation.

    - **level 1**: susceptible organisms, 5 tool budget, no comorbidities
    - **level 2**: some resistant organisms, 4 tool budget, mild renal impairment
    - **level 3**: MDR organisms, 3 tool budget, severe renal impairment + allergies
    """
    try:
        obs = env.reset(curriculum_level=level)
        return _obs_to_response(obs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error in /reset")
        raise HTTPException(status_code=500, detail=f"Environment error: {exc}")


@app.post(
    "/step",
    response_model=StepResponse,
    summary="Take an action in the current episode",
    tags=["OpenEnv"],
)
def step(req: StepRequest) -> StepResponse:
    """
    Apply an action to the current episode.

    **INVESTIGATE actions** execute a diagnostic tool and consume 1 budget unit:
    - `interpret_resistance`: classify a drug's MIC as Susceptible / Intermediate / Resistant
    - `check_guideline`: look up IDSA recommendation for a syndrome
    - `assess_patient_factors`: summarise renal dosing and allergy flags

    **COMMIT actions** finalise the prescription and compute the 5-component reward.
    Prescription must include: `drug`, `dose`, `duration`, `justification`.
    """
    action = AMRAction(
        action_type=req.action_type,
        tool_name=req.tool_name,
        tool_arg=req.tool_arg,
        prescription=req.prescription,
    )
    try:
        obs, reward, done = env.step(action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error in /step")
        raise HTTPException(status_code=500, detail=f"Environment error: {exc}")

    breakdown = env._last_reward_breakdown if done and req.action_type == "COMMIT" else None

    return StepResponse(
        observation=_obs_to_response(obs),
        reward=round(reward, 6),
        done=done,
        reward_breakdown=breakdown,
    )


@app.get(
    "/state",
    response_model=StateResponse,
    summary="Return full serialised episode state",
    tags=["OpenEnv"],
)
def state() -> StateResponse:
    """
    Return the full internal episode state.
    Useful for debugging, logging, and replaying episodes.
    """
    s = env.state()
    return StateResponse(
        patient=s["patient"],
        tool_results=s["tool_results"],
        budget_remaining=s["budget_remaining"],
        done=s["done"],
        curriculum_level=s["curriculum_level"],
        episode_step=s["episode_step"],
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Meta"],
)
def health() -> HealthResponse:
    """Liveness check — returns 200 OK when the server is ready."""
    return HealthResponse(status="ok", version="1.0.0")


@app.get(
    "/",
    response_class=HTMLResponse,
    summary="Environment status page",
    tags=["Meta"],
    include_in_schema=False,
)
def root() -> HTMLResponse:
    """
    Simple HTML status page shown on the HuggingFace Space root URL.
    Shows a before/after demo case to illustrate what the environment does.
    """
    html = textwrap.dedent("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <title>AMR-Steward — OpenEnv Environment</title>
      <style>
        body { font-family: 'Segoe UI', sans-serif; max-width: 860px; margin: 40px auto;
               padding: 0 20px; background: #0d1117; color: #e6edf3; }
        h1   { color: #58a6ff; }
        h2   { color: #79c0ff; border-bottom: 1px solid #30363d; padding-bottom: 6px; }
        code { background: #161b22; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }
        pre  { background: #161b22; padding: 16px; border-radius: 8px; overflow-x: auto; }
        .badge-bad  { background: #7d2a2a; color: #ff7b72; padding: 2px 8px; border-radius: 4px; }
        .badge-good { background: #1a4429; color: #56d364; padding: 2px 8px; border-radius: 4px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #30363d; padding: 8px 12px; text-align: left; }
        th { background: #161b22; }
        a  { color: #58a6ff; }
      </style>
    </head>
    <body>
      <h1>🦠 AMR-Steward</h1>
      <p>
        An <a href="https://github.com/openenv/openenv">OpenEnv</a> RL environment that trains an LLM
        to prescribe the correct antibiotic for drug-resistant bacterial infections —
        verified against EUCAST breakpoints and IDSA guidelines.
        <strong>No LLM judges. No subjectivity. Pure lookup tables.</strong>
      </p>

      <h2>The Problem</h2>
      <p>
        <strong>1.27 million people die from antimicrobial resistance every year</strong> — more than
        HIV or malaria. Prescribing the wrong antibiotic fails the patient <em>and</em> accelerates
        resistance. This environment trains an agent to investigate systematically before prescribing.
      </p>

      <h2>Demo Case</h2>
      <pre>Patient: 67-year-old female, ICU, central-line bloodstream infection.
Culture: Klebsiella pneumoniae isolated from blood cultures ×2.
Renal function: CrCl 35 mL/min (moderate impairment).
Allergies: None reported.</pre>

      <table>
        <thead><tr><th>Model</th><th>Prescription</th><th>Reward</th></tr></thead>
        <tbody>
          <tr>
            <td>Untrained Qwen-7B</td>
            <td><span class="badge-bad">Meropenem 1g IV q8h</span> — wrong, resistant organism</td>
            <td>0.12</td>
          </tr>
          <tr>
            <td>GRPO-trained model</td>
            <td><span class="badge-good">Ceftazidime/avibactam 2.5g IV q12h (renal-adjusted, CRE-active)</span></td>
            <td>0.91</td>
          </tr>
        </tbody>
      </table>

      <h2>Reward Components</h2>
      <table>
        <thead><tr><th>Component</th><th>Weight</th><th>Verifier</th></tr></thead>
        <tbody>
          <tr><td>R1 Microbiological activity</td><td>40%</td><td>EUCAST MIC lookup</td></tr>
          <tr><td>R2 Guideline concordance</td><td>25%</td><td>IDSA table lookup</td></tr>
          <tr><td>R3 Stewardship (narrowest drug)</td><td>15%</td><td>Conditional on R1</td></tr>
          <tr><td>R4 Dose / renal correctness</td><td>10%</td><td>Drug properties table</td></tr>
          <tr><td>R5 Reasoning grounding</td><td>10%</td><td>Tool call history regex</td></tr>
        </tbody>
      </table>

      <h2>API Endpoints</h2>
      <pre>GET  /reset?level=1   → start episode (level 1–3)
POST /step            → take action (INVESTIGATE or COMMIT)
GET  /state           → full episode state
GET  /health          → 200 OK liveness check
GET  /docs            → interactive Swagger UI</pre>

      <p>
        <a href="/docs">📖 Interactive API Docs (Swagger)</a> ·
        <a href="/redoc">📄 ReDoc</a> ·
        <a href="/health">✅ Health Check</a>
      </p>
    </body>
    </html>
    """)
    return HTMLResponse(content=html)
