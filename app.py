"""
app.py — AMR-Steward FastAPI server.

Uses a module-level AMREnvironment singleton for /reset and /step.
openenv-core's create_app creates a fresh instance per request (breaking
session state), so we own the reset/step/state/health routes directly.

Run locally:
  uvicorn app:app --reload --port 7860

HuggingFace Spaces:
  CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from env import AMRAction, AMREnvironment
from env.models import PatientCase

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
# Module-level singleton — persists across requests on the same worker
# ---------------------------------------------------------------------------
_ENV = AMREnvironment()

# Warm up data loaders at import time so the first request isn't cold.
try:
    from env.environment import _get_drug_properties, _get_eucast, _get_idsa
    _get_idsa()
    _get_drug_properties()
    _get_eucast()
    logger.info("Data layers loaded successfully.")
except Exception as exc:
    logger.warning("Data pre-load warning: %s", exc)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AMR-Steward",
    description="OpenEnv RL environment for antimicrobial stewardship.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_BASE = Path(__file__).resolve().parent

# Serve demo_web/ assets (JS, CSS, images if any) under /demo-static/
_DEMO_DIR = _BASE / "demo_web"
if _DEMO_DIR.exists():
    app.mount("/demo-static", StaticFiles(directory=str(_DEMO_DIR)), name="demo_static")

# Serve landing page assets under /static/ (Palak's build output)
_STATIC_DIR = _BASE / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    curriculum_level: int = 1
    seed: Optional[int] = None
    episode_id: Optional[str] = None
    patient: Optional[Dict[str, Any]] = None  # inject a specific PatientCase for reproducible eval


class StepRequest(BaseModel):
    action: Dict[str, Any]
    timeout_s: Optional[float] = None


# ---------------------------------------------------------------------------
# Core OpenEnv endpoints
# ---------------------------------------------------------------------------

@app.post("/reset", tags=["Environment Control"])
def reset(body: ResetRequest = ResetRequest()) -> JSONResponse:
    """Start a new episode. Returns the initial observation."""
    try:
        injected_patient: PatientCase | None = None
        if body.patient is not None:
            try:
                injected_patient = PatientCase(**body.patient)
            except Exception as exc:
                logger.warning("Invalid patient payload in /reset — using random patient: %s", exc)

        obs = _ENV.reset(
            seed=body.seed,
            episode_id=body.episode_id,
            curriculum_level=body.curriculum_level,
            patient=injected_patient,
        )
        return JSONResponse({
            "observation": obs.model_dump(),
            "reward": None,
            "done": False,
        })
    except Exception as exc:
        logger.exception("reset() failed")
        return JSONResponse({"detail": str(exc)}, status_code=500)


@app.post("/step", tags=["Environment Control"])
def step(body: StepRequest) -> JSONResponse:
    """Execute an action and return the resulting observation + reward."""
    try:
        action = AMRAction(**body.action)
        obs = _ENV.step(action)
        return JSONResponse({
            "observation": obs.model_dump(),
            "reward": obs.reward,
            "done": obs.done,
        })
    except Exception as exc:
        logger.exception("step() failed")
        return JSONResponse({"detail": str(exc)}, status_code=500)


@app.get("/state", tags=["Environment Control"])
def state() -> JSONResponse:
    """Return current episode state."""
    return JSONResponse(_ENV.state.model_dump())


@app.get("/health", tags=["Environment Control"])
def health() -> JSONResponse:
    """Liveness check."""
    return JSONResponse({"status": "healthy"})


# ---------------------------------------------------------------------------
# Demo + Landing page
# ---------------------------------------------------------------------------

@app.get("/demo", response_class=HTMLResponse, include_in_schema=False)
def demo() -> HTMLResponse:
    """Interactive browser demo — cinematic act-by-act experience."""
    demo_file = _DEMO_DIR / "index.html"
    if demo_file.exists():
        return HTMLResponse(content=demo_file.read_text(encoding="utf-8"))
    return HTMLResponse(content="<p>Demo not found.</p>", status_code=404)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root() -> HTMLResponse:
    html = textwrap.dedent("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <title>AMR-Steward - OpenEnv Environment</title>
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
      <h1>AMR-Steward</h1>
      <p>
        An <a href="https://github.com/meta-pytorch/OpenEnv">OpenEnv</a> RL environment that trains
        an LLM to prescribe the correct antibiotic for drug-resistant bacterial infections,
        verified against EUCAST breakpoints and IDSA guidelines.
        <strong>No LLM judges. No subjectivity. Pure lookup tables.</strong>
      </p>

      <h2>The Problem</h2>
      <p>
        <strong>1.27 million people die from antimicrobial resistance every year</strong>, more than
        HIV or malaria. Prescribing the wrong antibiotic fails the patient <em>and</em> accelerates
        resistance. This environment trains an agent to investigate systematically before prescribing.
      </p>

      <h2>Demo Case</h2>
      <pre>Patient: 67-year-old female, ICU, central-line bloodstream infection.
Culture: Klebsiella pneumoniae isolated from blood cultures x2.
Renal function: CrCl 35 mL/min (moderate impairment).
Allergies: None reported.</pre>

      <table>
        <thead><tr><th>Prescription</th><th>Drug</th><th>Reward</th></tr></thead>
        <tbody>
          <tr>
            <td>Broad-spectrum guess (no investigation)</td>
            <td><span class="badge-bad">Meropenem 1g IV q8h</span> — carbapenem-resistant organism, wrong drug</td>
            <td>0.12</td>
          </tr>
          <tr>
            <td>IDSA first-line, renal-adjusted (GRPO-trained model peak: 0.84)</td>
            <td><span class="badge-good">Ceftazidime/avibactam 2.5g IV q12h</span></td>
            <td>0.91</td>
          </tr>
        </tbody>
      </table>
      <p style="font-size:0.85em;color:#8b949e;">Reward 0.91 = optimal prescription for this case. Trained model avg: 0.71–0.84 across 224 curriculum cases. Random baseline: 0.05–0.10.</p>

      <h2>OpenEnv API</h2>
      <pre>POST /reset    body: {"curriculum_level": 1}
POST /step     body: {"action": {"action_type": "INVESTIGATE", "tool_name": "interpret_resistance", "tool_arg": "meropenem"}}
POST /step     body: {"action": {"action_type": "COMMIT", "prescription": {"drug": "...", "dose": "...", "duration": "...", "justification": "..."}}}
GET  /state    -> {episode_id, step_count, budget_remaining, ...}
GET  /health   -> {"status": "healthy"}
GET  /docs     -> interactive Swagger UI</pre>

      <p>
        <a href="/docs">Interactive API Docs</a> &middot;
        <a href="/health">Health Check</a> &middot;
        <a href="https://github.com/saaheerpurav/amr-steward">GitHub</a>
      </p>
    </body>
    </html>
    """)
    return HTMLResponse(content=html)
