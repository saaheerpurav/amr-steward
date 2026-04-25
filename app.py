"""
app.py — AMR-Steward FastAPI server.
Owns: Bhatia

Built on top of openenv-core's `create_app` so we get the standard OpenEnv
HTTP + WebSocket protocol for free (POST /reset, POST /step, GET /state,
GET /health, /docs, optional /web). Compatible with any OpenEnv HTTPEnvClient.

Run locally:
  uvicorn app:app --reload --port 7860

HuggingFace Spaces:
  CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
"""

from __future__ import annotations

import logging
import os
import textwrap

from fastapi.responses import HTMLResponse
from openenv.core.env_server import create_app

from env import AMRAction, AMREnvironment, AMRObservation

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("amr_steward.app")

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
# Build the OpenEnv-compliant app
# ---------------------------------------------------------------------------
# Passing the *class* (not an instance) lets the OpenEnv HTTP server
# spin up an isolated AMREnvironment per concurrent session, keyed by
# episode_id. This fixes the "global env shared across requests" bug.

MAX_CONCURRENT_ENVS = int(os.getenv("MAX_CONCURRENT_ENVS", "1"))

app = create_app(
    AMREnvironment,
    AMRAction,
    AMRObservation,
    env_name="amr-steward",
    max_concurrent_envs=MAX_CONCURRENT_ENVS,
)

# ---------------------------------------------------------------------------
# Add our HuggingFace Spaces landing page on top of the OpenEnv app.
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root() -> HTMLResponse:
    """HTML status page shown on the HuggingFace Space root URL."""
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
        <thead><tr><th>Model</th><th>Prescription</th><th>Reward</th></tr></thead>
        <tbody>
          <tr>
            <td>Untrained Llama-3.1-8B</td>
            <td><span class="badge-bad">Meropenem 1g IV q8h</span> - wrong, resistant organism</td>
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
          <tr><td>R5 Reasoning grounding</td><td>10%</td><td>Tool call history check</td></tr>
        </tbody>
      </table>

      <h2>OpenEnv API</h2>
      <pre>POST /reset    body: {"seed": 42, "episode_id": "...", "curriculum_level": 1}
POST /step     body: {"action": {"action_type": "INVESTIGATE", "tool_name": "...", "tool_arg": "..."}}
GET  /state    -> {episode_id, step_count, curriculum_level, budget_remaining, ...}
GET  /health   -> 200 OK liveness check
GET  /docs     -> interactive Swagger UI
WS   /ws       -> WebSocket transport (used by HTTPEnvClient)</pre>

      <p>
        <a href="/docs">Interactive API Docs (Swagger)</a> &middot;
        <a href="/health">Health Check</a>
      </p>
    </body>
    </html>
    """)
    return HTMLResponse(content=html)
