# BHATIA OWNS THIS FILE
# FastAPI app — exposes the OpenEnv standard endpoints

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from env import AMREnvironment, AMRAction

app = FastAPI(title="AMR-Steward", description="RL environment for antibiotic prescribing")

env = AMREnvironment()


class StepRequest(BaseModel):
    action_type: str                # "INVESTIGATE" | "COMMIT"
    tool_name: str | None = None
    tool_arg: str | None = None
    prescription: dict | None = None


@app.get("/reset")
def reset(level: int = 1):
    """Reset environment and return initial observation."""
    obs = env.reset(curriculum_level=level)
    return obs.__dict__


@app.post("/step")
def step(req: StepRequest):
    """Take an action. Returns observation, reward, done."""
    action = AMRAction(
        action_type=req.action_type,
        tool_name=req.tool_name,
        tool_arg=req.tool_arg,
        prescription=req.prescription,
    )
    obs, reward, done = env.step(action)
    return {"observation": obs.__dict__, "reward": reward, "done": done}


@app.get("/state")
def state():
    """Return full serialized episode state."""
    return env.state()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"message": "AMR-Steward environment running. Use /reset to start an episode."}
