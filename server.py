"""
CreditMaze — FastAPI Server
Compatible with: FastAPI >=0.115, pydantic >=2.7, uvicorn >=0.31

Endpoints (all required by OpenEnv spec):
  POST /reset     — start new episode
  POST /step      — submit action
  GET  /state     — get episode state (causal labels revealed after done)
  GET  /tasks     — list tasks + action schema
  POST /grader    — get grader metrics for completed episode
  POST /baseline  — trigger baseline inference script
  GET  /health    — health check
"""
from __future__ import annotations
import subprocess, json
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from environment.env import CreditMazeEnv
from environment.models import Action

app = FastAPI(
    title="CreditMaze",
    version="1.0.0",
    description=(
        "Procedurally generated RL environment for measuring credit assignment quality. "
        "Introduces PSIA, CCE, and MPCS as novel evaluation metrics."
    ),
)

env = CreditMazeEnv()


# ── Request schemas ───────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    tier:   str = "easy"
    domain: Optional[str] = None
    seed:   Optional[int] = None


class StepRequest(BaseModel):
    episode_id:      str
    action_id:       str
    reasoning:       Optional[str]   = None
    credit_estimate: Optional[float] = None


class GraderRequest(BaseModel):
    episode_id: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    req = req or ResetRequest()
    valid_tiers = ["easy", "medium", "hard", "multi-pivot"]
    if req.tier not in valid_tiers:
        raise HTTPException(400, f"tier must be one of {valid_tiers}")
    obs = env.reset(tier=req.tier, domain=req.domain, seed=req.seed)
    return obs.model_dump()


@app.post("/step")
def step(req: StepRequest):
    try:
        action = Action(
            action_id=req.action_id,
            reasoning=req.reasoning,
            credit_estimate=req.credit_estimate,
        )
        result = env.step(req.episode_id, action)
        return result.model_dump()
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/state")
def state(episode_id: str):
    try:
        return env.state(episode_id).model_dump()
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.get("/tasks")
def tasks():
    def grader_meta(task_id: str, summary: str) -> dict:
        return {
            "type": "llm",
            "prompt_template": (
                f"Score the agent's performance on {task_id} from 0.01 to 0.99 "
                f"based on whether it completed the task correctly and assigned "
                f"credit to the causally important step(s). {summary}"
            ),
        }

    return {
        "tasks": [
            {"id": "task_easy",        "difficulty": "easy",   "max_steps": 15, "grader": grader_meta("task_easy", "This alias maps to the environment's canonical easy task.")},
            {"id": "task_medium",      "difficulty": "medium", "max_steps": 15, "grader": grader_meta("task_medium", "This alias maps to the environment's canonical medium task.")},
            {"id": "task_hard",        "difficulty": "hard",   "max_steps": 15, "grader": grader_meta("task_hard", "This alias maps to the environment's canonical hard task.")},
            {"id": "corridor_easy",    "difficulty": "easy",   "max_steps": 15, "grader": grader_meta("corridor_easy", "The environment exposes the final normalized score through its internal grader.")},
            {"id": "research_medium",  "difficulty": "medium", "max_steps": 15, "grader": grader_meta("research_medium", "The environment exposes the final normalized score through its internal grader.")},
            {"id": "debugging_hard",   "difficulty": "hard",   "max_steps": 15, "grader": grader_meta("debugging_hard", "The environment exposes the final normalized score through its internal grader.")},
            {"id": "resource_hard",    "difficulty": "hard",   "max_steps": 15, "grader": grader_meta("resource_hard", "The environment exposes the final normalized score through its internal grader.")},
            {"id": "triage_multipivot","difficulty": "hard",   "max_steps": 15, "grader": grader_meta("triage_multipivot", "The environment exposes the final normalized score through its internal grader.")},
        ],
        "action_schema": {
            "type": "object",
            "properties": {
                "action_id":       {"type": "string",  "description": "Must be in observation.available_actions"},
                "reasoning":       {"type": "string",  "description": "Agent chain-of-thought"},
                "credit_estimate": {"type": "number",  "minimum": 0.0, "maximum": 1.0, "description": "Step importance [0,1]"},
            },
            "required": ["action_id"],
        },
        "metrics_explained": {
            "PSIA": "Pivotal Step Identification Accuracy",
            "CCE":  "Credit Calibration Error — MSE vs ground-truth",
            "TSR":  "Task Success Rate",
            "MPCS": "Multi-Pivot Coordination Score (Tier 4)",
        },
    }


@app.post("/grader")
def grader(req: GraderRequest):
    try:
        s = env.state(req.episode_id)
    except KeyError as e:
        raise HTTPException(404, str(e))

    if not s.episode_complete:
        raise HTTPException(400, "Episode not complete. Keep calling /step until done=True.")

    return {
        "episode_id":           req.episode_id,
        "outcome":              s.outcome,
        "score":                env.normalized_score(req.episode_id),
        "raw_reward":           round(s.cumulative_reward, 4),
        "session_psia":         s.session_psia,
        "session_cce":          s.session_cce,
        "session_tsr":          s.session_tsr,
        "session_mpcs":         s.session_mpcs,
        "episodes_completed":   s.episodes_completed,
        "causal_faithfulness":  s.causal_faithfulness,
        "pivotal_step_indices": s.pivotal_step_indices,
    }


@app.post("/baseline")
def baseline():
    try:
        result = subprocess.run(
            ["python", "baseline.py", "--n", "3"],
            capture_output=True, timeout=600, text=True,
        )
        if result.returncode != 0:
            raise HTTPException(500, f"Baseline script failed:\n{result.stderr[:800]}")
        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        return json.loads(lines[-1])
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "Baseline script timed out")
    except json.JSONDecodeError:
        raise HTTPException(500, "Baseline script did not produce valid JSON")


@app.get("/")
def root():
    return {
        "name": "CreditMaze",
        "status": "ok",
        "message": "CreditMaze is running.",
        "endpoints": [
            "/health",
            "/tasks",
            "/reset",
            "/step",
            "/state",
            "/grader",
            "/baseline",
        ],
    }


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0", "environment": "CreditMaze"}


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
