"""
CreditMaze — Pydantic models for OpenEnv compliance.
Observation, Action, StepResult, StepInfo, State.
"""
from __future__ import annotations
from typing import Optional, List, Dict
from pydantic import BaseModel, Field


class Observation(BaseModel):
    """Returned by reset() and step(). What the agent sees."""
    episode_id: str
    domain: str           # corridor | research | debugging | resource | triage
    tier: str             # easy | medium | hard | multi-pivot
    t_total: int          # Total designed steps in episode
    step_count: int       # Current step (0-indexed, increments after each step)
    max_steps: int        # Hard cap — episode terminates here
    context: str          # Full narrative context for this step
    available_actions: List[str]   # Valid action_ids at this step

    # Populated after first step (None on reset)
    last_action_taken:  Optional[str]   = None
    last_step_reward:   Optional[float] = None
    cumulative_reward:  Optional[float] = None
    episode_outcome:    Optional[str]   = None  # in_progress | success | failure

    # NOTE: pivotal_step_index is NEVER in Observation — ground-truth only


class Action(BaseModel):
    """Submitted by the agent at each step."""
    action_id: str = Field(..., description="Must be in observation.available_actions")
    reasoning: Optional[str] = Field(
        None, description="Agent chain-of-thought — stored for PSIA analysis"
    )
    # Agent self-reported importance: used for CCE computation
    credit_estimate: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="How important does the agent think THIS step is? [0,1]"
    )


class StepInfo(BaseModel):
    """Detailed info returned inside StepResult."""
    step_reward:         float
    # Populated only after done=True (labels hidden during episode)
    is_pivotal_step:     Optional[bool]  = None
    ground_truth_credit: Optional[float] = None
    psia_running:        float = 0.0   # Session PSIA so far
    cce_running:         float = 0.0   # Session CCE so far
    episode_outcome:     Optional[str]  = None


class StepResult(BaseModel):
    """Returned by step()."""
    observation: Observation
    reward: float
    done: bool
    info: StepInfo


class State(BaseModel):
    """Full episode state returned by state(). Causal labels revealed after done."""
    episode_id: str
    domain: str
    tier: str
    t_total: int
    step_count: int
    max_steps: int
    cumulative_reward: float
    episode_complete: bool
    outcome: Optional[str] = None

    # Causal metadata — only after done=True
    pivotal_step_indices:    Optional[List[int]]        = None
    pivotal_actions:         Optional[List[str]]        = None
    causal_chain:            Optional[List[str]]        = None
    decoy_steps:             Optional[List[int]]        = None
    counterfactual_outcomes: Optional[Dict[str, str]]   = None
    causal_faithfulness:     Optional[float]            = None
    min_steps_needed:        Optional[int]              = None

    # Session metrics
    session_psia:       float = 0.0
    session_cce:        float = 0.0
    session_tsr:        float = 0.0
    session_mpcs:       float = 0.0
    episodes_completed: int   = 0
