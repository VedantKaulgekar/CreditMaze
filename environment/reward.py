"""
CreditMaze — Per-step reward function.
Rewards are designed to provide partial progress signal without revealing
which step is pivotal (that would trivially solve the credit assignment problem).
"""
from __future__ import annotations


def compute_step_reward(
    outcome: str,
    step_idx: int,
    is_pivotal: bool,
    n_steps_taken: int,
    max_steps: int,
) -> float:
    """
    Compute the reward for one step.

    Design principles:
    - ALL in-progress steps (pivot and decoy alike) give identical small
      positive reward — this is critical so reward-following agents (GRPO)
      cannot distinguish the pivot from decoys via per-step signal.
    - Wrong pivotal action gives 0 (failure endpoint).
    - Episode success terminal gives large reward + efficiency bonus.
    - Only the terminal outcome (success / failure) separates episodes,
      which forces genuine credit assignment.
    """
    if outcome == "success":
        # Terminal success: base reward + efficiency bonus
        efficiency = max(0.0, 1.0 - (n_steps_taken / max_steps))
        return round(0.50 + 0.50 * efficiency, 4)

    if outcome == "failure":
        return 0.0

    # In-progress rewards — uniform for ALL steps (pivot and decoy alike).
    # A flat per-step reward ensures reward-following agents (e.g. GRPO)
    # cannot distinguish the pivot from decoys via the reward signal alone;
    # only the terminal outcome (success / failure) separates good from bad
    # episodes, which is what forces genuine credit assignment.
    return 0.06
