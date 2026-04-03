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
    - Decoy steps give small positive reward (they "succeed" locally)
    - Correct pivotal action gives medium reward
    - Wrong pivotal action gives 0 (failure endpoint)
    - Episode success terminal gives large reward
    - Efficiency bonus: fewer steps = higher final reward

    IMPORTANT: Step rewards deliberately do NOT reveal which step is pivotal.
    The pivotal step's correct action gives 0.12 immediately — indistinguishable
    from a near-pivot decoy — so the agent cannot simply follow immediate reward.
    This is what makes credit assignment hard.
    """
    if outcome == "success":
        # Terminal success: base reward + efficiency bonus
        efficiency = max(0.0, 1.0 - (n_steps_taken / max_steps))
        return round(0.50 + 0.50 * efficiency, 4)

    if outcome == "failure":
        return 0.0

    # In-progress rewards
    if is_pivotal:
        # Correct pivotal action — but small reward (not obviously pivotal)
        return 0.12

    # Decoy step — small positive reward for valid action
    return 0.04
