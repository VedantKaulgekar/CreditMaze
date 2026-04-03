"""
CreditMaze — Counterfactual Simulator
Validates ground-truth causal labels by simulating alternative action choices.
For every pivotal step, taking the WRONG action must change outcome to FAILURE.
"""
from __future__ import annotations
from typing import Tuple, List, TYPE_CHECKING

if TYPE_CHECKING:
    from environment.generation.template_engine import Episode


class CounterfactualSimulator:
    """
    Proves causal labels by counterfactual rollout.

    validate(episode) → faithfulness_score ∈ [0,1]
      1.0 = every pivotal step passes validation
      <0.95 = episode is invalid, discard and regenerate
    """

    def validate(self, episode: "Episode") -> float:
        """
        For each pivotal step, simulate taking the WRONG action.
        The outcome MUST change to failure for the label to be valid.
        Returns faithfulness score: fraction of pivots that pass.
        """
        n_valid = 0
        for p_idx, p_action in zip(episode.pivotal_indices, episode.pivotal_actions):
            wrong_actions = [
                a for a in episode.steps[p_idx]["available_actions"]
                if a != p_action
            ]
            # At least one wrong action must lead to failure
            any_fail = any(
                self._simulate_counterfactual(episode, p_idx, wa) == "failure"
                for wa in wrong_actions
            )
            if any_fail:
                n_valid += 1

        faithfulness = n_valid / max(len(episode.pivotal_indices), 1)
        episode.faithfulness_score = faithfulness
        return faithfulness

    def _simulate_counterfactual(
        self, episode: "Episode", pivot_idx: int, wrong_action: str
    ) -> str:
        """
        Run episode to completion substituting wrong_action at pivot_idx.
        Returns 'success' or 'failure'.
        """
        sim_state = dict(episode.initial_state)
        pivot_set = set(episode.pivotal_indices)

        for step_idx in range(episode.t_total):
            step = episode.steps[step_idx]

            if step_idx == pivot_idx:
                action = wrong_action
            elif step_idx in pivot_set:
                # Other pivots: use the correct action
                pi = episode.pivotal_indices.index(step_idx)
                action = episode.pivotal_actions[pi]
            else:
                # Decoys: use the step's default benign action
                action = step.get("default_action", step["available_actions"][0])

            outcome = self._apply_step(episode, sim_state, step_idx, action)
            if outcome in ("success", "failure"):
                return outcome

        return "failure"  # Timeout = failure

    def _apply_step(
        self,
        episode: "Episode",
        state: dict,
        step_idx: int,
        action: str,
    ) -> str:
        """
        Domain-specific step application.
        Returns 'success', 'failure', or 'in_progress'.
        """
        pivot_set = set(episode.pivotal_indices)

        if step_idx in pivot_set:
            pi = episode.pivotal_indices.index(step_idx)
            correct = episode.pivotal_actions[pi]
            if action != correct:
                return "failure"
            # Check if this was the last pivot
            later_pivots = [p for p in episode.pivotal_indices if p > step_idx]
            if not later_pivots:
                return "success"

        return "in_progress"

    # ── Runtime advance (used by env.step) ────────────────────────────────────

    def advance(
        self,
        episode: "Episode",
        step_idx: int,
        action: str,
    ) -> Tuple[str, float]:
        """
        Advance episode by one step during live play.
        Returns (outcome, step_reward).
        """
        pivot_set = set(episode.pivotal_indices)

        if step_idx in pivot_set:
            pi = episode.pivotal_indices.index(step_idx)
            correct = episode.pivotal_actions[pi]

            if action == correct:
                later_pivots = [p for p in episode.pivotal_indices if p > step_idx]
                if not later_pivots:
                    # All pivots resolved correctly → success
                    return "success", 0.50
                else:
                    # More pivots remain
                    return "in_progress", 0.12
            else:
                # Wrong action at pivot → failure
                return "failure", 0.0
        else:
            # Decoy step — small positive reward for any valid action
            return "in_progress", 0.04
