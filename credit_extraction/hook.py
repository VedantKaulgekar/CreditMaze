"""
CreditMaze — Credit Extraction Hook
Algorithm-agnostic interface for extracting per-step credit estimates
from any RL training method (GRPO, PPO, iStar, HCAPO, etc.).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple


class CreditExtractor(ABC):
    """
    Base class for extracting per-step credit estimates from a trained agent.

    After an episode completes, call extract() with the full trajectory.
    Returns {step_index: credit_estimate} in [0,1] — used by SessionMetrics
    to compute PSIA and CCE against ground-truth labels.

    Usage:
        extractor = GRPOExtractor(n_rollouts=8)
        # ... collect rollouts ...
        credits = extractor.extract(trajectory, episode_id)
        metrics.record(episode, step_history_with_credits, gt_labels, outcome)
    """

    @abstractmethod
    def extract(
        self,
        trajectory: List[Dict],
        episode_id: str,
    ) -> Dict[int, float]:
        """
        Extract per-step credit estimates from trajectory.

        Args:
            trajectory: List of dicts, one per step:
                {
                    "step_idx": int,
                    "action":   str,
                    "reward":   float,
                    "obs":      dict,
                    "reasoning": Optional[str],
                }
            episode_id: Used to retrieve buffered rollouts.

        Returns:
            {step_index: credit_estimate} — values need not sum to 1.
            Will be normalised before PSIA/CCE computation.
        """
        ...

    def normalise(self, raw: Dict[int, float]) -> Dict[int, float]:
        """Min-max normalise to [0,1]."""
        if not raw:
            return {}
        mn = min(raw.values())
        mx = max(raw.values())
        if mx == mn:
            return {t: 0.5 for t in raw}
        return {t: (v - mn) / (mx - mn) for t, v in raw.items()}
