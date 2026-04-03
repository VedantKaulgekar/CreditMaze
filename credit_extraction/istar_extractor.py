"""
CreditMaze — iStar Credit Extractor
Implements leave-one-out marginal contribution estimation.
Reference: arXiv 2509.19199 (iStar, Liu et al., Sep 2025)
"""
from __future__ import annotations
from typing import List, Dict, Optional, Callable
from .hook import CreditExtractor


class IStarExtractor(CreditExtractor):
    """
    iStar credit extraction via implicit PRM leave-one-out.

    For each step t, estimate: score(full_traj) - score(traj_without_t).
    Steps with large marginal contribution are likely causal.

    Known limitation (exposed by CreditMaze):
    When the pivotal step and a nearby decoy produce similar context states,
    their leave-one-out marginal contributions collapse to similar values.
    CreditMaze Tier 3 (adversarial) is specifically designed to expose this.
    """

    def __init__(self, scorer_fn: Optional[Callable] = None):
        """
        Args:
            scorer_fn: Callable(context_str) → float [0,1]
                       PRM that scores trajectory quality.
                       If None, uses a simple recency-weighted heuristic.
        """
        self.scorer_fn = scorer_fn

    def extract(
        self,
        trajectory: List[Dict],
        episode_id: str,
    ) -> Dict[int, float]:
        if self.scorer_fn is None:
            return self._heuristic_loo(trajectory)

        full_context  = self._build_context(trajectory)
        full_score    = self.scorer_fn(full_context)
        credit: Dict[int, float] = {}

        for t in range(len(trajectory)):
            without_t   = self._context_without(trajectory, t)
            score_wo    = self.scorer_fn(without_t)
            # Marginal contribution of step t
            credit[t]   = max(0.0, full_score - score_wo)

        return self.normalise(credit)

    # ── Context building ──────────────────────────────────────────────────────

    def _build_context(self, trajectory: List[Dict]) -> str:
        return " ".join(
            f"[{s.get('action', '')}] {s.get('reasoning', '')}"
            for s in trajectory
        )

    def _context_without(self, trajectory: List[Dict], exclude: int) -> str:
        return " ".join(
            f"[{s.get('action', '')}] {s.get('reasoning', '')}"
            for i, s in enumerate(trajectory)
            if i != exclude
        )

    # ── Heuristic fallback ────────────────────────────────────────────────────

    def _heuristic_loo(self, trajectory: List[Dict]) -> Dict[int, float]:
        """
        Without a real PRM: approximate by comparing step reward
        to average trajectory reward. Steps significantly above average
        are considered high credit.
        """
        if not trajectory:
            return {}
        rewards  = [s["reward"] for s in trajectory]
        mean_r   = sum(rewards) / len(rewards)
        std_r    = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5 or 1.0
        credit   = {}
        for t, step in enumerate(trajectory):
            # Z-score normalised reward as credit proxy
            z = (step["reward"] - mean_r) / std_r
            credit[t] = max(0.0, z)
        return self.normalise(credit) if any(v > 0 for v in credit.values()) else {t: 1.0/len(trajectory) for t in credit}
