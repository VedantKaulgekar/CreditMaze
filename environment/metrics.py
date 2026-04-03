"""
CreditMaze — Session Metrics
Computes PSIA, CCE, TSR, and MPCS across all episodes in a session.
These are the three novel metrics CreditMaze introduces to the field.
"""
from __future__ import annotations
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from environment.generation.template_engine import Episode


class SessionMetrics:
    """
    Accumulates per-episode credit assignment quality metrics.

    PSIA  — Pivotal Step Identification Accuracy
    CCE   — Credit Calibration Error
    TSR   — Task Success Rate
    MPCS  — Multi-Pivot Coordination Score (Tier 4)
    """

    def __init__(self):
        self._psia:  List[float] = []
        self._cce:   List[float] = []
        self._tsr:   List[float] = []
        self._mpcs:  List[float] = []
        self.n_complete: int = 0

    # ── Record one completed episode ──────────────────────────────────────────

    def record(
        self,
        episode: "Episode",
        step_history: List[Tuple[str, Optional[float]]],  # [(action_id, credit_estimate), ...]
        gt_labels: Dict[int, float],
        outcome: str,
    ) -> dict:
        """
        Record metrics for one completed episode.
        Returns dict of per-episode metric values.
        """
        n_pivot    = len(episode.pivotal_indices)
        pivot_set  = set(episode.pivotal_indices)
        n_steps    = len(step_history)

        # ── TSR ──────────────────────────────────────────────────────────────
        tsr = 1.0 if outcome == "success" else 0.0
        self._tsr.append(tsr)

        # ── Extract agent credit estimates ────────────────────────────────────
        agent_credits: Dict[int, float] = {}
        for t, (_, credit_est) in enumerate(step_history):
            if credit_est is not None:
                agent_credits[t] = credit_est
            else:
                # Fallback: assign equal credit to all steps (naive baseline)
                agent_credits[t] = 1.0 / max(n_steps, 1)

        # Normalise to [0,1] preserving relative order
        agent_credits = _normalise(agent_credits)

        # ── PSIA ─────────────────────────────────────────────────────────────
        # Did agent assign top-N credit to the true pivotal steps?
        if agent_credits:
            sorted_steps = sorted(
                agent_credits.keys(),
                key=lambda t: agent_credits[t],
                reverse=True,
            )
            top_n      = set(sorted_steps[:n_pivot])
            psia_score = len(top_n & pivot_set) / n_pivot
        else:
            psia_score = 0.0
        self._psia.append(psia_score)

        # ── CCE ───────────────────────────────────────────────────────────────
        # MSE between agent estimates and ground-truth labels
        sq_errors = []
        for t, gt in gt_labels.items():
            est = agent_credits.get(t, 0.5)
            sq_errors.append((est - gt) ** 2)
        cce_score = float(np.mean(sq_errors)) if sq_errors else 0.5
        self._cce.append(cce_score)

        # ── MPCS (multi-pivot episodes only) ─────────────────────────────────
        mpcs_score = None
        if n_pivot > 1 and agent_credits:
            top_k      = set(sorted_steps[:n_pivot])
            mpcs_score = len(top_k & pivot_set) / n_pivot
            self._mpcs.append(mpcs_score)

        self.n_complete += 1

        return {
            "tsr":  tsr,
            "psia": round(psia_score, 4),
            "cce":  round(cce_score,  4),
            "mpcs": round(mpcs_score, 4) if mpcs_score is not None else None,
        }

    # ── Session-level aggregates ──────────────────────────────────────────────

    @property
    def psia(self) -> float:
        return float(np.mean(self._psia)) if self._psia else 0.0

    @property
    def cce(self) -> float:
        return float(np.mean(self._cce)) if self._cce else 0.5

    @property
    def tsr(self) -> float:
        return float(np.mean(self._tsr)) if self._tsr else 0.0

    @property
    def mpcs(self) -> float:
        return float(np.mean(self._mpcs)) if self._mpcs else 0.0

    def summary(self) -> dict:
        return {
            "n_episodes": self.n_complete,
            "PSIA": round(self.psia, 4),
            "CCE":  round(self.cce,  4),
            "TSR":  round(self.tsr,  4),
            "MPCS": round(self.mpcs, 4),
        }

    def reset(self):
        self._psia.clear()
        self._cce.clear()
        self._tsr.clear()
        self._mpcs.clear()
        self.n_complete = 0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalise(credits: Dict[int, float]) -> Dict[int, float]:
    """Min-max normalise credit dict to [0,1]."""
    if not credits:
        return {}
    vals = list(credits.values())
    mn, mx = min(vals), max(vals)
    if mx == mn:
        constant = min(max(mx, 0.0), 1.0)
        return {t: constant for t in credits}
    return {t: (v - mn) / (mx - mn) for t, v in credits.items()}


def compute_gt_labels(episode) -> Dict[int, float]:
    """
    Ground-truth credit labels for a completed episode.
    Uniform distribution over pivotal steps; 0 for decoys.
    """
    n_pivot = len(episode.pivotal_indices)
    labels: Dict[int, float] = {}
    for t in range(episode.t_total):
        labels[t] = (1.0 / n_pivot) if t in set(episode.pivotal_indices) else 0.0
    return labels
