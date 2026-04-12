"""
CreditMaze — Session Metrics
Computes PSIA, CCE, TSR, and MPCS across all episodes in a session.

PSIA  — Pivotal Step Identification Accuracy
CCE   — Credit Calibration Error
TSR   — Task Success Rate
MPCS  — Multi-Pivot Coordination Score (multi-pivot tier only)

Primary credit signal is RETROSPECTIVE: agent assigns credit after seeing
full trajectory + outcome. Forward per-step guesses are a fallback only.
"""
from __future__ import annotations
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from environment.generation.template_engine import Episode


class SessionMetrics:

    def __init__(self):
        self._psia:  List[float] = []
        self._cce:   List[float] = []
        self._tsr:   List[float] = []
        self._mpcs:  List[float] = []
        self.n_complete: int = 0
        self._last_metrics: dict = {}

    # ── Record one completed episode ──────────────────────────────────────────

    def record(
        self,
        episode: "Episode",
        step_history: List[Tuple[str, Optional[float]]],  # [(action_id, credit_estimate), ...]
        gt_labels: Dict[int, float],
        outcome: str,
    ) -> dict:
        result = self._compute(episode, step_history, gt_labels, outcome)
        self._psia.append(result["psia"])
        self._cce.append(result["cce"])
        self._tsr.append(result["tsr"])
        if result["mpcs"] is not None:
            self._mpcs.append(result["mpcs"])
        self.n_complete += 1
        self._last_metrics = result
        return result

    def replace_last(
        self,
        episode: "Episode",
        step_history: List[Tuple[str, Optional[float]]],
        gt_labels: Dict[int, float],
        outcome: str,
    ) -> dict:
        """
        Replace the most recently recorded episode's metrics with recomputed
        values using retrospective credits. Called after submit_retrospective_credits().
        """
        if not self._psia:
            return self.record(episode, step_history, gt_labels, outcome)

        result = self._compute(episode, step_history, gt_labels, outcome)

        # Replace last entry in all lists
        self._psia[-1] = result["psia"]
        self._cce[-1]  = result["cce"]
        self._tsr[-1]  = result["tsr"]

        n_pivot = len(episode.pivotal_indices)
        if n_pivot > 1:
            if self._mpcs:
                self._mpcs[-1] = result["mpcs"] if result["mpcs"] is not None else self._mpcs[-1]
            elif result["mpcs"] is not None:
                self._mpcs.append(result["mpcs"])

        self._last_metrics = result
        return result

    @property
    def last_episode_metrics(self) -> dict:
        return self._last_metrics

    # ── Core computation ──────────────────────────────────────────────────────

    def _compute(
        self,
        episode: "Episode",
        step_history: List[Tuple[str, Optional[float]]],
        gt_labels: Dict[int, float],
        outcome: str,
    ) -> dict:
        n_pivot   = len(episode.pivotal_indices)
        pivot_set = set(episode.pivotal_indices)
        n_steps   = len(step_history)

        # TSR
        tsr = 1.0 if outcome == "success" else 0.0

        # Extract credit estimates
        agent_credits: Dict[int, float] = {}
        for t, (_, credit_est) in enumerate(step_history):
            if credit_est is not None:
                agent_credits[t] = credit_est
            else:
                # Uniform fallback — signals that credit is uninformative
                agent_credits[t] = 1.0 / max(n_steps, 1)

        agent_credits = _normalise(agent_credits)

        # PSIA
        if agent_credits:
            _vals    = list(agent_credits.values())
            all_equal = (max(_vals) - min(_vals)) < 1e-9
            sorted_steps = sorted(agent_credits.keys(), key=lambda t: agent_credits[t], reverse=True)
            if all_equal:
                psia_score = n_pivot / max(len(agent_credits), 1)
                top_n      = set(sorted_steps[:n_pivot])
            else:
                top_n      = set(sorted_steps[:n_pivot])
                psia_score = len(top_n & pivot_set) / n_pivot
        else:
            sorted_steps = []
            top_n        = set()
            psia_score   = 0.0

        # CCE — only over steps actually played
        sq_errors = []
        for t, gt in gt_labels.items():
            if t >= n_steps:
                continue
            est = agent_credits.get(t, 0.5)
            sq_errors.append((est - gt) ** 2)
        cce_score = float(np.mean(sq_errors)) if sq_errors else 0.5

        # MPCS (multi-pivot only)
        mpcs_score = None
        if n_pivot > 1 and agent_credits:
            top_k      = set(sorted_steps[:n_pivot])
            mpcs_score = len(top_k & pivot_set) / n_pivot

        # Diagnostics
        top_attributed_step   = sorted_steps[0] if sorted_steps else None
        top_attributed_action = (
            step_history[top_attributed_step][0]
            if top_attributed_step is not None and top_attributed_step < len(step_history)
            else None
        )
        top_attributed_credit = (
            round(agent_credits[top_attributed_step], 4)
            if top_attributed_step is not None else None
        )
        pivotal_step_rank = None
        if sorted_steps:
            for idx, step_idx in enumerate(sorted_steps, start=1):
                if step_idx in pivot_set:
                    pivotal_step_rank = idx
                    break

        non_pivots         = [t for t in agent_credits if t not in pivot_set]
        played_pivots      = [t for t in pivot_set if t in agent_credits]
        best_pivot_credit  = max((agent_credits[t] for t in played_pivots), default=0.0)
        best_decoy_credit  = max((agent_credits[t] for t in non_pivots), default=0.0)
        attribution_gap    = round(best_pivot_credit - best_decoy_credit, 4)
        false_positive_steps = sorted(top_n - pivot_set)
        success_with_wrong_attribution = (outcome == "success") and (psia_score < 1.0)

        return {
            "tsr":   tsr,
            "psia":  round(psia_score, 4),
            "cce":   round(cce_score,  4),
            "mpcs":  round(mpcs_score, 4) if mpcs_score is not None else None,
            "top_attributed_step":   top_attributed_step,
            "top_attributed_action": top_attributed_action,
            "top_attributed_credit": top_attributed_credit,
            "pivotal_step_rank":     pivotal_step_rank,
            "false_positive_steps":  false_positive_steps,
            "attribution_gap":       attribution_gap,
            "success_with_wrong_attribution": success_with_wrong_attribution,
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
    def mpcs(self) -> Optional[float]:
        return float(np.mean(self._mpcs)) if self._mpcs else None

    def summary(self) -> dict:
        return {
            "n_episodes": self.n_complete,
            "PSIA": round(self.psia, 4),
            "CCE":  round(self.cce,  4),
            "TSR":  round(self.tsr,  4),
            "MPCS": round(self.mpcs, 4) if self.mpcs is not None else None,
        }

    def reset(self):
        self._psia.clear()
        self._cce.clear()
        self._tsr.clear()
        self._mpcs.clear()
        self.n_complete = 0
        self._last_metrics = {}


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
    """Ground-truth credit labels. Uniform over pivotal steps; 0 for decoys."""
    n_pivot = len(episode.pivotal_indices)
    labels: Dict[int, float] = {}
    for t in range(episode.t_total):
        labels[t] = (1.0 / n_pivot) if t in set(episode.pivotal_indices) else 0.0
    return labels