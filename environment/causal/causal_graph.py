"""
CreditMaze — Causal Graph
DAG where pivotal nodes have direct edges to the outcome node.
Decoy nodes are causally isolated — no path to outcome.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Set


@dataclass
class CausalNode:
    step_index: int
    is_pivotal: bool
    causal_children: List[int] = field(default_factory=list)
    causal_parents:  List[int] = field(default_factory=list)


class CausalGraph:
    """
    Directed Acyclic Graph encoding causal structure of an episode.

    Virtual outcome node = t_total (index beyond all real steps).
    Pivotal nodes: direct edge → outcome node.
    Decoy nodes:   no path to outcome node.
    """

    OUTCOME_NODE = -1  # sentinel

    def __init__(self, t_total: int, pivotal_indices: List[int]):
        self.t_total = t_total
        self.pivotal_indices: Set[int] = set(pivotal_indices)
        self.nodes: Dict[int, CausalNode] = {}
        self._build()

    # ── Construction ──────────────────────────────────────────────────────────

    def _build(self) -> None:
        for t in range(self.t_total):
            is_pivot = t in self.pivotal_indices
            self.nodes[t] = CausalNode(
                step_index=t,
                is_pivotal=is_pivot,
                causal_children=[self.OUTCOME_NODE] if is_pivot else [],
            )

    # ── Credit assignment ─────────────────────────────────────────────────────

    def compute_ground_truth_credit(self) -> Dict[int, float]:
        """
        Ground-truth credit: equal share among pivotal steps.
        Decoy steps: 0.0.
        """
        credit: Dict[int, float] = {}
        n_pivot = len(self.pivotal_indices)
        for t in range(self.t_total):
            credit[t] = (1.0 / n_pivot) if t in self.pivotal_indices else 0.0
        return credit

    # ── Validation ────────────────────────────────────────────────────────────

    def validate_decoy_isolation(self) -> bool:
        """
        Assert no decoy node has a causal path to the outcome.
        Raises AssertionError if violated (episode should be discarded).
        """
        for t, node in self.nodes.items():
            if not node.is_pivotal:
                assert not node.causal_children, (
                    f"Decoy node {t} has causal children {node.causal_children} — invalid episode"
                )
        return True

    def get_causal_ancestors(self, node_idx: int) -> Set[int]:
        """Return all ancestors of node_idx in the causal graph."""
        ancestors: Set[int] = set()
        queue = list(self.nodes.get(node_idx, CausalNode(-1, False)).causal_parents)
        while queue:
            parent = queue.pop()
            if parent not in ancestors and parent in self.nodes:
                ancestors.add(parent)
                queue.extend(self.nodes[parent].causal_parents)
        return ancestors

    def summary(self) -> dict:
        return {
            "t_total": self.t_total,
            "pivotal_indices": sorted(self.pivotal_indices),
            "decoy_indices": [t for t in range(self.t_total) if t not in self.pivotal_indices],
            "n_pivot": len(self.pivotal_indices),
        }
