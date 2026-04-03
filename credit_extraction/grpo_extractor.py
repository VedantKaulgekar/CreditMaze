"""
CreditMaze — GRPO Credit Extractor
Extracts per-step credit via group-relative advantage estimation.
Requires G rollouts of the same episode to compare.
"""
from __future__ import annotations
from typing import List, Dict, Optional
from .hook import CreditExtractor


class GRPOExtractor(CreditExtractor):
    """
    GRPO credit extraction via group-relative reward comparison.

    For each step t, compute the average reward of rollouts that took
    action A at step t minus those that took a different action.
    Steps where action choice most consistently separates high/low reward
    trajectories get the highest credit.

    Known limitation (exposed by CreditMaze):
    GRPO cannot distinguish pivotal steps from high-immediate-reward decoys
    because decoys with large local reward inflate their group advantage.
    """

    def __init__(self, n_rollouts: int = 8):
        self.n_rollouts = n_rollouts
        # episode_id → list of {"traj": [...], "reward": float}
        self._buffer: Dict[str, List[Dict]] = {}

    def add_rollout(
        self,
        episode_id: str,
        trajectory: List[Dict],
        total_reward: float,
    ) -> None:
        """Buffer one rollout. Call before extract()."""
        if episode_id not in self._buffer:
            self._buffer[episode_id] = []
        self._buffer[episode_id].append({"traj": trajectory, "reward": total_reward})

    def extract(
        self,
        trajectory: List[Dict],
        episode_id: str,
    ) -> Dict[int, float]:
        rollouts = self._buffer.get(episode_id, [])

        if len(rollouts) < 2:
            # Fallback: uniform credit
            return {t: 1.0 / len(trajectory) for t in range(len(trajectory))}

        mean_r = sum(r["reward"] for r in rollouts) / len(rollouts)
        credit: Dict[int, float] = {}

        for t, step in enumerate(trajectory):
            action_t = step["action"]
            same = [r for r in rollouts
                    if t < len(r["traj"]) and r["traj"][t]["action"] == action_t]
            diff = [r for r in rollouts
                    if t < len(r["traj"]) and r["traj"][t]["action"] != action_t]

            if same and diff:
                r_same = sum(r["reward"] for r in same) / len(same)
                r_diff = sum(r["reward"] for r in diff) / len(diff)
                credit[t] = max(0.0, r_same - r_diff)
            else:
                # Cannot estimate — assign mean-subtracted reward
                credit[t] = max(0.0, step["reward"] - mean_r)

        return self.normalise(credit)

    def clear(self, episode_id: str) -> None:
        self._buffer.pop(episode_id, None)
