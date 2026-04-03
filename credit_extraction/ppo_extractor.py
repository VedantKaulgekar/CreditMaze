"""
CreditMaze — PPO Credit Extractor
Extracts per-step credit via TD-error from critic value estimates.
"""
from __future__ import annotations
from typing import List, Dict, Optional, Any
from .hook import CreditExtractor


class PPOExtractor(CreditExtractor):
    """
    PPO credit extraction via temporal difference error.

    TD-error at step t: |r_t + γ·V(s_{t+1}) - V(s_t)|
    Steps with large TD-error caused a significant value shift
    and are likely causally important.

    Known limitation (exposed by CreditMaze):
    Critic overfits to decoys with high immediate reward, causing their
    value estimates to be inflated. The TD-error at decoy steps becomes
    high too, masking the true pivotal step.
    """

    def __init__(self, critic_fn=None, gamma: float = 0.99):
        """
        Args:
            critic_fn: Callable(obs_dict) → float  (critic value estimate)
                       If None, falls back to cumulative reward heuristic.
            gamma: Discount factor.
        """
        self.critic_fn = critic_fn
        self.gamma     = gamma

    def extract(
        self,
        trajectory: List[Dict],
        episode_id: str,
    ) -> Dict[int, float]:
        if self.critic_fn is None:
            return self._heuristic_extract(trajectory)

        # Get critic values for all states
        values = []
        for step in trajectory:
            v = self.critic_fn(step.get("obs", {}))
            values.append(v)
        values.append(0.0)  # Terminal state V = 0

        # TD-error as credit proxy
        credit: Dict[int, float] = {}
        for t, step in enumerate(trajectory):
            td = abs(step["reward"] + self.gamma * values[t + 1] - values[t])
            credit[t] = td

        return self.normalise(credit)

    def _heuristic_extract(self, trajectory: List[Dict]) -> Dict[int, float]:
        """
        Fallback without critic: use cumulative reward from each step onward.
        Steps earlier in successful trajectories get more credit.
        """
        T      = len(trajectory)
        credit = {}
        suffix_reward = 0.0
        for t in reversed(range(T)):
            suffix_reward = trajectory[t]["reward"] + self.gamma * suffix_reward
            credit[t]     = suffix_reward
        return self.normalise(credit)
