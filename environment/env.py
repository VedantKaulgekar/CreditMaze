"""
CreditMaze — Main Environment Class
Implements the full OpenEnv spec: reset(), step(), state().
"""
from __future__ import annotations
import hashlib
from typing import Optional, Dict, List, Tuple

from environment.models import Observation, Action, StepResult, StepInfo, State
from environment.reward import compute_step_reward
from environment.metrics import SessionMetrics, compute_gt_labels
from environment.causal.counterfactual import CounterfactualSimulator
from environment.generation.template_engine import TemplateEngine, Episode


# ── Episode runtime state ─────────────────────────────────────────────────────

class EpisodeState:
    def __init__(self, episode: Episode, episode_id: str):
        self.episode    = episode
        self.episode_id = episode_id
        self.step_count = 0
        self.done       = False
        self.outcome    = "in_progress"
        self.cumulative_reward = 0.0
        # List of (action_id, credit_estimate) per step
        self.step_history: List[Tuple[str, Optional[float]]] = []
        self.resolved_pivots = set()

    @property
    def current_step(self) -> dict:
        idx = min(self.step_count, self.episode.t_total - 1)
        return self.episode.steps[idx]


# ── Main environment ──────────────────────────────────────────────────────────

class CreditMazeEnv:
    """
    CreditMaze RL Environment.

    reset(tier, domain, seed) → Observation
    step(episode_id, action)  → StepResult
    state(episode_id)         → State
    """

    def __init__(self, config: dict = {}):
        self.engine    = TemplateEngine()
        self.simulator = CounterfactualSimulator()
        self.metrics   = SessionMetrics()
        self._episodes: Dict[str, EpisodeState] = {}
        self.max_regen_attempts = 10

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(
        self,
        tier: str = "easy",
        domain: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Observation:
        """
        Generate a new episode, validate causal structure, return initial observation.
        Re-generates if causal faithfulness < 0.95.
        """
        episode = self._generate_validated(tier, domain, seed)
        episode_id = self._make_episode_id(episode, seed)
        episode.episode_id = episode_id

        ep_state = EpisodeState(episode, episode_id)
        self._episodes[episode_id] = ep_state

        return self._make_obs(ep_state)

    def _generate_validated(
        self,
        tier: str,
        domain: Optional[str],
        seed: Optional[int],
    ) -> Episode:
        """Generate and validate episode; retry if faithfulness < 0.95."""
        for attempt in range(self.max_regen_attempts):
            s = (seed + attempt) if seed is not None else None
            episode = self.engine.generate(tier=tier, domain=domain, seed=s)
            faith   = self.simulator.validate(episode)
            if faith >= 0.95:
                return episode
        # Use last generated episode even if faithfulness is slightly low
        return episode

    # ── step ──────────────────────────────────────────────────────────────────

    def step(self, episode_id: str, action: Action) -> StepResult:
        """
        Process one agent action.
        Returns StepResult with observation, reward, done, info.
        Ground-truth credit labels are only revealed when done=True.
        """
        ep = self._get_episode(episode_id)
        if ep.done:
            raise ValueError(f"Episode {episode_id} is already complete.")

        step_idx = ep.step_count
        is_pivotal = step_idx in set(ep.episode.pivotal_indices)

        # Validate action
        available = ep.current_step.get("available_actions", [])
        action_id = action.action_id
        invalid_action = bool(available and action_id not in available)

        # Record history
        ep.step_history.append((action_id, action.credit_estimate))
        ep.step_count += 1

        if invalid_action:
            outcome = "failure"
        else:
            # Advance causal simulator
            outcome, _ = self.simulator.advance(
                ep.episode, step_idx, action_id
            )

        pivot_set = set(ep.episode.pivotal_indices)
        if step_idx in pivot_set and outcome != "failure":
            ep.resolved_pivots.add(step_idx)

        if outcome == "success" and step_idx < ep.episode.t_total - 1:
            outcome = "in_progress"

        if step_idx == ep.episode.t_total - 1 and outcome == "in_progress":
            outcome = "success" if ep.resolved_pivots == pivot_set else "failure"

        if ep.step_count >= ep.episode.t_total and outcome == "in_progress":
            outcome = "failure"

        # Compute final step reward
        reward = compute_step_reward(
            outcome=outcome,
            step_idx=step_idx,
            is_pivotal=is_pivotal,
            n_steps_taken=ep.step_count,
            max_steps=ep.episode.max_steps,
        )

        ep.cumulative_reward += reward
        ep.outcome = outcome

        # Check termination
        done = (
            outcome in ("success", "failure")
            or ep.step_count >= ep.episode.max_steps
        )
        ep.done = done

        # Compute ground-truth labels and update session metrics after episode ends
        gt_labels    = None
        ep_metrics   = {}
        if done:
            gt_labels  = compute_gt_labels(ep.episode)
            ep_metrics = self.metrics.record(
                episode=ep.episode,
                step_history=ep.step_history,
                gt_labels=gt_labels,
                outcome=ep.outcome,
            )

        # Build observation for next step
        obs = self._make_obs(ep, last_action=action_id, last_reward=reward)

        # Populate is_pivotal_step and ground_truth_credit only after done
        is_piv_this = (step_idx in set(ep.episode.pivotal_indices)) if done else None
        gt_credit   = gt_labels.get(step_idx) if gt_labels else None

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info=StepInfo(
                step_reward=reward,
                is_pivotal_step=is_piv_this,
                ground_truth_credit=gt_credit,
                psia_running=round(self.metrics.psia, 4),
                cce_running=round(self.metrics.cce,  4),
                episode_outcome=ep.outcome if done else "in_progress",
            ),
        )

    # ── state ─────────────────────────────────────────────────────────────────

    def state(self, episode_id: str) -> State:
        """
        Return current episode state.
        Causal labels (pivotal_step_indices, causal_chain, etc.) are only
        included after done=True.
        """
        ep = self._get_episode(episode_id)
        done = ep.done
        episode = ep.episode

        return State(
            episode_id=episode_id,
            domain=episode.domain,
            tier=episode.tier,
            t_total=episode.t_total,
            step_count=ep.step_count,
            max_steps=episode.max_steps,
            cumulative_reward=round(ep.cumulative_reward, 4),
            episode_complete=done,
            outcome=ep.outcome,
            # Causal labels only after done
            pivotal_step_indices=episode.pivotal_indices if done else None,
            pivotal_actions=episode.pivotal_actions if done else None,
            causal_chain=episode.causal_chain if done else None,
            decoy_steps=episode.decoy_steps if done else None,
            counterfactual_outcomes=episode.counterfactuals if done else None,
            causal_faithfulness=episode.faithfulness_score,
            min_steps_needed=episode.min_steps,
            session_psia=round(self.metrics.psia, 4),
            session_cce=round(self.metrics.cce,  4),
            session_tsr=round(self.metrics.tsr,  4),
            session_mpcs=round(self.metrics.mpcs, 4),
            episodes_completed=self.metrics.n_complete,
        )

    def normalized_score(self, episode_id: str) -> float:
        """Return a deterministic per-episode score in [0, 1]."""
        ep = self._get_episode(episode_id)
        max_reward = self._max_possible_reward(ep.episode)
        if max_reward <= 0:
            return 0.0
        score = ep.cumulative_reward / max_reward
        return round(min(max(score, 0.0), 1.0), 4)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _get_episode(self, episode_id: str) -> EpisodeState:
        if episode_id not in self._episodes:
            raise KeyError(f"Episode '{episode_id}' not found. Call /reset first.")
        return self._episodes[episode_id]

    def _make_episode_id(self, episode: Episode, seed: Optional[int]) -> str:
        seed_part = "none" if seed is None else str(seed)
        fingerprint = "|".join([
            episode.tier,
            episode.domain,
            seed_part,
            ",".join(map(str, episode.pivotal_indices)),
            ",".join(episode.pivotal_actions),
            str(episode.t_total),
        ])
        return hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:8]

    def _max_possible_reward(self, episode: Episode) -> float:
        """Maximum achievable cumulative reward for a successful trajectory."""
        pivot_set = set(episode.pivotal_indices)
        total = 0.0
        for step_idx in range(episode.t_total):
            if step_idx == episode.t_total - 1:
                total += compute_step_reward(
                    outcome="success",
                    step_idx=step_idx,
                    is_pivotal=(step_idx in pivot_set),
                    n_steps_taken=episode.t_total,
                    max_steps=episode.max_steps,
                )
            else:
                total += compute_step_reward(
                    outcome="in_progress",
                    step_idx=step_idx,
                    is_pivotal=(step_idx in pivot_set),
                    n_steps_taken=step_idx + 1,
                    max_steps=episode.max_steps,
                )
        return round(total, 4)

    def _make_obs(
        self,
        ep: EpisodeState,
        last_action: Optional[str] = None,
        last_reward: Optional[float] = None,
    ) -> Observation:
        step_idx = min(ep.step_count, ep.episode.t_total - 1)
        step     = ep.episode.steps[step_idx]

        return Observation(
            episode_id=ep.episode_id,
            domain=ep.episode.domain,
            tier=ep.episode.tier,
            t_total=ep.episode.t_total,
            step_count=ep.step_count,
            max_steps=ep.episode.max_steps,
            context=step["context"],
            available_actions=step.get("available_actions", []),
            last_action_taken=last_action,
            last_step_reward=round(last_reward, 4) if last_reward is not None else None,
            cumulative_reward=round(ep.cumulative_reward, 4),
            episode_outcome=ep.outcome,
        )
