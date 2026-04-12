"""
CreditMaze — Main Environment Class
Implements the full OpenEnv spec: reset(), step(), state().

Credit assignment is RETROSPECTIVE:
- During play, credit_estimate per step is stored as a fallback only.
- After done=True, the agent calls submit_retrospective_credits() with a
  {step_index: credit} map for the whole trajectory.
- PSIA/CCE/MPCS are computed from retrospective credits when available,
  falling back to per-step forward guesses only if no retrospective call
  was made (e.g. random baseline).
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
        # List of (action_id, step_index, credit_estimate) per step
        self.step_history: List[Tuple[str, int, Optional[float]]] = []
        self.resolved_pivots = set()
        self.episode_metrics: Dict[str, object] = {}
        # Retrospective credit map submitted after done=True: {step_idx: float}
        self.retrospective_credits: Optional[Dict[int, float]] = None
        self.credit_source: Optional[str] = None   # "retrospective" | "forward"

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
    submit_retrospective_credits(episode_id, credits) → dict  (NEW)
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
        episode = self._generate_validated(tier, domain, seed)
        episode_id = self._make_episode_id(episode, seed)
        episode.episode_id = episode_id

        ep_state = EpisodeState(episode, episode_id)
        self._episodes[episode_id] = ep_state

        return self._make_obs(ep_state)

    def _generate_validated(self, tier, domain, seed) -> Episode:
        for attempt in range(self.max_regen_attempts):
            s = (seed + attempt) if seed is not None else None
            episode = self.engine.generate(tier=tier, domain=domain, seed=s)
            faith   = self.simulator.validate(episode)
            if faith >= 0.95:
                return episode
        return episode

    # ── step ──────────────────────────────────────────────────────────────────

    def step(self, episode_id: str, action: Action) -> StepResult:
        ep = self._get_episode(episode_id)
        if ep.done:
            raise ValueError(f"Episode {episode_id} is already complete.")

        step_idx   = ep.step_count
        is_pivotal = step_idx in set(ep.episode.pivotal_indices)

        available    = ep.current_step.get("available_actions", [])
        action_id    = action.action_id
        invalid_action = bool(available and action_id not in available)

        # Store (action_id, step_index, forward_credit_estimate)
        ep.step_history.append((action_id, step_idx, action.credit_estimate))
        ep.step_count += 1

        if invalid_action:
            outcome = "failure"
        else:
            outcome, _ = self.simulator.advance(ep.episode, step_idx, action_id)

        pivot_set = set(ep.episode.pivotal_indices)
        if step_idx in pivot_set and outcome != "failure":
            ep.resolved_pivots.add(step_idx)

        if outcome == "success" and step_idx < ep.episode.t_total - 1:
            outcome = "in_progress"

        if step_idx == ep.episode.t_total - 1 and outcome == "in_progress":
            outcome = "success" if ep.resolved_pivots == pivot_set else "failure"

        if ep.step_count >= ep.episode.t_total and outcome == "in_progress":
            outcome = "failure"

        reward = compute_step_reward(
            outcome=outcome,
            step_idx=step_idx,
            is_pivotal=is_pivotal,
            n_steps_taken=ep.step_count,
            max_steps=ep.episode.max_steps,
        )

        ep.cumulative_reward += reward
        ep.outcome = outcome

        done = (
            outcome in ("success", "failure")
            or ep.step_count >= ep.episode.max_steps
        )
        ep.done = done

        # On episode end: compute metrics using whatever credits we have.
        # If retrospective credits arrive later (via submit_retrospective_credits),
        # metrics will be recomputed and replaced.
        gt_labels  = None
        ep_metrics = {}
        if done:
            gt_labels  = compute_gt_labels(ep.episode)
            # Use forward guesses as initial fallback
            forward_history = [(a, c) for a, _, c in ep.step_history]
            ep_metrics = self.metrics.record(
                episode=ep.episode,
                step_history=forward_history,
                gt_labels=gt_labels,
                outcome=ep.outcome,
            )
            ep.episode_metrics = ep_metrics
            ep.credit_source = "forward"

        obs = self._make_obs(ep, last_action=action_id, last_reward=reward)

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

    # ── submit_retrospective_credits ──────────────────────────────────────────

    def submit_retrospective_credits(
        self,
        episode_id: str,
        credits: Dict[str, float],   # {step_index_str: credit_float}
    ) -> dict:
        """
        PRIMARY credit signal. Call once after done=True with a map of
        step_index → credit [0,1] for the entire trajectory.

        Recomputes PSIA/CCE/MPCS using these retrospective estimates and
        replaces the forward-guess-based metrics recorded at episode end.
        """
        ep = self._get_episode(episode_id)
        if not ep.done:
            raise ValueError("Episode is not complete. Cannot submit retrospective credits yet.")

        # Parse and store
        retro: Dict[int, float] = {}
        for k, v in credits.items():
            try:
                idx = int(k)
                retro[idx] = float(max(0.0, min(1.0, v)))
            except (ValueError, TypeError):
                pass

        ep.retrospective_credits = retro
        ep.credit_source = "retrospective"

        # Build step_history with retrospective credits replacing forward guesses
        retro_history = [
            (action_id, retro.get(step_idx))
            for action_id, step_idx, _ in ep.step_history
        ]

        gt_labels = compute_gt_labels(ep.episode)

        # Remove the previously-recorded forward-guess metrics entry and replace
        self.metrics.replace_last(
            episode=ep.episode,
            step_history=retro_history,
            gt_labels=gt_labels,
            outcome=ep.outcome,
        )

        ep.episode_metrics = self.metrics.last_episode_metrics

        return {
            "episode_id":   episode_id,
            "credit_source": "retrospective",
            "n_steps_credited": len(retro),
            "psia": round(self.metrics.psia, 4),
            "cce":  round(self.metrics.cce,  4),
            "mpcs": round(self.metrics.mpcs, 4) if self.metrics.mpcs is not None else None,
            "attribution_gap": ep.episode_metrics.get("attribution_gap"),
            "psia_score": ep.episode_metrics.get("psia"),
            "top_attributed_step": ep.episode_metrics.get("top_attributed_step"),
            "pivotal_step_rank":   ep.episode_metrics.get("pivotal_step_rank"),
        }

    # ── state ─────────────────────────────────────────────────────────────────

    def state(self, episode_id: str) -> State:
        ep   = self._get_episode(episode_id)
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
            session_mpcs=round(self.metrics.mpcs, 4) if self.metrics.mpcs is not None else None,
            episodes_completed=self.metrics.n_complete,
            top_attributed_step=ep.episode_metrics.get("top_attributed_step") if done else None,
            top_attributed_action=ep.episode_metrics.get("top_attributed_action") if done else None,
            top_attributed_credit=ep.episode_metrics.get("top_attributed_credit") if done else None,
            pivotal_step_rank=ep.episode_metrics.get("pivotal_step_rank") if done else None,
            false_positive_steps=ep.episode_metrics.get("false_positive_steps") if done else None,
            attribution_gap=ep.episode_metrics.get("attribution_gap") if done else None,
            success_with_wrong_attribution=ep.episode_metrics.get("success_with_wrong_attribution") if done else None,
            credit_source=ep.credit_source if done else None,
        )

    def normalized_score(self, episode_id: str) -> float:
        ep = self._get_episode(episode_id)
        max_reward = self._max_possible_reward(ep.episode)
        if max_reward <= 0:
            return 0.01
        score = ep.cumulative_reward / max_reward
        return round(min(max(score, 0.01), 0.99), 4)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _get_episode(self, episode_id: str) -> EpisodeState:
        if episode_id not in self._episodes:
            raise KeyError(f"Episode '{episode_id}' not found. Call /reset first.")
        return self._episodes[episode_id]

    def _make_episode_id(self, episode: Episode, seed: Optional[int]) -> str:
        seed_part = "none" if seed is None else str(seed)
        fingerprint = "|".join([
            episode.tier, episode.domain, seed_part,
            ",".join(map(str, episode.pivotal_indices)),
            ",".join(episode.pivotal_actions),
            str(episode.t_total),
        ])
        return hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:8]

    def _max_possible_reward(self, episode: Episode) -> float:
        pivot_set = set(episode.pivotal_indices)
        total = 0.0
        for step_idx in range(episode.t_total):
            if step_idx == episode.t_total - 1:
                total += compute_step_reward(
                    outcome="success", step_idx=step_idx,
                    is_pivotal=(step_idx in pivot_set),
                    n_steps_taken=episode.t_total, max_steps=episode.max_steps,
                )
            else:
                total += compute_step_reward(
                    outcome="in_progress", step_idx=step_idx,
                    is_pivotal=(step_idx in pivot_set),
                    n_steps_taken=step_idx + 1, max_steps=episode.max_steps,
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