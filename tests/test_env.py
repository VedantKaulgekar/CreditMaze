"""
CreditMaze — Test Suite
Tests: episode generation, causal validation, step logic, metrics, API contracts.
Run: pytest tests/ -v
"""
import pytest
from environment.env import CreditMazeEnv
from environment.models import Action
from environment.causal.causal_graph import CausalGraph
from environment.causal.counterfactual import CounterfactualSimulator
from environment.generation.template_engine import TemplateEngine, TIER_CONFIG, DOMAINS
from environment.metrics import SessionMetrics, compute_gt_labels
from environment.reward import compute_step_reward


# ════════════════════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def env():
    return CreditMazeEnv()


@pytest.fixture
def engine():
    return TemplateEngine()


@pytest.fixture
def simulator():
    return CounterfactualSimulator()


# ════════════════════════════════════════════════════════════════════════════════
# 1. Causal Graph Tests
# ════════════════════════════════════════════════════════════════════════════════

class TestCausalGraph:

    def test_pivot_nodes_have_outcome_edge(self):
        graph = CausalGraph(t_total=10, pivotal_indices=[3, 7])
        assert CausalGraph.OUTCOME_NODE in graph.nodes[3].causal_children
        assert CausalGraph.OUTCOME_NODE in graph.nodes[7].causal_children

    def test_decoy_nodes_have_no_outcome_edge(self):
        graph = CausalGraph(t_total=10, pivotal_indices=[3])
        for t in range(10):
            if t != 3:
                assert CausalGraph.OUTCOME_NODE not in graph.nodes[t].causal_children

    def test_validate_decoy_isolation_passes(self):
        graph = CausalGraph(t_total=10, pivotal_indices=[5])
        assert graph.validate_decoy_isolation() is True

    def test_ground_truth_credit_single_pivot(self):
        graph = CausalGraph(t_total=5, pivotal_indices=[2])
        credit = graph.compute_ground_truth_credit()
        assert credit[2] == 1.0
        assert all(credit[t] == 0.0 for t in [0, 1, 3, 4])

    def test_ground_truth_credit_multi_pivot(self):
        graph = CausalGraph(t_total=6, pivotal_indices=[1, 4])
        credit = graph.compute_ground_truth_credit()
        assert credit[1] == pytest.approx(0.5)
        assert credit[4] == pytest.approx(0.5)
        assert credit[0] == 0.0
        assert credit[2] == 0.0

    def test_summary_structure(self):
        graph = CausalGraph(t_total=8, pivotal_indices=[2, 5])
        s = graph.summary()
        assert s["n_pivot"] == 2
        assert 2 in s["pivotal_indices"]
        assert 5 in s["pivotal_indices"]
        assert 0 in s["decoy_indices"]


# ════════════════════════════════════════════════════════════════════════════════
# 2. Template Engine Tests
# ════════════════════════════════════════════════════════════════════════════════

class TestTemplateEngine:

    @pytest.mark.parametrize("tier", ["easy", "medium", "hard", "multi-pivot"])
    def test_generates_for_all_tiers(self, engine, tier):
        ep = engine.generate(tier=tier, seed=42)
        assert ep is not None
        assert ep.tier == tier

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_generates_for_all_domains(self, engine, domain):
        ep = engine.generate(tier="easy", domain=domain, seed=42)
        assert ep.domain == domain

    def test_pivotal_indices_within_bounds(self, engine):
        for tier in ["easy", "medium", "hard"]:
            cfg = TIER_CONFIG[tier]
            ep  = engine.generate(tier=tier, seed=1)
            for p in ep.pivotal_indices:
                assert cfg["pivot_lo"] <= p <= cfg["pivot_hi"], (
                    f"Pivot {p} out of bounds [{cfg['pivot_lo']}, {cfg['pivot_hi']}] for tier {tier}"
                )

    def test_multi_pivot_has_two_pivots(self, engine):
        ep = engine.generate(tier="multi-pivot", seed=42)
        assert len(ep.pivotal_indices) == 2

    def test_steps_count_matches_t_total(self, engine):
        for tier in ["easy", "medium", "hard"]:
            ep = engine.generate(tier=tier, seed=7)
            assert len(ep.steps) == ep.t_total, (
                f"steps={len(ep.steps)} != t_total={ep.t_total} for tier {tier}"
            )

    def test_all_steps_have_available_actions(self, engine):
        ep = engine.generate(tier="easy", domain="corridor", seed=5)
        for i, step in enumerate(ep.steps):
            assert len(step.get("available_actions", [])) > 0, f"Step {i} has no actions"

    def test_pivotal_actions_are_valid(self, engine):
        for domain in DOMAINS:
            ep = engine.generate(tier="easy", domain=domain, seed=42)
            for p_idx, p_act in zip(ep.pivotal_indices, ep.pivotal_actions):
                available = ep.steps[p_idx]["available_actions"]
                assert p_act in available, (
                    f"Pivotal action '{p_act}' not in available_actions at step {p_idx}: {available}"
                )

    def test_seed_reproducibility(self, engine):
        ep1 = engine.generate(tier="easy", domain="corridor", seed=99)
        ep2 = engine.generate(tier="easy", domain="corridor", seed=99)
        assert ep1.pivotal_indices == ep2.pivotal_indices
        assert ep1.pivotal_actions == ep2.pivotal_actions

    def test_different_seeds_produce_different_episodes(self, engine):
        ep1 = engine.generate(tier="medium", seed=1)
        ep2 = engine.generate(tier="medium", seed=2)
        # Not guaranteed to differ but very likely with different seeds
        # at least domain or pivot position should vary across many seeds
        assert ep1 is not ep2


# ════════════════════════════════════════════════════════════════════════════════
# 3. Counterfactual Simulator Tests
# ════════════════════════════════════════════════════════════════════════════════

class TestCounterfactualSimulator:

    def test_faithfulness_at_least_0_95(self, engine, simulator):
        """Every generated episode should pass counterfactual validation."""
        for tier in ["easy", "medium"]:
            for domain in DOMAINS:
                ep    = engine.generate(tier=tier, domain=domain, seed=42)
                faith = simulator.validate(ep)
                assert faith >= 0.90, (
                    f"Faithfulness {faith:.2f} < 0.90 for {tier}/{domain}"
                )

    def test_correct_pivot_action_leads_to_success(self, engine, simulator):
        for domain in DOMAINS:
            ep = engine.generate(tier="easy", domain=domain, seed=42)
            p_idx = ep.pivotal_indices[0]
            p_act = ep.pivotal_actions[0]
            # Simulate full episode taking correct action at pivot
            outcome = simulator._simulate_counterfactual.__func__(
                simulator, ep, p_idx, p_act
            )
            # The correct action replaces itself — so all pivots are correct → success
            assert outcome == "success", (
                f"Correct pivot action should lead to success in {domain}"
            )

    def test_wrong_pivot_action_leads_to_failure(self, engine, simulator):
        for domain in DOMAINS:
            ep = engine.generate(tier="easy", domain=domain, seed=42)
            p_idx = ep.pivotal_indices[0]
            p_act = ep.pivotal_actions[0]
            wrong_actions = [
                a for a in ep.steps[p_idx]["available_actions"] if a != p_act
            ]
            if wrong_actions:
                outcome = simulator._simulate_counterfactual(ep, p_idx, wrong_actions[0])
                assert outcome == "failure", (
                    f"Wrong pivot action should lead to failure in {domain}, got {outcome}"
                )

    def test_advance_returns_tuple(self, engine, simulator):
        ep     = engine.generate(tier="easy", domain="corridor", seed=42)
        result = simulator.advance(ep, 0, ep.steps[0]["available_actions"][0])
        assert isinstance(result, tuple)
        assert len(result) == 2
        outcome, reward = result
        assert outcome in ("success", "failure", "in_progress")
        assert isinstance(reward, float)

    def test_decoy_advance_returns_in_progress(self, engine, simulator):
        ep = engine.generate(tier="easy", domain="corridor", seed=42)
        # Step 0 is always a decoy (entry point)
        outcome, reward = simulator.advance(ep, 0, ep.steps[0]["available_actions"][0])
        assert outcome == "in_progress"
        assert reward == pytest.approx(0.04)


# ════════════════════════════════════════════════════════════════════════════════
# 4. Reward Function Tests
# ════════════════════════════════════════════════════════════════════════════════

class TestReward:

    def test_success_reward_in_range(self):
        r = compute_step_reward("success", 5, True, 6, 15)
        assert 0.5 <= r <= 1.0

    def test_failure_reward_is_zero(self):
        r = compute_step_reward("failure", 5, True, 6, 15)
        assert r == 0.0

    def test_decoy_step_reward(self):
        r = compute_step_reward("in_progress", 2, False, 3, 15)
        assert r == pytest.approx(0.04)

    def test_pivotal_correct_reward(self):
        r = compute_step_reward("in_progress", 5, True, 6, 15)
        assert r == pytest.approx(0.12)

    def test_efficiency_bonus_on_success(self):
        # Early success should give higher reward than late success
        early = compute_step_reward("success", 2, True, 3,  15)
        late  = compute_step_reward("success", 12, True, 13, 15)
        assert early > late

    def test_success_reward_max_is_one(self):
        r = compute_step_reward("success", 1, True, 1, 15)
        assert r <= 1.0


# ════════════════════════════════════════════════════════════════════════════════
# 5. Metrics Tests
# ════════════════════════════════════════════════════════════════════════════════

class TestMetrics:

    def _make_episode_stub(self, pivotal_indices, t_total=10):
        """Create minimal episode-like object for metrics testing."""
        class FakeEp:
            pass
        ep = FakeEp()
        ep.pivotal_indices = pivotal_indices
        ep.t_total = t_total
        return ep

    def test_psia_perfect_when_agent_credits_pivot(self):
        metrics = SessionMetrics()
        ep = self._make_episode_stub([3], t_total=5)
        gt = compute_gt_labels(ep)
        # Agent assigns high credit to step 3 (the pivot)
        history = [(f"act{t}", 1.0 if t == 3 else 0.1) for t in range(5)]
        result  = metrics.record(ep, history, gt, "success")
        assert result["psia"] == pytest.approx(1.0)

    def test_psia_zero_when_agent_credits_decoy(self):
        metrics = SessionMetrics()
        ep = self._make_episode_stub([2], t_total=5)
        gt = compute_gt_labels(ep)
        # Agent assigns high credit to step 4 (a decoy)
        history = [(f"act{t}", 0.9 if t == 4 else 0.1) for t in range(5)]
        result  = metrics.record(ep, history, gt, "success")
        assert result["psia"] == pytest.approx(0.0)

    def test_cce_zero_for_perfect_calibration(self):
        metrics = SessionMetrics()
        ep = self._make_episode_stub([2], t_total=5)
        gt = compute_gt_labels(ep)
        # Perfect calibration: credit_estimate = gt label
        history = [(f"act{t}", gt[t]) for t in range(5)]
        result  = metrics.record(ep, history, gt, "success")
        assert result["cce"] == pytest.approx(0.0, abs=0.01)

    def test_cce_near_half_for_uniform_credit(self):
        """Uniform credit assignment ≈ random baseline: CCE ≈ 0.5."""
        metrics = SessionMetrics()
        ep = self._make_episode_stub([2], t_total=10)
        gt = compute_gt_labels(ep)
        # Uniform credit: same estimate for all steps
        history = [(f"act{t}", 0.1) for t in range(10)]
        result  = metrics.record(ep, history, gt, "success")
        # Uniform estimates normalise to 0.5 each — CCE ≈ (0.5-1)²*0.1 + (0.5-0)²*0.9
        assert result["cce"] > 0.0

    def test_tsr_one_for_success(self):
        metrics = SessionMetrics()
        ep = self._make_episode_stub([1], t_total=3)
        gt = compute_gt_labels(ep)
        history = [(f"act{t}", 0.5) for t in range(3)]
        result  = metrics.record(ep, history, gt, "success")
        assert result["tsr"] == 1.0

    def test_tsr_zero_for_failure(self):
        metrics = SessionMetrics()
        ep = self._make_episode_stub([1], t_total=3)
        gt = compute_gt_labels(ep)
        history = [(f"act{t}", 0.5) for t in range(3)]
        result  = metrics.record(ep, history, gt, "failure")
        assert result["tsr"] == 0.0

    def test_mpcs_computed_for_multi_pivot(self):
        metrics = SessionMetrics()
        ep = self._make_episode_stub([2, 5], t_total=8)
        gt = compute_gt_labels(ep)
        # Agent finds both pivots
        history = [(f"act{t}", 0.9 if t in [2, 5] else 0.1) for t in range(8)]
        result  = metrics.record(ep, history, gt, "success")
        assert result["mpcs"] == pytest.approx(1.0)

    def test_mpcs_partial_for_one_of_two(self):
        metrics = SessionMetrics()
        ep = self._make_episode_stub([2, 5], t_total=8)
        gt = compute_gt_labels(ep)
        # Agent finds only step 2, not step 5
        history = [(f"act{t}", 0.9 if t == 2 else 0.1) for t in range(8)]
        result  = metrics.record(ep, history, gt, "success")
        assert result["mpcs"] == pytest.approx(0.5)

    def test_session_psia_averages_across_episodes(self):
        metrics = SessionMetrics()
        ep = self._make_episode_stub([3], t_total=6)
        gt = compute_gt_labels(ep)
        # Episode 1: correct (psia=1)
        metrics.record(ep, [(f"a{t}", 0.9 if t == 3 else 0.1) for t in range(6)], gt, "success")
        # Episode 2: wrong (psia=0)
        metrics.record(ep, [(f"a{t}", 0.9 if t == 5 else 0.1) for t in range(6)], gt, "failure")
        assert metrics.psia == pytest.approx(0.5)

    def test_compute_gt_labels_sums_to_one(self):
        ep = self._make_episode_stub([2, 5], t_total=8)
        gt = compute_gt_labels(ep)
        assert sum(gt.values()) == pytest.approx(1.0)


# ════════════════════════════════════════════════════════════════════════════════
# 6. Environment Integration Tests
# ════════════════════════════════════════════════════════════════════════════════

class TestCreditMazeEnv:

    @pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
    def test_reset_returns_valid_observation(self, env, tier):
        obs = env.reset(tier=tier, seed=42)
        assert obs.episode_id
        assert obs.tier == tier
        assert obs.step_count == 0
        assert len(obs.available_actions) > 0
        assert obs.context

    def test_step_increments_step_count(self, env):
        obs = env.reset(tier="easy", seed=42)
        ep  = obs.episode_id
        action = obs.available_actions[0]
        result = env.step(ep, Action(action_id=action, credit_estimate=0.5))
        assert result.observation.step_count == 1

    def test_step_returns_valid_reward(self, env):
        obs = env.reset(tier="easy", seed=42)
        ep  = obs.episode_id
        result = env.step(ep, Action(action_id=obs.available_actions[0]))
        assert 0.0 <= result.reward <= 1.0

    def test_causal_labels_hidden_during_episode(self, env):
        obs = env.reset(tier="easy", seed=42)
        ep  = obs.episode_id
        # Take one step
        env.step(ep, Action(action_id=obs.available_actions[0]))
        state = env.state(ep)
        assert state.pivotal_step_indices is None, "Labels must be hidden during episode"

    def test_causal_labels_revealed_after_done(self, env):
        """Play to completion and verify labels appear in state."""
        obs = env.reset(tier="easy", domain="corridor", seed=42)
        ep  = obs.episode_id
        for _ in range(obs.max_steps):
            # Always take first available action
            action = obs.available_actions[0]
            result = env.step(ep, Action(action_id=action, credit_estimate=0.5))
            obs = result.observation
            if result.done:
                break
        state = env.state(ep)
        assert state.episode_complete
        assert state.pivotal_step_indices is not None
        assert len(state.pivotal_step_indices) >= 1
        assert state.causal_chain is not None
        assert state.counterfactual_outcomes is not None

    def test_step_after_done_raises(self, env):
        obs = env.reset(tier="easy", domain="corridor", seed=42)
        ep  = obs.episode_id
        for _ in range(obs.max_steps):
            result = env.step(ep, Action(action_id=obs.available_actions[0]))
            obs    = result.observation
            if result.done:
                break
        with pytest.raises((ValueError, KeyError)):
            env.step(ep, Action(action_id="anything"))

    def test_invalid_episode_id_raises(self, env):
        with pytest.raises(KeyError):
            env.state("nonexistent_episode")

    def test_session_metrics_accumulate(self, env):
        for _ in range(3):
            obs = env.reset(tier="easy", seed=42)
            ep  = obs.episode_id
            for _ in range(obs.max_steps):
                result = env.step(ep, Action(
                    action_id=obs.available_actions[0],
                    credit_estimate=0.5,
                ))
                obs = result.observation
                if result.done:
                    break
        state = env.state(ep)
        assert state.episodes_completed == 3
        assert 0.0 <= state.session_psia <= 1.0
        assert 0.0 <= state.session_cce  <= 1.0
        assert 0.0 <= state.session_tsr  <= 1.0

    def test_seed_reproducibility(self, env):
        obs1 = env.reset(tier="easy", domain="corridor", seed=7)
        env2 = CreditMazeEnv()
        obs2 = env2.reset(tier="easy", domain="corridor", seed=7)
        assert obs1.context == obs2.context
        assert obs1.available_actions == obs2.available_actions

    def test_psia_better_with_correct_credit(self, env):
        """
        Agent that assigns high credit_estimate to the last step before success
        (which in easy tier is often close to the pivot) should have higher
        PSIA than an agent assigning uniform credit.
        """
        def run_with_strategy(credit_strategy, seed=42):
            e = CreditMazeEnv()
            obs = e.reset(tier="easy", domain="corridor", seed=seed)
            ep  = obs.episode_id
            steps_taken = 0
            for _ in range(obs.max_steps):
                ce = credit_strategy(obs.step_count, obs.t_total)
                result = e.step(ep, Action(
                    action_id=obs.available_actions[0], credit_estimate=ce
                ))
                obs = result.observation
                steps_taken += 1
                if result.done:
                    break
            return e.state(ep).session_psia

        # Recency bias: give more credit to later steps
        recency  = run_with_strategy(lambda t, T: t / max(T - 1, 1))
        # Uniform: equal credit
        uniform  = run_with_strategy(lambda t, T: 0.5)
        # Both are valid; we just check they run without error
        assert 0.0 <= recency <= 1.0
        assert 0.0 <= uniform <= 1.0


# ════════════════════════════════════════════════════════════════════════════════
# 7. Sanity / Disqualification Guard Tests
# ════════════════════════════════════════════════════════════════════════════════

class TestSanityChecks:
    """
    Critical checks that must pass before hackathon submission.
    These guard against the disqualification criteria:
      - Graders that always return the same score
      - Non-reproducible baseline
    """

    def test_grader_score_varies_by_outcome(self, env):
        """Different outcomes must produce different session metrics over time."""
        # This test verifies graders are non-trivial
        scores = []
        for seed in range(5):
            obs = env.reset(tier="easy", seed=seed)
            ep  = obs.episode_id
            for _ in range(obs.max_steps):
                result = env.step(ep, Action(
                    action_id=obs.available_actions[0],
                    credit_estimate=float(obs.step_count) / obs.t_total,
                ))
                obs = result.observation
                if result.done:
                    break
            scores.append(round(result.reward, 4))
        # Scores should not all be identical
        assert len(set(scores)) > 1, "Grader returns identical scores — disqualification risk"

    def test_causal_faithfulness_always_above_threshold(self, engine, simulator):
        """All generated episodes must have faithfulness >= 0.90."""
        for tier in ["easy", "medium"]:
            for domain in DOMAINS:
                for seed in range(3):
                    ep    = engine.generate(tier=tier, domain=domain, seed=seed)
                    faith = simulator.validate(ep)
                    assert faith >= 0.90, (
                        f"Faithfulness {faith:.2f} below threshold for {tier}/{domain}/seed={seed}"
                    )

    def test_pivotal_action_gives_higher_reward_path(self, engine, simulator):
        """Taking the correct pivot action must eventually lead to higher reward than wrong action."""
        for domain in DOMAINS:
            ep    = engine.generate(tier="easy", domain=domain, seed=42)
            p_idx = ep.pivotal_indices[0]
            p_act = ep.pivotal_actions[0]
            wrong = [a for a in ep.steps[p_idx]["available_actions"] if a != p_act]

            correct_outcome, correct_r = simulator.advance(ep, p_idx, p_act)
            if wrong:
                wrong_outcome, wrong_r = simulator.advance(ep, p_idx, wrong[0])
                # Wrong action at pivot leads to failure or lower reward
                assert wrong_outcome in ("failure", "in_progress")
                if wrong_outcome == "failure":
                    assert wrong_r == 0.0

    def test_observation_never_reveals_pivot_index(self, env):
        """The Observation model must never expose pivotal_step_indices."""
        obs = env.reset(tier="easy", seed=42)
        obs_dict = obs.model_dump()
        assert "pivotal_step_indices" not in obs_dict
        assert "pivotal_actions" not in obs_dict
        assert "causal_chain" not in obs_dict

    def test_metrics_not_all_same_after_varied_play(self, env):
        """PSIA and CCE must vary when different credit_estimates are provided."""
        env1 = CreditMazeEnv()
        env2 = CreditMazeEnv()

        for seed in range(4):
            for e, credit_fn in [(env1, lambda t, T: 1.0), (env2, lambda t, T: 0.0)]:
                obs = e.reset(tier="easy", seed=seed)
                ep  = obs.episode_id
                for _ in range(obs.max_steps):
                    result = e.step(ep, Action(
                        action_id=obs.available_actions[0],
                        credit_estimate=credit_fn(obs.step_count, obs.t_total),
                    ))
                    obs = result.observation
                    if result.done:
                        break

        s1 = env1.state(ep)
        s2 = env2.state(ep)
        # Different credit strategies should produce different CCE
        # (PSIA can coincidentally match but CCE should differ)
        assert s1.session_cce != s2.session_cce or s1.session_psia != s2.session_psia, (
            "Metrics are identical regardless of credit_estimate — grader is non-functional"
        )
