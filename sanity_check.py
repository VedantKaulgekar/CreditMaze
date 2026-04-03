"""
CreditMaze — Pre-Submission Sanity Check
Run this before submitting to catch any disqualification risks.

Usage: python sanity_check.py
All checks must print PASSED. Any FAILED = fix before submitting.
"""
import sys
from environment.env import CreditMazeEnv
from environment.models import Action
from environment.generation.template_engine import TemplateEngine, DOMAINS
from environment.causal.counterfactual import CounterfactualSimulator
from environment.metrics import compute_gt_labels

PASS = "\033[92mPASSED\033[0m"
FAIL = "\033[91mFAILED\033[0m"
failures = []


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        failures.append(name)
    return condition


def run_episode_to_end(env, tier, seed):
    obs = env.reset(tier=tier, seed=seed)
    ep  = obs.episode_id
    for _ in range(obs.max_steps):
        result = env.step(ep, Action(
            action_id=obs.available_actions[0],
            credit_estimate=float(obs.step_count) / max(obs.t_total, 1),
        ))
        obs = result.observation
        if result.done:
            break
    return env.state(ep), result.reward


print("\n" + "=" * 60)
print("CreditMaze — Pre-Submission Sanity Check")
print("=" * 60)

# ── 1. Episode generation ─────────────────────────────────────────────────────
print("\n[1] Episode Generation")
engine    = TemplateEngine()
simulator = CounterfactualSimulator()

for domain in DOMAINS:
    try:
        ep    = engine.generate(tier="easy", domain=domain, seed=42)
        faith = simulator.validate(ep)
        check(
            f"Domain '{domain}' generates valid episode",
            faith >= 0.90,
            f"faithfulness={faith:.3f}",
        )
        check(
            f"Domain '{domain}' pivotal action in available_actions",
            all(
                ep.pivotal_actions[i] in ep.steps[ep.pivotal_indices[i]]["available_actions"]
                for i in range(len(ep.pivotal_indices))
            ),
        )
    except Exception as e:
        check(f"Domain '{domain}' generates without error", False, str(e))

# ── 2. Causal labels hidden during episode ────────────────────────────────────
print("\n[2] Causal Label Visibility")
env = CreditMazeEnv()
obs = env.reset(tier="easy", seed=42)
ep  = obs.episode_id
env.step(ep, Action(action_id=obs.available_actions[0]))
mid_state = env.state(ep)
check("Pivotal indices hidden mid-episode",    mid_state.pivotal_step_indices is None)
check("Causal chain hidden mid-episode",       mid_state.causal_chain is None)
check("Counterfactuals hidden mid-episode",    mid_state.counterfactual_outcomes is None)

# ── 3. Labels revealed after done ────────────────────────────────────────────
print("\n[3] Post-Episode Label Revelation")
state, _ = run_episode_to_end(CreditMazeEnv(), "easy", 42)
check("Episode marked complete",              state.episode_complete)
check("Pivotal indices revealed after done",  state.pivotal_step_indices is not None)
check("Causal chain revealed after done",     state.causal_chain is not None)
check("Counterfactuals revealed after done",  state.counterfactual_outcomes is not None)
check("Causal faithfulness > 0.90",           (state.causal_faithfulness or 0) >= 0.90,
      f"faithfulness={state.causal_faithfulness:.3f}")

# ── 4. Grader non-triviality ──────────────────────────────────────────────────
print("\n[4] Grader Non-Triviality")
rewards = []
for seed in range(6):
    env2 = CreditMazeEnv()
    s, r = run_episode_to_end(env2, "easy", seed)
    rewards.append(round(r, 4))
unique_rewards = len(set(rewards))
check("Rewards vary across episodes", unique_rewards > 1,
      f"unique values: {unique_rewards}/6, values: {rewards}")

psias = []
for seed in range(6):
    env3 = CreditMazeEnv()
    obs3 = env3.reset(tier="easy", seed=seed)
    ep3  = obs3.episode_id
    for _ in range(obs3.max_steps):
        result = env3.step(ep3, Action(
            action_id=obs3.available_actions[0],
            credit_estimate=float(obs3.step_count) / max(obs3.t_total, 1),
        ))
        obs3 = result.observation
        if result.done:
            break
    psias.append(round(env3.state(ep3).session_psia, 4))
check("PSIA varies across episodes", len(set(psias)) > 1,
      f"unique PSIA values: {len(set(psias))}/6")

# ── 5. Multi-tier difficulty gradient ─────────────────────────────────────────
print("\n[5] Difficulty Gradient")
tier_psias = {}
for tier in ["easy", "medium", "hard"]:
    psia_list = []
    for seed in range(3):
        e   = CreditMazeEnv()
        obs = e.reset(tier=tier, seed=seed)
        ep  = obs.episode_id
        for _ in range(obs.max_steps):
            # Naive recency-biased agent: credit proportional to step index
            result = e.step(ep, Action(
                action_id=obs.available_actions[0],
                credit_estimate=float(obs.step_count) / max(obs.t_total, 1),
            ))
            obs = result.observation
            if result.done:
                break
        psia_list.append(e.state(ep).session_psia)
    tier_psias[tier] = sum(psia_list) / len(psia_list)

check("Easy PSIA > Hard PSIA (difficulty gradient exists)",
      tier_psias["easy"] >= tier_psias["hard"],
      f"easy={tier_psias['easy']:.3f} medium={tier_psias['medium']:.3f} hard={tier_psias['hard']:.3f}")

# ── 6. Observation schema compliance ─────────────────────────────────────────
print("\n[6] OpenEnv Schema Compliance")
env4 = CreditMazeEnv()
obs4 = env4.reset(tier="easy", seed=42)
obs_dict = obs4.model_dump()

required_obs_fields = ["episode_id", "domain", "tier", "t_total", "step_count",
                       "max_steps", "context", "available_actions"]
for field in required_obs_fields:
    check(f"Observation has field '{field}'", field in obs_dict)

check("Observation does NOT expose pivotal_step_indices",
      "pivotal_step_indices" not in obs_dict)

# Step and check StepResult fields
result4 = env4.step(obs4.episode_id, Action(action_id=obs4.available_actions[0]))
result_dict = result4.model_dump()
for field in ["observation", "reward", "done", "info"]:
    check(f"StepResult has field '{field}'", field in result_dict)
check("Reward in [0, 1]", 0.0 <= result4.reward <= 1.0, f"reward={result4.reward}")

# ── 7. Session metrics range check ───────────────────────────────────────────
print("\n[7] Session Metrics Range")
env5 = CreditMazeEnv()
for seed in range(4):
    s, _ = run_episode_to_end(env5, "easy", seed)

final_state = s
check("Session PSIA in [0,1]", 0.0 <= final_state.session_psia <= 1.0,
      f"psia={final_state.session_psia}")
check("Session CCE in [0,1]",  0.0 <= final_state.session_cce  <= 1.0,
      f"cce={final_state.session_cce}")
check("Session TSR in [0,1]",  0.0 <= final_state.session_tsr  <= 1.0,
      f"tsr={final_state.session_tsr}")
check("Episodes completed = 4", final_state.episodes_completed == 4,
      f"got {final_state.episodes_completed}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if failures:
    print(f"\033[91m{len(failures)} check(s) FAILED:\033[0m")
    for f in failures:
        print(f"  - {f}")
    print("\nFix these before submitting.\n")
    sys.exit(1)
else:
    print(f"\033[92mAll checks PASSED — safe to submit!\033[0m\n")
    sys.exit(0)
