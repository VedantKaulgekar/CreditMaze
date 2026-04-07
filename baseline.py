"""
CreditMaze — Baseline Inference Script
Compatible with: openai >=2.7, httpx >=0.28

Usage:
  python baseline.py                      # 5 episodes per tier
  python baseline.py --n 1 --tiers easy   # quick smoke test
  python baseline.py --model gpt-4o-mini  # choose model explicitly

If OPENAI_API_KEY is not set, runs a random-policy baseline automatically.
"""
from __future__ import annotations
import os, json, argparse, random
import httpx

ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY      = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME") or os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-72B-Instruct")
N_EPISODES = 5
_LLM_DISABLED = False

# Lazy-import openai so script works without it installed
try:
    from openai import OpenAI
    _openai_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY, max_retries=0) if API_KEY else None
except ImportError:
    _openai_client = None

SYSTEM_PROMPT = """You are an expert reasoning agent completing multi-step decision tasks.

At each step:
1. Read the context carefully
2. Identify which action is most likely CAUSALLY important to the final outcome
   (not just locally rewarding — most steps are decoys with small positive reward)
3. Estimate how important THIS step is [0.0-1.0]

Key insight: Only 1-2 steps out of many actually determine the outcome.
High immediate reward does NOT mean the step is causally pivotal.

Respond ONLY with valid JSON (no markdown fences):
{"action_id": "...", "reasoning": "...", "credit_estimate": 0.0}"""


def _call_agent(obs: dict, model: str) -> dict:
    """Call LLM agent or fall back to random baseline."""
    global _LLM_DISABLED
    prompt = (
        f"Domain: {obs['domain']} | Tier: {obs['tier']}\n"
        f"Step {obs['step_count'] + 1} of {obs['t_total']} total steps\n\n"
        f"Context:\n{obs['context']}\n\n"
        f"Available actions: {obs['available_actions']}\n\n"
        "Remember: only 1-2 steps are causally pivotal. "
        "Most steps are decoys with small positive reward. "
        "Identify which step actually changes the outcome."
    )

    if _openai_client and not _LLM_DISABLED:
        try:
            response = _openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"  [warn] OpenAI call failed for model '{model}': {e} - using random fallback")
            error_text = str(e).lower()
            if "insufficient_quota" in error_text or "429" in error_text:
                _LLM_DISABLED = True

    # Random baseline (no API key, or API error)
    return {
        "action_id":       random.choice(obs["available_actions"]),
        "reasoning":       "Random baseline",
        "credit_estimate": round(random.random(), 2),
    }


def run_episode(http: httpx.Client, tier: str, domain: str, seed: int, model: str) -> dict:
    """Run one complete episode. Returns per-episode metrics."""
    # Reset
    obs = http.post("/reset", json={"tier": tier, "domain": domain, "seed": seed}).json()
    episode_id   = obs["episode_id"]
    total_reward = 0.0

    for _ in range(obs["max_steps"]):
        decision  = _call_agent(obs, model=model)
        action_id = decision.get("action_id", obs["available_actions"][0])

        # Validate action
        if action_id not in obs["available_actions"]:
            action_id = obs["available_actions"][0]

        result = http.post("/step", json={
            "episode_id":      episode_id,
            "action_id":       action_id,
            "reasoning":       decision.get("reasoning", ""),
            "credit_estimate": float(decision.get("credit_estimate", 0.5)),
        }).json()

        total_reward += result["reward"]
        obs           = result["observation"]
        if result["done"]:
            break

    # Grader — POST with JSON body (compatible with updated server.py)
    grader = http.post("/grader", json={"episode_id": episode_id}).json()

    return {
        "outcome":     obs.get("episode_outcome", "failure"),
        "reward":      round(total_reward, 4),
        "psia":        grader.get("session_psia", 0.0),
        "cce":         grader.get("session_cce",  0.5),
        "tsr":         grader.get("session_tsr",  0.0),
        "mpcs":        grader.get("session_mpcs", 0.0),
    }


def run_tier(tier: str, n: int = N_EPISODES, model: str = MODEL_NAME) -> dict:
    """Run N episodes for one tier and aggregate."""
    domains  = ["corridor", "research", "debugging", "resource", "triage"]
    results  = []

    with httpx.Client(base_url=ENV_URL, timeout=120) as http:
        for i in range(n):
            domain = domains[i % len(domains)]
            try:
                r = run_episode(http, tier=tier, domain=domain, seed=42 + i, model=model)
                results.append(r)
                print(f"  ep{i+1}: {r['outcome']:10s} reward={r['reward']:.3f} psia={r['psia']:.3f}")
            except Exception as e:
                print(f"  ep{i+1}: ERROR - {e}")

    if not results:
        return {"tier": tier, "n": 0, "tsr": 0.0, "psia": 0.0, "cce": 0.5, "mpcs": 0.0}

    avg = lambda key: round(sum(r[key] for r in results) / len(results), 4)
    return {
        "tier":        tier,
        "n":           len(results),
        "tsr":         avg("tsr"),
        "psia":        avg("psia"),
        "cce":         avg("cce"),
        "mpcs":        avg("mpcs"),
        "avg_reward":  avg("reward"),
    }


def main(tiers=None, n=N_EPISODES, model=MODEL_NAME):
    tiers = tiers or ["easy", "medium", "hard", "multi-pivot"]
    all_results = {}

    mode = f"LLM baseline ({model})" if _openai_client else "Random baseline (no API key)"
    print(f"\nCreditMaze Baseline - {mode}")
    print("=" * 55)

    for tier in tiers:
        print(f"\nTier: {tier}  ({n} episodes)")
        result = run_tier(tier, n=n, model=model)
        all_results[tier] = result
        print(f"  -> TSR={result['tsr']:.3f}  PSIA={result['psia']:.3f}  CCE={result['cce']:.3f}  MPCS={result['mpcs']:.3f}")

    print("\n" + "=" * 55)
    print(f"{'Tier':<14} {'TSR':>6} {'PSIA':>6} {'CCE':>6} {'MPCS':>6}")
    print("-" * 42)
    for tier, r in all_results.items():
        print(f"{tier:<14} {r['tsr']:>6.3f} {r['psia']:>6.3f} {r['cce']:>6.3f} {r['mpcs']:>6.3f}")

    # Final JSON line (consumed by /baseline endpoint)
    print(json.dumps(all_results))
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CreditMaze Baseline Runner")
    parser.add_argument("--tiers", nargs="+", default=["easy", "medium", "hard", "multi-pivot"])
    parser.add_argument("--n", type=int, default=N_EPISODES)
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()
    main(tiers=args.tiers, n=args.n, model=args.model)
