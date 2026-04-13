"""
CreditMaze — Baseline Inference Script
Compatible with: openai >=2.7, httpx >=0.28

Usage:
  python baseline.py                          # 3 episodes per task
  python baseline.py --n 1 --tasks task_easy  # quick smoke test
  python baseline.py --model gpt-4o-mini      # explicit model

If no API key is set, runs a reproducible random-policy baseline.

Scores are PSIA-weighted composites: 0.4*TSR + 0.4*PSIA + 0.2*(1-CCE).
This measures credit assignment quality, not just task completion.
"""
from __future__ import annotations
import os, json, argparse, random, re
import httpx

ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY      = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME") or os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-72B-Instruct")
N_EPISODES   = 3

_LLM_DISABLED = False

try:
    from openai import OpenAI
    _openai_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY, max_retries=0) if API_KEY else None
except ImportError:
    _openai_client = None

# ── Task definitions (must match server TASK_CONFIG) ─────────────────────────
TASK_CONFIGS = {
    "task_easy":         {"tier": "easy",        "domain": "triage"},
    "task_medium":       {"tier": "medium",       "domain": "research"},
    "task_hard":         {"tier": "hard",         "domain": "debugging"},
    "resource_hard":     {"tier": "hard",         "domain": "resource"},
    "triage_multipivot": {"tier": "multi-pivot",  "domain": "triage"},
}

# ── System prompts ────────────────────────────────────────────────────────────
ACTION_PROMPT = """You are completing a professional decision-making task.
At each step, choose the most appropriate action from the list provided.
Think carefully about what each action actually does — not all options are equal.
Respond ONLY with valid JSON (no markdown fences):
{"action_id": "...", "reasoning": "..."}"""

RETRO_PROMPT = """The episode is now complete. You have seen every step and the final outcome.

Your task: assign a CAUSAL CREDIT score (0.0 to 1.0) to each step.
Ask: "How much did THIS specific step cause the final outcome?"

Rules:
- The step that was the decisive turning point gets credit close to 1.0
- Steps that were routine or had no real effect get credit close to 0.0
- Succeeding at many steps does NOT mean they all deserve credit
- Use step numbers exactly as shown (starting from 1)

Respond ONLY with a JSON object mapping step number to credit.
Example for 5 steps where step 3 was decisive:
{"1": 0.05, "2": 0.05, "3": 0.90, "4": 0.05, "5": 0.05}
No explanation. JSON only."""


def _call_llm_action(obs: dict, model: str) -> dict:
    """Choose an action. Returns {action_id, reasoning}."""
    global _LLM_DISABLED
    if not _openai_client or _LLM_DISABLED:
        return {"action_id": random.choice(obs["available_actions"]), "reasoning": "random"}

    prompt = (
        f"Task domain: {obs['domain']} | Difficulty: {obs['tier']}\n"
        f"Step {obs['step_count'] + 1} of {obs['t_total']}\n\n"
        f"Situation:\n{obs['context']}\n\n"
        f"Available actions:\n" +
        "\n".join(f"  {i+1}. {a}" for i, a in enumerate(obs["available_actions"]))
    )
    try:
        resp = _openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": ACTION_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content)
        action = parsed.get("action_id", "")
        if action not in obs["available_actions"]:
            # LLM may have given a number or partial match — try to resolve
            for a in obs["available_actions"]:
                if action.lower() in a.lower() or a.lower() in action.lower():
                    action = a
                    break
            else:
                action = random.choice(obs["available_actions"])
        return {"action_id": action, "reasoning": parsed.get("reasoning", "")}
    except Exception as exc:
        err = str(exc).lower()
        if any(k in err for k in ["quota", "429", "rate_limit", "ratelimit"]):
            _LLM_DISABLED = True
            print(f"  [warn] LLM rate-limited, switching to random for rest of run")
        return {"action_id": random.choice(obs["available_actions"]), "reasoning": "fallback"}


def _call_llm_credit(steps: list, outcome: str, model: str) -> dict:
    """
    Retrospective credit assignment after episode ends.
    Returns {step_number_str: credit_float} with 1-based keys.
    Falls back to informed heuristic if LLM unavailable.
    """
    global _LLM_DISABLED
    n = len(steps)

    if not _openai_client or _LLM_DISABLED:
        # Heuristic fallback: assign slightly higher credit to later steps
        # (better than uniform — at least introduces some variance)
        credits = {}
        for i, s in enumerate(steps):
            # Slight recency bias as a weak heuristic
            credits[str(i + 1)] = round(0.1 + 0.8 * (i / max(n - 1, 1)), 3)
        return credits

    traj = "\n".join(
        f"  Step {s['step']}: {s['action']}\n    Context: {(s.get('context_snippet') or '')[:120]}"
        for s in steps
    )
    prompt = (
        f"Outcome: {outcome.upper()}\n\n"
        f"Trajectory ({n} steps):\n{traj}\n\n"
        + RETRO_PROMPT
    )
    try:
        resp = _openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = json.loads(resp.choices[0].message.content)
        # Validate and sanitise
        credits = {}
        for k, v in raw.items():
            try:
                credits[str(int(k))] = round(float(max(0.0, min(1.0, v))), 4)
            except (ValueError, TypeError):
                pass
        # Auto-translate: if keys are 0-based (0..n-1) shift to 1-based
        if credits:
            min_k = min(int(k) for k in credits)
            max_k = max(int(k) for k in credits)
            if min_k == 0 and max_k == n - 1:
                credits = {str(int(k) + 1): v for k, v in credits.items()}
        return credits
    except Exception:
        # Heuristic fallback
        return {str(i + 1): round(0.1 + 0.8 * (i / max(n - 1, 1)), 3) for i in range(n)}


def run_episode(http: httpx.Client, tier: str, domain: str, seed: int, model: str) -> dict:
    """
    Run one complete episode with retrospective credit submission.
    Returns per-episode metrics using episode-level (not session) values.
    """
    obs = http.post("/reset", json={"tier": tier, "domain": domain, "seed": seed}).json()
    episode_id = obs["episode_id"]
    steps = []

    # ── Play through episode ──────────────────────────────────────────────────
    for _ in range(obs["max_steps"]):
        step_index = obs["step_count"]           # 0-based, captured before stepping
        context_before = obs.get("context", "")

        decision  = _call_llm_action(obs, model=model)
        action_id = decision["action_id"]

        result = http.post("/step", json={
            "episode_id": episode_id,
            "action_id":  action_id,
            "reasoning":  decision.get("reasoning", ""),
        }).json()

        steps.append({
            "step":            result["observation"]["step_count"],   # 1-based display
            "step_index":      step_index,                            # 0-based
            "action":          action_id,
            "reward":          result["reward"],
            "done":            result["done"],
            "context_snippet": context_before[:150],
        })
        obs = result["observation"]
        if result["done"]:
            break

    outcome = obs.get("episode_outcome", "failure")

    # ── Retrospective credit assignment (primary PSIA signal) ─────────────────
    retro_1based = _call_llm_credit(steps, outcome, model)
    # Convert 1-based keys to 0-based for /credit endpoint
    n_steps = len(steps)
    retro_0based = {}
    if retro_1based:
        max_k = max(int(k) for k in retro_1based)
        for k, v in retro_1based.items():
            ki = int(k)
            # If keys are 1-based (1..n), subtract 1
            if max_k == n_steps and ki >= 1:
                retro_0based[str(ki - 1)] = v
            else:
                retro_0based[str(ki)] = v

    credit_result = http.post("/credit", json={
        "episode_id": episode_id,
        "credits":    retro_0based,
    }).json()

    # ── Grader (composite score) ──────────────────────────────────────────────
    grader = http.post("/grader", json={"episode_id": episode_id}).json()

    # Use per-episode values, NOT session averages
    episode_psia = credit_result.get("psia_score")
    episode_cce  = credit_result.get("cce")

    return {
        "outcome":       outcome,
        "score":         grader.get("score", 0.01),      # composite PSIA-weighted
        "raw_reward":    sum(s["reward"] for s in steps),
        "psia":          episode_psia if episode_psia is not None else 0.0,
        "cce":           episode_cce  if episode_cce  is not None else 0.5,
        "tsr":           1.0 if outcome == "success" else 0.0,
        "mpcs":          credit_result.get("mpcs"),
        "n_steps":       len(steps),
        "credit_source": credit_result.get("credit_source", "unknown"),
    }


def run_task(task_id: str, n: int = N_EPISODES, model: str = MODEL_NAME, seed_offset: int = 0) -> dict:
    """Run N episodes for one task and aggregate."""
    cfg     = TASK_CONFIGS[task_id]
    results = []

    with httpx.Client(base_url=ENV_URL, timeout=180) as http:
        for i in range(n):
            seed = 42 + seed_offset + i
            try:
                r = run_episode(http, tier=cfg["tier"], domain=cfg["domain"], seed=seed, model=model)
                results.append(r)
                print(
                    f"  ep{i+1} (seed={seed}): {r['outcome']:10s}  "
                    f"score={r['score']:.3f}  psia={r['psia']:.3f}  "
                    f"cce={r['cce']:.3f}  steps={r['n_steps']}  "
                    f"[{r['credit_source']}]"
                )
            except Exception as exc:
                print(f"  ep{i+1}: ERROR — {exc}")

    if not results:
        return {"task_id": task_id, "n": 0, "score": 0.01, "tsr": 0.0, "psia": 0.0, "cce": 0.5, "mpcs": None}

    def avg(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    mpcs_vals = [r["mpcs"] for r in results if r.get("mpcs") is not None]

    return {
        "task_id":    task_id,
        "n":          len(results),
        "score":      avg("score"),
        "tsr":        avg("tsr"),
        "psia":       avg("psia"),
        "cce":        avg("cce"),
        "mpcs":       round(sum(mpcs_vals) / len(mpcs_vals), 4) if mpcs_vals else None,
        "avg_reward": avg("raw_reward"),
    }


def main(tasks=None, n=N_EPISODES, model=MODEL_NAME):
    tasks = tasks or list(TASK_CONFIGS.keys())
    all_results = {}

    mode = f"LLM ({model})" if (_openai_client and not _LLM_DISABLED) else "Random heuristic (no API key)"
    print(f"\nCreditMaze Baseline — {mode}")
    print(f"Scoring: 0.4*TSR + 0.4*PSIA + 0.2*(1-CCE)  |  {n} episode(s) per task")
    print("=" * 65)

    for task_id in tasks:
        cfg = TASK_CONFIGS[task_id]
        print(f"\nTask: {task_id}  (tier={cfg['tier']}, domain={cfg['domain']}, n={n})")
        result = run_task(task_id, n=n, model=model)
        all_results[task_id] = result
        mpcs_str = f"{result['mpcs']:.3f}" if result["mpcs"] is not None else "N/A"
        print(
            f"  → score={result['score']:.3f}  "
            f"TSR={result['tsr']:.3f}  PSIA={result['psia']:.3f}  "
            f"CCE={result['cce']:.3f}  MPCS={mpcs_str}"
        )

    print("\n" + "=" * 65)
    print(f"{'Task':<22} {'Score':>7} {'TSR':>6} {'PSIA':>6} {'CCE':>6} {'MPCS':>7}")
    print("-" * 58)
    for task_id, r in all_results.items():
        mpcs_str = f"{r['mpcs']:.3f}" if r["mpcs"] is not None else "  N/A"
        print(f"{task_id:<22} {r['score']:>7.3f} {r['tsr']:>6.3f} {r['psia']:>6.3f} {r['cce']:>6.3f} {mpcs_str:>7}")

    # Final JSON line consumed by /baseline endpoint
    print(json.dumps(all_results))
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CreditMaze Baseline Runner")
    parser.add_argument("--tasks", nargs="+", default=list(TASK_CONFIGS.keys()))
    parser.add_argument("--n",     type=int,   default=N_EPISODES)
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()
    main(tasks=args.tasks, n=args.n, model=args.model)