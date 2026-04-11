"""
Hackathon-compatible inference entrypoint for CreditMaze.

This script emits only [START], [STEP], and [END] log lines so it can be used
by automated evaluators. It uses the OpenAI client for LLM calls and falls back
to a random valid-action policy when no model credentials are available.

Runs all tasks defined in TASK_CONFIG sequentially so the validator sees
3+ [END] lines with the correct task= field.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from typing import Optional

import httpx
from openai import OpenAI

ENV_URL       = os.getenv("ENV_URL", "http://127.0.0.1:7860")
API_BASE_URL  = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME    = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
# When set, run only this task; otherwise run all tasks
TASK_NAME     = os.getenv("CREDITMAZE_TASK", "")
BENCHMARK     = os.getenv("CREDITMAZE_BENCHMARK", "creditmaze")
HF_TOKEN      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "")
SEED          = int(os.getenv("CREDITMAZE_SEED", "42"))
TIMEOUT       = float(os.getenv("CREDITMAZE_TIMEOUT", "60"))

# Maps openenv.yaml task IDs → environment reset params
# IDs MUST exactly match the id: fields in openenv.yaml
TASK_CONFIG = {
    "task_easy":        {"tier": "easy",         "domain": "corridor"},
    "task_medium":      {"tier": "medium",        "domain": "research"},
    "task_hard":        {"tier": "hard",          "domain": "debugging"},
    "resource_hard":    {"tier": "hard",          "domain": "resource"},
    "triage_multipivot":{"tier": "multi-pivot",   "domain": "triage"},
}

SYSTEM_PROMPT = (
    "You are evaluating a long-horizon decision environment. "
    "Choose exactly one valid action_id from the provided list and estimate how "
    "important the current step is to final success. "
    "Respond with JSON only: "
    '{"action_id":"...", "reasoning":"...", "credit_estimate":0.0}'
)


# ── Structured log emitters ───────────────────────────────────────────────────
# Format is STRICT — field names and order must match exactly.

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(task: str, success: bool, steps: int, score: float, rewards: list) -> None:
    """
    CRITICAL: task= field MUST be present and match openenv.yaml task id.
    Validator uses task= to link output to grader definition.
    """
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] task={task} success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Server management ─────────────────────────────────────────────────────────

def maybe_start_local_server() -> Optional[subprocess.Popen]:
    if "127.0.0.1" not in ENV_URL and "localhost" not in ENV_URL:
        return None
    try:
        with httpx.Client(timeout=2.0) as http:
            if http.get(f"{ENV_URL}/health").status_code == 200:
                return None
    except Exception:
        pass
    return subprocess.Popen(
        [sys.executable, "server.py"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def wait_for_server() -> None:
    deadline = time.time() + TIMEOUT
    while time.time() < deadline:
        try:
            with httpx.Client(timeout=2.0) as http:
                if http.get(f"{ENV_URL}/health").status_code == 200:
                    return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("environment_unreachable")


# ── Action selection ──────────────────────────────────────────────────────────

def choose_action(client: Optional[OpenAI], obs: dict, task_name: str) -> tuple:
    if client is None:
        return ({
            "action_id": random.choice(obs["available_actions"]),
            "reasoning": "Random fallback",
            "credit_estimate": 0.5,
        }, "random_fallback:no_credentials")

    prompt = (
        f"Task: {task_name}\n"
        f"Domain: {obs['domain']}\n"
        f"Tier: {obs['tier']}\n"
        f"Step: {obs['step_count'] + 1}/{obs['t_total']}\n"
        f"Context:\n{obs['context']}\n\n"
        f"Available actions: {obs['available_actions']}"
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0,
            stream=False,
        )
        content = (completion.choices[0].message.content or "").strip()
        parsed  = json.loads(content)
        credit  = float(parsed.get("credit_estimate", 0.5))
        parsed["credit_estimate"] = min(max(credit, 0.0), 1.0)
        return parsed, None
    except Exception as exc:
        return ({
            "action_id": random.choice(obs["available_actions"]),
            "reasoning": "Random fallback",
            "credit_estimate": 0.5,
        }, f"random_fallback:model_call_failed:{type(exc).__name__}")


# ── Single task runner ────────────────────────────────────────────────────────

def run_task(
    http: httpx.Client,
    client: Optional[OpenAI],
    task_id: str,
    model_label: str,
) -> None:
    """Run one complete episode for task_id and emit [START]→[STEP]→[END]."""
    task_cfg = TASK_CONFIG.get(task_id, TASK_CONFIG["task_medium"])
    rewards: list = []
    steps_taken = 0
    success = False
    score   = 0.01

    log_start(task=task_id, env=BENCHMARK, model=model_label)

    try:
        obs      = http.post("/reset", json={**task_cfg, "seed": SEED}).json()
        max_steps = int(obs["max_steps"])

        for step in range(1, max_steps + 1):
            decision, step_error = choose_action(client, obs, task_id)
            action_id = decision.get("action_id", obs["available_actions"][0])
            if action_id not in obs["available_actions"]:
                action_id = obs["available_actions"][0]

            result  = http.post("/step", json={
                "episode_id":     obs["episode_id"],
                "action_id":      action_id,
                "reasoning":      decision.get("reasoning", ""),
                "credit_estimate": float(decision.get("credit_estimate", 0.5)),
            }).json()

            reward = float(result.get("reward", 0.0))
            done   = bool(result.get("done", False))
            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_id, reward=reward, done=done, error=step_error)

            obs = result["observation"]
            if done:
                break

        grader  = http.post("/grader", json={"episode_id": obs["episode_id"]}).json()
        score   = float(grader.get("score", 0.01))
        score   = min(max(score, 0.01), 0.99)
        success = str(grader.get("outcome", "failure")) == "success"

    except Exception as exc:
        score = min(max(score, 0.01), 0.99)

    log_end(
        task=task_id,
        success=success,
        steps=steps_taken,
        score=score,
        rewards=rewards,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _ = LOCAL_IMAGE_NAME
    server_proc  = maybe_start_local_server()
    model_label  = MODEL_NAME if HF_TOKEN else "random-fallback"
    client       = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN, max_retries=0) if HF_TOKEN else None

    # Decide which tasks to run:
    # - If CREDITMAZE_TASK is set → run only that one task (single-task mode)
    # - Otherwise → run ALL tasks so validator sees 3+ [END] lines
    if TASK_NAME and TASK_NAME in TASK_CONFIG:
        tasks_to_run = [TASK_NAME]
    else:
        tasks_to_run = list(TASK_CONFIG.keys())

    try:
        wait_for_server()
        with httpx.Client(base_url=ENV_URL, timeout=TIMEOUT) as http:
            for task_id in tasks_to_run:
                run_task(http, client, task_id, model_label)
    finally:
        if server_proc is not None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()


if __name__ == "__main__":
    main()
