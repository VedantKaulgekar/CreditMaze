"""
Hackathon-compatible inference entrypoint for CreditMaze.

This script emits only [START], [STEP], and [END] log lines so it can be used
by automated evaluators. It uses the OpenAI client for LLM calls and falls back
to a deterministic random policy when no model credentials are available.
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

ENV_URL = os.getenv("ENV_URL", "http://127.0.0.1:7860")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("CREDITMAZE_TASK", "research_medium")
BENCHMARK = os.getenv("CREDITMAZE_BENCHMARK", "creditmaze")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "")
SEED = int(os.getenv("CREDITMAZE_SEED", "42"))
TIMEOUT = float(os.getenv("CREDITMAZE_TIMEOUT", "30"))

TASK_CONFIG = {
    "corridor_easy": {"tier": "easy", "domain": "corridor"},
    "research_medium": {"tier": "medium", "domain": "research"},
    "debugging_hard": {"tier": "hard", "domain": "debugging"},
    "resource_hard": {"tier": "hard", "domain": "resource"},
    "triage_multipivot": {"tier": "multi-pivot", "domain": "triage"},
}

SYSTEM_PROMPT = (
    "You are evaluating a long-horizon decision environment. "
    "Choose exactly one valid action_id from the provided list and estimate how "
    "important the current step is to final success. "
    "Respond with JSON only: "
    '{"action_id":"...", "reasoning":"...", "credit_estimate":0.0}'
)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def maybe_start_local_server() -> Optional[subprocess.Popen]:
    if "127.0.0.1" not in ENV_URL and "localhost" not in ENV_URL:
        return None

    try:
        with httpx.Client(timeout=2.0) as http:
            response = http.get(f"{ENV_URL}/health")
            if response.status_code == 200:
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
                response = http.get(f"{ENV_URL}/health")
                if response.status_code == 200:
                    return
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("environment_unreachable")


def choose_action(client: Optional[OpenAI], obs: dict) -> dict:
    if client is None:
        return {
            "action_id": random.choice(obs["available_actions"]),
            "reasoning": "Random fallback",
            "credit_estimate": 0.5,
        }

    prompt = (
        f"Task: {TASK_NAME}\n"
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
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            stream=False,
        )
        content = (completion.choices[0].message.content or "").strip()
        parsed = json.loads(content)
        credit = float(parsed.get("credit_estimate", 0.5))
        parsed["credit_estimate"] = min(max(credit, 0.0), 1.0)
        return parsed
    except Exception:
        return {
            "action_id": random.choice(obs["available_actions"]),
            "reasoning": "Random fallback",
            "credit_estimate": 0.5,
        }


def main() -> None:
    _ = LOCAL_IMAGE_NAME
    server_proc = maybe_start_local_server()
    rewards: list[float] = []
    steps_taken = 0
    success = False
    score = 0.0
    task_cfg = TASK_CONFIG.get(TASK_NAME, TASK_CONFIG["research_medium"])
    model_label = MODEL_NAME if HF_TOKEN else "random-fallback"
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN, max_retries=0) if HF_TOKEN else None

    log_start(task=TASK_NAME, env=BENCHMARK, model=model_label)

    try:
        wait_for_server()
        with httpx.Client(base_url=ENV_URL, timeout=TIMEOUT) as http:
            obs = http.post("/reset", json={**task_cfg, "seed": SEED}).json()
            max_steps = int(obs["max_steps"])

            for step in range(1, max_steps + 1):
                decision = choose_action(client, obs)
                action_id = decision.get("action_id", obs["available_actions"][0])
                if action_id not in obs["available_actions"]:
                    action_id = obs["available_actions"][0]

                result = http.post(
                    "/step",
                    json={
                        "episode_id": obs["episode_id"],
                        "action_id": action_id,
                        "reasoning": decision.get("reasoning", ""),
                        "credit_estimate": float(decision.get("credit_estimate", 0.5)),
                    },
                ).json()

                reward = float(result.get("reward", 0.0))
                done = bool(result.get("done", False))
                rewards.append(reward)
                steps_taken = step
                log_step(step=step, action=action_id, reward=reward, done=done, error=None)

                obs = result["observation"]
                if done:
                    break

            grader = http.post("/grader", json={"episode_id": obs["episode_id"]}).json()
            score = float(grader.get("score", 0.0))
            success = str(grader.get("outcome", "failure")) == "success"

    except Exception:
        success = False
    finally:
        if server_proc is not None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
