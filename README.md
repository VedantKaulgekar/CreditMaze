---
title: CreditMaze
emoji: "🧭"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# CreditMaze

**CreditMaze is an OpenEnv-compatible benchmark for evaluating whether long-horizon AI agents can identify the evidence or decision that actually caused success or failure.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://openenv.ai)

---

## What Is CreditMaze?

CreditMaze is a long-horizon agent benchmark where each episode contains one decisive step, or multiple jointly decisive steps in the multi-pivot tier, hidden among realistic decoys.

It measures two things at once:

1. Did the agent solve the task?
2. Did it give the most credit to the step that actually mattered?

## Why This Matters

In real multi-step AI workflows, many actions look useful even though only a small number actually determine the outcome. Standard benchmarks usually check only whether the agent finished the task, not whether it understood which earlier decision or evidence fragment caused success or failure.

CreditMaze is designed to evaluate that missing ability. Each episode contains hidden pivotal steps, realistic decoy actions, and ground-truth causal labels, so evaluators can measure not just task success, but whether the agent correctly identified the actions that actually mattered.

### A Simple Example

Imagine an agent reading many sources before writing a research answer. A normal benchmark checks whether the final answer is right. CreditMaze also checks whether the agent identified the one key source or decision that truly changed the result instead of giving equal importance to every step.

### Why Now

Recent agentic RL work, including ICLR 2026 discussion around long-horizon LLM training, has highlighted credit assignment as a central bottleneck: rewards are sparse and delayed, but trajectories contain many intermediate decisions and long reasoning traces. In that setting, it becomes hard to tell which earlier step truly caused the final result.

This matters beyond RL too. In debugging, incident response, triage, or research synthesis, we want agents that can not only reach an answer, but also explain which step or evidence fragment actually drove it.

### Why LLM Agents Make It Harder

In classical RL, a step is often a small atomic action. In LLM agents, a single step can include a long chain of reasoning plus an external action. That makes long-horizon trajectories much more expensive to analyze and makes step-level credit assignment much noisier.

As a result, two trajectories may end with the same outcome while hiding very different causal stories. A benchmark that only measures final success cannot distinguish luck from correct credit assignment.

### The Gap CreditMaze Fills

Existing agent benchmarks like WebShop, ALFWorld, SWE-bench, and TheAgentCompany mainly measure whether the final goal was achieved, not whether the agent identified the causally decisive intermediate step.

CreditMaze fills that gap by providing:

1. Ground-truth causal labels for each episode, verified by counterfactual simulation.
2. PSIA, CCE, and MPCS, which measure attribution quality directly rather than only final success.
3. A shared, reproducible testbed where task success and causal attribution quality can be compared under the same conditions.

The current benchmark content is intentionally varied across practical workflows rather than relying on a single fixed puzzle per tier. Research, debugging, resource-allocation, and triage tasks now include multiple scenario templates with stronger decoy evidence, so the pivotal step is less likely to announce itself through obviously filler context.

### Attribution Metrics

| Metric   | Definition                                                                                    |
| -------- | --------------------------------------------------------------------------------------------- |
| **PSIA** | Pivotal Step Identification Accuracy - did the agent assign highest credit to the true pivot? |
| **CCE**  | Credit Calibration Error - MSE between agent credit estimates and ground-truth labels         |
| **MPCS** | Multi-Pivot Coordination Score - fraction of jointly-pivotal steps found (Tier 4)             |
| **TSR**  | Task Success Rate - did the underlying task succeed?                                           |

---

## Environment Description

### Action Space

```json
{
  "action_id": "string  (required) - must be in observation.available_actions",
  "reasoning": "string  (optional) - agent chain-of-thought",
  "credit_estimate": "float   (optional) - agent-reported step importance [0.0, 1.0]"
}
```

### Observation Space

```json
{
  "episode_id": "string",
  "domain": "corridor | research | debugging | resource | triage",
  "tier": "easy | medium | hard | multi-pivot",
  "t_total": "int - total designed steps",
  "step_count": "int - current step index",
  "max_steps": "int - hard termination cap",
  "context": "string - full narrative context for this step",
  "available_actions": ["list of valid action_ids"],
  "last_step_reward": "float | null",
  "cumulative_reward": "float | null",
  "episode_outcome": "in_progress | success | failure | null"
}
```

**Note:** `pivotal_step_indices`, `causal_chain`, and `counterfactual_outcomes` are **never** in the Observation. They are ground-truth labels revealed only via `GET /state` after `done=True`.

### Task Descriptions

For submission and evaluation, CreditMaze exposes three canonical benchmark tasks plus two additional hard-task variants.

| Task ID             | Tier        | Domain    | Description                                                                                     |
| ------------------- | ----------- | --------- | ----------------------------------------------------------------------------------------------- |
| `task_easy`         | easy        | corridor  | Calibration task: identify the decisive route in a branching environment with plausible distractions. |
| `task_medium`       | medium      | research  | Reconcile contradictory evidence without over-generalizing from persuasive but incomplete sources. |
| `task_hard`         | hard        | debugging | Fix the root-cause bug in the correct dependency order while visible symptom fixes tempt the agent off path. |
| `resource_hard`     | hard        | resource  | Make the one irreversible resource commitment that preserves success before the window closes. |
| `triage_multipivot` | multi-pivot | triage    | Identify multiple jointly-causal signals in a noisy investigation full of strong correlates. |

Internally, the canonical tasks map to the environment domains `corridor`, `research`, and `debugging`.

The underlying scenario templates now cover more practical settings such as billing systems, CRM sync, fraud analysis, support retrieval, hospital surge planning, cloud outages, and subscription churn investigations.

### Difficulty Tiers

| Tier        | Steps | Pivots | Pivot Position | Decoy Similarity | Expected PSIA (GRPO) |
| ----------- | ----- | ------ | -------------- | ---------------- | -------------------- |
| Easy        | 10    | 1      | steps 5-9      | Low (0.2)        | ~0.65                |
| Medium      | 14    | 1      | steps 1-5      | Medium (0.5)     | ~0.42                |
| Hard        | 12    | 1      | steps 1-3      | High (0.8)       | ~0.20                |
| Multi-pivot | 12    | 2      | distributed    | Very high (0.9)  | ~0.25                |

### Reward Function

| Situation                             | Reward                       |
| ------------------------------------- | ---------------------------- |
| Early decoy step                      | 0.04                         |
| Mid-episode decoy step                | 0.06 to 0.08                 |
| Late decoy step                       | up to 0.10                   |
| Correct pivot action (not final)      | 0.12                         |
| Incorrect pivot action                | 0.0 (episode fails)          |
| Episode success (final pivot correct) | 0.5 + 0.5 x efficiency_bonus |
| Episode failure                       | 0.0                          |

Invalid actions also terminate the episode with `0.0` reward.

**Design intent:** Step rewards are deliberately **non-revealing**. Some decoys become locally attractive later in the episode, so a reward-greedy policy can still focus on the wrong step.

---

The `/grader` endpoint separately returns a normalized `score` in the open interval `0.01` to `0.99` plus `raw_reward`, session metrics, and attribution diagnostics for analysis.

## Setup

### Local

```bash
git clone https://github.com/VedantKaulgekar/CreditMaze.git
cd creditmaze
pip install -r requirements.txt
python server.py
```

### Docker

```bash
docker build -t creditmaze:latest .
docker run -p 7860:7860 creditmaze:latest
```

---

## API Reference

| Endpoint    | Method | Description                              |
| ----------- | ------ | ---------------------------------------- |
| `/reset`    | POST   | Start a new episode                      |
| `/step`     | POST   | Submit one action                        |
| `/state`    | GET    | Inspect episode state; causal labels appear only after completion |
| `/tasks`    | GET    | List evaluator-facing task catalog       |
| `/grader`   | POST   | Return final score, session metrics, and attribution diagnostics for a completed episode |
| `/baseline` | POST   | Run the aggregate baseline script and return its JSON summary |
| `/health`   | GET    | Health check                             |

### `/grader` Output

The grader returns:

- `score` in the open interval `0.01` to `0.99`
- `raw_reward`
- session-level `PSIA`, `CCE`, `TSR`, and `MPCS`
- `causal_faithfulness`
- `pivotal_step_indices`
- attribution diagnostics such as:
  - `top_attributed_step`
  - `top_attributed_action`
  - `pivotal_step_rank`
  - `false_positive_steps`
  - `attribution_gap`
  - `success_with_wrong_attribution`

## Submission Inference

For evaluator-compatible runs, use the root-level `inference.py` script. It uses the OpenAI client, reads `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` / `OPENAI_API_KEY` / `API_KEY`, emits strict `[START]`, `[STEP]`, and `[END]` lines, and can auto-start the local environment server when `ENV_URL` points at `localhost`.

By default, `inference.py` runs every evaluator-facing task so the validator can observe multiple complete episodes:
- `task_easy`
- `task_medium`
- `task_hard`
- `resource_hard`
- `triage_multipivot`

If `CREDITMAZE_TASK` is set, it runs only that one task.

```bash
python inference.py
```

To evaluate a specific task:

```bash
CREDITMAZE_TASK=task_hard python inference.py
```

### Example Session

```bash
# Reset
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"tier": "easy", "seed": 42}'

# Step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"episode_id": "abc12345", "action_id": "<choose one from available_actions>", "credit_estimate": 0.3}'

# State (after done=True - reveals causal labels)
curl "http://localhost:7860/state?episode_id=abc12345"
```

---

## Baselines

The default aggregate runner is `python baseline.py`. It runs the benchmark tiers, prints a human-readable summary, and emits one final JSON line that is also used by the `/baseline` endpoint.

Behavior:

- with `OPENAI_API_KEY`, `HF_TOKEN`, or `API_KEY` set, it uses the configured model
- without a key, it automatically falls back to a random valid-action baseline

Recommended commands:

```bash
python baseline.py
python baseline.py --n 1
```

For honest reporting, regenerate model-backed numbers before publishing them and avoid mixing LLM-backed results with fallback runs after quota or rate-limit failures.

---

## Running Tests

```bash
pytest tests/ -v
```

## Pre-Submission Check

```bash
python sanity_check.py
```

---

## Credit Extraction

CreditMaze ships with extractors for major 2025-2026 RL algorithms:

```python
from credit_extraction import GRPOExtractor, PPOExtractor, IStarExtractor

# GRPO: needs multiple rollouts
extractor = GRPOExtractor(n_rollouts=8)
extractor.add_rollout(episode_id, trajectory, total_reward)
credits = extractor.extract(trajectory, episode_id)

# PPO: needs critic function
extractor = PPOExtractor(critic_fn=my_critic)
credits = extractor.extract(trajectory, episode_id)

# iStar: needs PRM scorer
extractor = IStarExtractor(scorer_fn=my_prm)
credits = extractor.extract(trajectory, episode_id)
```

---

## Why Evaluators Might Use It

CreditMaze is useful when you want to know not only whether an agent finished a long task, but whether it understood **what actually mattered** inside that task.

Examples:

- a debugging agent that eventually passes tests but over-credits symptom fixes instead of the root cause
- an incident-response agent that resolves the issue but credits the wrong signal
- a research agent that gives the right answer but cannot identify which evidence changed the conclusion

That makes the benchmark relevant for:

- RL research
- LLM agent evaluation
- workflow auditability
- root-cause analysis
- decision-trace debugging

---

## Research Contribution

CreditMaze introduces:

1. **Custom attribution metrics** such as PSIA, CCE, and MPCS for evaluating whether an agent credited the right step, not just whether it succeeded
2. **Ground-truth causal labels** verified by counterfactual simulation for every episode
3. **Jointly-causal multi-pivot tasks** where success depends on identifying more than one decisive step
4. **Algorithm-agnostic credit extraction hooks** so different training or attribution methods can be compared on the same environment

---

## Citation

```bibtex
@misc{creditmaze2026,
  title   = {CreditMaze: A Benchmark for Causal Decision Attribution in Long-Horizon Agent Workflows},
  year    = {2026},
  note    = {Meta PyTorch OpenEnv Hackathon submission}
}
```
