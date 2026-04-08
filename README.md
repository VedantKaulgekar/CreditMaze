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

**An OpenEnv-compatible RL benchmark purpose-built to isolate and measure credit assignment quality in long-horizon LLM agent tasks.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://openenv.ai)

---

## What Is CreditMaze?

Every episode contains exactly one (or more, in Tier 4) decision that **causally determines** the final outcome. All other decisions are carefully crafted **decoys** - actions that look important and generate small positive reward, but are causally irrelevant.

This lets CreditMaze **directly measure credit assignment quality**: did the agent's training algorithm correctly identify which step actually mattered?

## Why This Matters

In long-horizon agent tasks, success often depends on one critical earlier decision, while many other steps look important but are actually distractions. Standard RL benchmarks usually measure only whether the agent eventually succeeds, not whether it assigned credit to the step that truly caused the outcome.

CreditMaze is designed to evaluate that missing ability. Each episode contains hidden pivotal steps, realistic decoy actions, and ground-truth causal labels, so researchers can measure not just task success, but whether an agent correctly identified the actions that actually mattered.

### Why Now

Recent agentic RL work, including ICLR 2026 discussion around long-horizon LLM training, has highlighted credit assignment as a central bottleneck: rewards are sparse and delayed, but trajectories contain many intermediate decisions and long reasoning traces. In that setting, it becomes hard to tell which earlier step truly caused the final result.

This problem is especially important for methods such as GRPO and related value-free approaches, where a final trajectory-level reward can wash out the contribution of the specific step that actually mattered.

### Why LLM Agents Make It Harder

In classical RL, a step is often a small atomic action. In LLM agents, a single step can include a long chain of reasoning plus an external action. That makes long-horizon trajectories much more expensive to analyze and makes step-level credit assignment much noisier.

As a result, two trajectories may end with the same outcome while hiding very different causal stories. A benchmark that only measures final success cannot distinguish luck from correct credit assignment.

### The Gap CreditMaze Fills

Recent credit-assignment methods are often evaluated on different tasks and domains, which makes side-by-side comparison difficult. Existing agent benchmarks like WebShop, ALFWorld, SWE-bench, and TheAgentCompany mainly measure whether the final goal was achieved, not whether the agent identified the causally decisive intermediate step.

CreditMaze fills that gap by providing:

1. Ground-truth causal credit labels for every episode, verified by counterfactual simulation.
2. PSIA, CCE, and MPCS, which measure credit assignment quality directly rather than only final success.
3. A shared, reproducible testbed where different 2025-2026 credit-assignment methods can be compared under the same conditions.

### The Three Novel Metrics

| Metric   | Definition                                                                                    |
| -------- | --------------------------------------------------------------------------------------------- |
| **PSIA** | Pivotal Step Identification Accuracy - did the agent assign highest credit to the true pivot? |
| **CCE**  | Credit Calibration Error - MSE between agent credit estimates and ground-truth labels         |
| **MPCS** | Multi-Pivot Coordination Score - fraction of jointly-pivotal steps found (Tier 4)             |

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
| `task_easy`         | easy        | corridor  | Calibration task: a minimal branching-decision environment with one causally decisive junction. |
| `task_medium`       | medium      | research  | Resolve contradicting research sources to produce correct qualified synthesis.                  |
| `task_hard`         | hard        | debugging | Fix bugs in the correct dependency order - wrong order creates irresolvable cycle.              |
| `resource_hard`     | hard        | resource  | Allocate a time-sensitive resource before an irreversible commitment window closes.             |
| `triage_multipivot` | multi-pivot | triage    | Identify multiple jointly-causal signals from high-correlation noise.                           |

Internally, the canonical tasks map to the environment domains `corridor`, `research`, and `debugging`.

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
| Decoy step (any valid action)         | 0.04                         |
| Correct pivot action (not final)      | 0.12                         |
| Incorrect pivot action                | 0.0 (episode fails)          |
| Episode success (final pivot correct) | 0.5 + 0.5 × efficiency_bonus |
| Episode failure                       | 0.0                          |

Invalid actions also terminate the episode with `0.0` reward.

**Design intent:** Step rewards are deliberately **non-revealing** - the pivotal step's reward (0.12) is indistinguishable from some decoy steps. This is what makes credit assignment hard.

---

The `/grader` endpoint separately returns a normalized `score` in `[0, 1]` plus `raw_reward` for analysis.

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
| `/reset`    | POST   | Start new episode                        |
| `/step`     | POST   | Submit action                            |
| `/state`    | GET    | Get episode state (labels after done)    |
| `/tasks`    | GET    | List evaluator-facing task catalog       |
| `/grader`   | POST   | Get grader metrics for completed episode |
| `/baseline` | POST   | Run baseline inference script            |
| `/health`   | GET    | Health check                             |

## Submission Inference

For hackathon submission, use the root-level `inference.py` script. It uses the OpenAI client, reads `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` / `OPENAI_API_KEY`, emits strict `[START]`, `[STEP]`, and `[END]` lines, and can auto-start the local environment server when `ENV_URL` points at `localhost`.

By default, `inference.py` runs the canonical evaluator tasks:
- `task_easy`
- `task_medium`
- `task_hard`

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
  -d '{"episode_id": "abc12345", "action_id": "continue_forward", "credit_estimate": 0.3}'

# State (after done=True - reveals causal labels)
curl "http://localhost:7860/state?episode_id=abc12345"
```

---

## Baseline Scores

The default reproducible command is `python baseline.py`, which now runs `easy`, `medium`, `hard`, and `multi-pivot` and prints a final JSON summary. With `OPENAI_API_KEY` set, it uses the OpenAI client against the configured model. Without a key, it falls back to a deterministic random policy for smoke testing.

The table below reports the current reproducible random-fallback baseline. Evaluators can regenerate model-backed scores by setting `OPENAI_API_KEY` and rerunning `baseline.py` or `inference.py`.

Current reproducible local baseline, measured with deterministic random fallback (no API key), 5 episodes per tier:

| Tier        | TSR   | PSIA  | CCE   | MPCS  |
| ----------- | ----- | ----- | ----- | ----- |
| Easy        | 0.613 | 0.000 | 0.316 | -     |
| Medium      | 0.475 | 0.000 | 0.339 | -     |
| Hard        | 0.389 | 0.000 | 0.346 | -     |
| Multi-pivot | 0.280 | 0.049 | 0.338 | 0.327 |

_Run `python baseline.py` for the current default aggregate benchmark. Set `OPENAI_API_KEY` to use an LLM and regenerate model-backed scores before publishing them._
_Without API key: deterministic random-policy fallback runs automatically._

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

## Research Contribution

CreditMaze introduces:

1. **PSIA and CCE** - first metrics that directly measure credit assignment quality (not just task success)
2. **Ground-truth causal labels** - verified via counterfactual simulation for every episode
3. **Multi-pivot Tier 4** - first benchmark for jointly-causal credit assignment (MPCS metric)
4. **Algorithm-agnostic credit extraction hook** - plug in GRPO, PPO, iStar, HCAPO, or any future method

---

## Citation

```
@misc{creditmaze2026,
  title   = {CreditMaze: An RL Environment for Measuring Credit Assignment Quality},
  year    = {2026},
  note    = {Meta PyTorch OpenEnv Hackathon submission}
}
```
