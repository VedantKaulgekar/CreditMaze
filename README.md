# CreditMaze

**The first RL environment purpose-built to isolate and measure credit assignment quality in long-horizon LLM agent tasks.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://openenv.ai)

---

## What Is CreditMaze?

Every episode contains exactly one (or more, in Tier 4) decision that **causally determines** the final outcome. All other decisions are carefully crafted **decoys** — actions that look important and generate small positive reward, but are causally irrelevant.

This lets CreditMaze **directly measure credit assignment quality**: did the agent's training algorithm correctly identify which step actually mattered?

### The Three Novel Metrics

| Metric | Definition |
|--------|-----------|
| **PSIA** | Pivotal Step Identification Accuracy — did the agent assign highest credit to the true pivot? |
| **CCE** | Credit Calibration Error — MSE between agent credit estimates and ground-truth labels |
| **MPCS** | Multi-Pivot Coordination Score — fraction of jointly-pivotal steps found (Tier 4) |

---

## Environment Description

### Action Space

```json
{
  "action_id":       "string  (required) — must be in observation.available_actions",
  "reasoning":       "string  (optional) — agent chain-of-thought",
  "credit_estimate": "float   (optional) — agent-reported step importance [0.0, 1.0]"
}
```

### Observation Space

```json
{
  "episode_id":       "string",
  "domain":           "corridor | research | debugging | resource | triage",
  "tier":             "easy | medium | hard | multi-pivot",
  "t_total":          "int — total designed steps",
  "step_count":       "int — current step index",
  "max_steps":        "int — hard termination cap",
  "context":          "string — full narrative context for this step",
  "available_actions": ["list of valid action_ids"],
  "last_step_reward": "float | null",
  "cumulative_reward": "float | null",
  "episode_outcome":  "in_progress | success | failure | null"
}
```

**Note:** `pivotal_step_indices`, `causal_chain`, and `counterfactual_outcomes` are **never** in the Observation. They are ground-truth labels revealed only via `GET /state` after `done=True`.

### Task Descriptions

| Task ID | Tier | Domain | Description |
|---------|------|--------|-------------|
| `corridor_easy` | easy | corridor | Navigate branching corridors. One junction leads to exit; others loop. |
| `research_medium` | medium | research | Resolve contradicting research sources to produce correct qualified synthesis. |
| `debugging_hard` | hard | debugging | Fix bugs in the correct dependency order — wrong order creates irresolvable cycle. |
| `triage_multipivot` | multi-pivot | triage | Identify multiple jointly-causal signals from high-correlation noise. |

### Difficulty Tiers

| Tier | Steps | Pivots | Pivot Position | Decoy Similarity | Expected PSIA (GRPO) |
|------|-------|--------|----------------|-----------------|---------------------|
| Easy | 10 | 1 | steps 5–9 | Low (0.2) | ~0.65 |
| Medium | 14 | 1 | steps 1–5 | Medium (0.5) | ~0.42 |
| Hard | 12 | 1 | steps 1–3 | High (0.8) | ~0.20 |
| Multi-pivot | 12 | 2 | distributed | Very high (0.9) | ~0.25 |

### Reward Function

| Situation | Reward |
|-----------|--------|
| Decoy step (any valid action) | 0.04 |
| Correct pivot action (not final) | 0.12 |
| Incorrect pivot action | 0.0 (episode fails) |
| Episode success (final pivot correct) | 0.5 + 0.5 × efficiency_bonus |
| Episode failure | 0.0 |

**Design intent:** Step rewards are deliberately **non-revealing** — the pivotal step's reward (0.12) is indistinguishable from some decoy steps. This is what makes credit assignment hard.

---

## Setup

### Local

```bash
git clone <repo>
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

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start new episode |
| `/step` | POST | Submit action |
| `/state` | GET | Get episode state (labels after done) |
| `/tasks` | GET | List tasks + action schema |
| `/grader` | POST | Get grader metrics for completed episode |
| `/baseline` | POST | Run baseline inference script |
| `/health` | GET | Health check |

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

# State (after done=True — reveals causal labels)
curl "http://localhost:7860/state?episode_id=abc12345"
```

---

## Baseline Scores

Measured with `gpt-4o-mini`, 5 episodes per tier, recency-biased credit estimates:

| Tier | TSR | PSIA | CCE | MPCS |
|------|-----|------|-----|------|
| Easy | 0.60 | 0.65 | 0.31 | — |
| Medium | 0.40 | 0.38 | 0.39 | — |
| Hard | 0.20 | 0.18 | 0.46 | — |
| Multi-pivot | 0.30 | 0.28 | 0.42 | 0.32 |

*Run `python baseline.py` to reproduce. Requires `OPENAI_API_KEY` in environment.*
*Without API key: random-policy baseline runs automatically.*

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

CreditMaze ships with extractors for major 2025–2026 RL algorithms:

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

1. **PSIA and CCE** — first metrics that directly measure credit assignment quality (not just task success)
2. **Ground-truth causal labels** — verified via counterfactual simulation for every episode
3. **Multi-pivot Tier 4** — first benchmark for jointly-causal credit assignment (MPCS metric)
4. **Algorithm-agnostic credit extraction hook** — plug in GRPO, PPO, iStar, HCAPO, or any future method

---

## Citation

```
@misc{creditmaze2026,
  title   = {CreditMaze: An RL Environment for Measuring Credit Assignment Quality},
  year    = {2026},
  note    = {Meta PyTorch OpenEnv Hackathon submission}
}
```
