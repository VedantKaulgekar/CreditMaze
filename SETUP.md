# CreditMaze — Windows Setup Guide

## Step 1 — Install dependencies

```powershell
cd creditmaze
pip install -r requirements.txt
```

This will upgrade your existing packages to be compatible. The `openenv-core`
conflicts in the install log are from OTHER tools in your Anaconda environment —
they do not affect CreditMaze itself.

---

## Step 2 — Run the server

```powershell
python server.py
```

Expected output:
```
INFO:     Started server process [...]
INFO:     Uvicorn running on http://0.0.0.0:7860
```

Leave this terminal open.

---

## Step 3 — Smoke test (new terminal)

```powershell
# Health check
curl http://localhost:7860/health

# Reset an episode
curl -X POST http://localhost:7860/reset `
  -H "Content-Type: application/json" `
  -d '{"tier": "easy", "seed": 42}'

# List tasks
curl http://localhost:7860/tasks
```

---

## Step 4 — Run tests

```powershell
pytest tests/ -v
```

Expected: all tests pass.

---

## Step 5 — Run sanity check

```powershell
python sanity_check.py
```

Expected: `All checks PASSED — safe to submit!`

---

## Step 6 — Run baseline (optional — needs OpenAI API key)

```powershell
# With API key
$env:OPENAI_API_KEY = "sk-..."
python baseline.py --n 2

# Without API key (random policy baseline — still produces valid scores)
python baseline.py --n 2
```

---

## Step 7 — Docker build

```powershell
docker build -t creditmaze:latest .
docker run -p 7860:7860 creditmaze:latest
```

---

## Complete API Reference

| Endpoint | Method | Body | Description |
|----------|--------|------|-------------|
| `/health` | GET | — | Health check |
| `/reset` | POST | `{"tier": "easy", "seed": 42}` | Start new episode |
| `/step` | POST | `{"episode_id": "...", "action_id": "...", "credit_estimate": 0.5}` | Submit action |
| `/state` | GET | `?episode_id=...` | Get episode state |
| `/tasks` | GET | — | List tasks + schema |
| `/grader` | POST | `{"episode_id": "..."}` | Get grader metrics |
| `/baseline` | POST | — | Run baseline script |

### Full episode walkthrough

```powershell
# 1. Reset
$reset = curl -X POST http://localhost:7860/reset `
  -H "Content-Type: application/json" `
  -d '{"tier": "easy", "domain": "corridor", "seed": 42}' | ConvertFrom-Json

$ep_id = $reset.episode_id
Write-Host "Episode: $ep_id"
Write-Host "Actions: $($reset.available_actions)"

# 2. Step (repeat until done=true)
$step = curl -X POST http://localhost:7860/step `
  -H "Content-Type: application/json" `
  -d "{`"episode_id`": `"$ep_id`", `"action_id`": `"$($reset.available_actions[0])`", `"credit_estimate`": 0.5}" | ConvertFrom-Json

Write-Host "Reward: $($step.reward), Done: $($step.done)"

# 3. After done=true — get causal labels
$state = curl "http://localhost:7860/state?episode_id=$ep_id" | ConvertFrom-Json
Write-Host "Pivotal steps: $($state.pivotal_step_indices)"
Write-Host "Causal chain: $($state.causal_chain)"

# 4. Get grader score
$grader = curl -X POST http://localhost:7860/grader `
  -H "Content-Type: application/json" `
  -d "{`"episode_id`": `"$ep_id`"}" | ConvertFrom-Json
Write-Host "PSIA: $($grader.session_psia), CCE: $($grader.session_cce)"
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'environment'`**
→ Make sure you're running from the `creditmaze/` directory, not a parent folder.

**`Address already in use`**
→ Another process is on port 7860. Either kill it or change the port:
```powershell
python server.py  # then edit server.py: port=7861
```

**`openenv-core` conflicts in pip install**
→ These are conflicts with OTHER tools (gradio, fastmcp, etc.) in your Anaconda base.
CreditMaze itself does not use openenv-core at runtime — it's only for validation.
The server and tests will work fine.

**`422 Unprocessable Entity` on `/grader`**
→ Make sure you're POSTing JSON body `{"episode_id": "..."}`, not a query parameter.

**Random baseline scores**
→ If no `OPENAI_API_KEY` is set, the baseline uses a random agent. Scores will be low
but consistent — this is expected and does not disqualify the submission.
