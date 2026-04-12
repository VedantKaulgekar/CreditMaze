"""
CreditMaze — FastAPI Server
Compatible with: FastAPI >=0.115, pydantic >=2.7, uvicorn >=0.31
"""
from __future__ import annotations
import subprocess, json, random, os
from typing import Optional, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from environment.env import CreditMazeEnv
from environment.models import Action

app = FastAPI(
    title="CreditMaze",
    version="1.0.0",
    description=(
        "Procedurally generated benchmark for evaluating whether long-horizon AI agents "
        "identify the evidence or decision that actually caused success or failure. "
        "Introduces PSIA, CCE, and MPCS as causal attribution metrics."
    ),
)

env = CreditMazeEnv()

# ── Task config ───────────────────────────────────────────────────────────────

TASK_CONFIG = {
    "task_easy":         {"tier": "easy",        "domain": "triage"},
    "task_medium":       {"tier": "medium",       "domain": "research"},
    "task_hard":         {"tier": "hard",         "domain": "debugging"},
    "resource_hard":     {"tier": "hard",         "domain": "resource"},
    "triage_multipivot": {"tier": "multi-pivot",  "domain": "triage"},
}

TASK_META = {
    "task_easy": {
        "name": "Incident Triage",
        "emoji": "🏥",
        "difficulty": "easy",
        "what": "An AI agent investigates a real-world incident (hospital outbreak, system outage, or conversion drop) and must identify the true causal signal from a set of correlated but non-causal noise signals.",
        "challenge": "Several signals are highly correlated with the outcome and look like strong candidates. Only one has a verified causal mechanism. The agent must distinguish causation from correlation.",
        "why_fail": "The agent flagged a high-correlation signal as causal without verifying whether it had an actual mechanism — confusing correlation with causation.",
        "why_succeed": "The agent correctly identified the signal with a verified causal mechanism, ignoring the high-correlation but non-causal confounders.",
        "what_psia_means": "Did the agent give high credit to the step where it correctly classified the true causal signal — not the earlier steps investigating high-correlation decoys?",
        "steps_explain": "Each step investigates a signal. High correlation is misleading. Only the signal with a verified mechanism is the true cause.",
    },
    "task_medium": {
        "name": "Research Synthesis",
        "emoji": "📚",
        "difficulty": "medium",
        "what": "An AI agent reviews multiple research sources on a topic and must produce a correct synthesis.",
        "challenge": "Two sources directly contradict each other. Several other sources are persuasive but irrelevant. The critical step is synthesising the contradiction correctly.",
        "why_fail": "The agent over-trusted one source and ignored contradicting evidence, producing an overclaimed or underclaimed synthesis.",
        "why_succeed": "The agent correctly identified the conditional contradiction and qualified its synthesis appropriately.",
        "what_psia_means": "Did the agent rate the contradiction-resolution step as the most important — not the earlier source-reading steps which looked busy but were causally irrelevant?",
        "steps_explain": "Each step reviews a source or synthesises findings. Only the synthesis step is causally decisive.",
    },
    "task_hard": {
        "name": "Code Debugging",
        "emoji": "🐛",
        "difficulty": "hard",
        "what": "An AI agent debugs a codebase with multiple bugs and must fix them in the right order.",
        "challenge": "One bug has a hidden upstream dependency. Fixing visible bugs first makes the function look repaired but leaves the root cause latent. The order of the first fix is the only thing that matters.",
        "why_fail": "The agent fixed the most visible bug first, creating a dependency cycle that could not be resolved.",
        "why_succeed": "The agent identified the upstream dependency and fixed the root bug first, clearing the path for all other fixes.",
        "what_psia_means": "Did the agent correctly attribute the outcome to the fix-order decision step, not to all the subsequent bug fixes which were causally irrelevant?",
        "steps_explain": "Each step applies a bug fix. Fix order has a hidden dependency. Only the pivotal ordering decision matters.",
    },
    "resource_hard": {
        "name": "Resource Allocation",
        "emoji": "⚡",
        "difficulty": "hard",
        "what": "An AI agent allocates limited resources for a project, each with different commitment deadlines.",
        "challenge": "One resource has an irreversible commitment window. Once that window closes, no amount of money or time can recover it. Everything else is flexible.",
        "why_fail": "The agent spent budget on flexible resources first and missed the irreversible commitment window for the one resource that couldn't wait.",
        "why_succeed": "The agent identified the one time-critical irreversible resource and committed it before the window closed.",
        "what_psia_means": "Did the agent assign high credit to the irreversible-window step, not to the many routine allocation steps that looked equally important?",
        "steps_explain": "Each step allocates a resource. Most can be renegotiated later. One commitment is permanent and must be made in the right window.",
    },
    "triage_multipivot": {
        "name": "Causal Triage",
        "emoji": "🔬",
        "difficulty": "hard (multi-pivot)",
        "what": "An AI agent investigates an incident and must identify which signals are true causes vs coincidental correlates.",
        "challenge": "Multiple signals are highly correlated with the outcome. But correlation is not causation. Only a subset are actual causes. The AI must distinguish mechanism from coincidence.",
        "why_fail": "The agent confused correlation with causation — flagging a signal that merely correlates with the problem rather than one with a verified causal mechanism.",
        "why_succeed": "The agent correctly identified both causal signals by their mechanism, distinguishing them from high-correlation confounders.",
        "what_psia_means": "Did the agent assign highest importance to the steps where it classified the true causal signals — not the high-correlation decoys?",
        "steps_explain": "Each step analyses a signal. High correlation looks convincing. Only signals with a verified mechanism (not just correlation) are true causes.",
    },
}

# ── Retrospective credit prompt ───────────────────────────────────────────────

RETRO_CREDIT_PROMPT = (
    "The episode is now complete. You have full visibility of every step taken and the final outcome.\n\n"
    "Your task: assign a CAUSAL CREDIT score (0.0 to 1.0) to EACH step.\n"
    "This measures: how much did THIS step CAUSE the final outcome?\n\n"
    "Rules:\n"
    "- The step that directly determined the outcome gets credit close to 1.0\n"
    "- Steps with zero causal effect (decoys) get credit close to 0.0\n"
    "- Completing many steps ≠ those steps caused the outcome. Only the DECISIVE step(s) matter.\n"
    "- Use the step numbers shown in the trajectory above (they start at 1).\n\n"
    "Respond ONLY with a JSON object mapping step number (as shown) to credit float.\n"
    "IMPORTANT: Use the step numbers exactly as shown (1, 2, 3, ...), not zero-based indices.\n"
    "Example for a 5-step episode where step 3 was decisive:\n"
    "{\"1\": 0.05, \"2\": 0.05, \"3\": 0.95, \"4\": 0.05, \"5\": 0.05}\n"
    "No preamble. No explanation. JSON only."
)

UI_SYSTEM_PROMPT = (
    "You are interacting with the CreditMaze benchmark. "
    "Choose exactly one valid action_id from the list you are given. "
    'Respond with JSON only: {"action_id":"...", "reasoning":"..."}'
)


def _homepage_html() -> str:
    task_meta_json = json.dumps(TASK_META)
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>CreditMaze — Can your AI tell which decision actually mattered?</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    :root{
      --bg:#07090f;--bg2:#0d1117;--card:#111927;--border:rgba(255,255,255,.07);
      --border-h:rgba(99,102,241,.3);--t:#e2e8f0;--t2:#94a3b8;--t3:#4b5563;
      --pri:#14b8a6;--sec:#818cf8;--acc:#f59e0b;--ok:#34d399;--err:#f87171;--warn:#fb923c;
      --r:12px;--r2:8px;--rp:999px;
    }
    body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--t);min-height:100vh}
    .wrap{max-width:1100px;margin:0 auto;padding:0 20px 60px}

    nav{display:flex;align-items:center;gap:12px;padding:16px 0 8px}
    .logo{font-size:1.2rem;font-weight:900;background:linear-gradient(135deg,var(--pri),var(--sec));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
    .badge-nav{font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;padding:3px 9px;border-radius:var(--rp);background:rgba(129,140,248,.1);border:1px solid rgba(129,140,248,.2);color:var(--sec)}
    nav .sp{flex:1}
    .nav-link{font-size:.82rem;color:var(--t2);text-decoration:none;padding:5px 10px;border-radius:var(--r2)}
    .nav-link:hover{background:rgba(255,255,255,.04)}

    .hero{text-align:center;padding:48px 16px 40px}
    .hero h1{font-size:clamp(1.9rem,4vw,3.1rem);font-weight:900;letter-spacing:-.04em;line-height:1.1;background:linear-gradient(135deg,#f1f5f9 30%,var(--pri) 70%,var(--sec));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
    .hero-sub{max-width:640px;margin:14px auto 0;color:var(--t2);font-size:.95rem;line-height:1.75}
    .hero-insight{margin:28px auto 0;max-width:640px;padding:18px 22px;border-radius:var(--r);background:rgba(20,184,166,.06);border:1px solid rgba(20,184,166,.15);font-size:.9rem;line-height:1.7;color:var(--t)}
    .hero-insight strong{color:var(--pri)}

    /* KEY DIFFERENCE callout */
    .key-diff{margin:20px auto 0;max-width:640px;padding:16px 20px;border-radius:var(--r);background:rgba(245,158,11,.06);border:1px solid rgba(245,158,11,.18);font-size:.88rem;line-height:1.7;color:var(--t2)}
    .key-diff strong{color:var(--acc)}

    .metric-strip{display:flex;justify-content:center;gap:10px;margin-top:24px;flex-wrap:wrap}
    .mp{padding:8px 16px;border-radius:var(--rp);background:rgba(255,255,255,.04);border:1px solid var(--border);font-size:.8rem;font-weight:600;color:var(--t2)}
    .mp span{color:#fff;font-family:'JetBrains Mono',monospace;margin-right:5px}

    .card{background:var(--card);border:1px solid var(--border);border-radius:18px;padding:28px;margin-bottom:20px}
    .card-title{font-size:1.1rem;font-weight:800;letter-spacing:-.02em;margin-bottom:4px}
    .card-sub{color:var(--t2);font-size:.85rem;line-height:1.6;margin-bottom:20px}

    .task-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px}
    .task-btn{padding:14px;border-radius:var(--r);border:2px solid var(--border);background:rgba(255,255,255,.02);cursor:pointer;text-align:left;transition:all .18s;position:relative}
    .task-btn:hover{border-color:var(--border-h);background:rgba(129,140,248,.05)}
    .task-btn.selected{border-color:var(--pri);background:rgba(20,184,166,.07)}
    .task-btn .tb-emoji{font-size:1.4rem;margin-bottom:6px}
    .task-btn .tb-name{font-size:.88rem;font-weight:700;color:var(--t);line-height:1.2;margin-bottom:4px}
    .task-btn .tb-diff{font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;padding:2px 7px;border-radius:var(--rp)}
    .diff-easy{background:rgba(52,211,153,.1);color:var(--ok);border:1px solid rgba(52,211,153,.2)}
    .diff-medium{background:rgba(245,158,11,.1);color:var(--acc);border:1px solid rgba(245,158,11,.2)}
    .diff-hard{background:rgba(248,113,113,.1);color:var(--err);border:1px solid rgba(248,113,113,.2)}

    .task-explain{margin-top:16px;padding:18px 20px;border-radius:var(--r);background:rgba(255,255,255,.025);border:1px solid var(--border);display:none}
    .task-explain.show{display:block}
    .te-row{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}
    .te-box{padding:12px 14px;border-radius:var(--r2);background:rgba(255,255,255,.02);border:1px solid var(--border)}
    .te-label{font-size:.67rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--t3);margin-bottom:5px}
    .te-text{font-size:.83rem;color:var(--t2);line-height:1.6}

    .cfg-row{display:grid;grid-template-columns:80px 1fr 1fr 1fr;gap:12px;margin-top:16px;align-items:end}
    .field label{display:block;font-size:.69rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--t3);margin-bottom:5px}
    .field input,.field select{width:100%;background:rgba(255,255,255,.04);border:1px solid var(--border);border-radius:var(--r2);padding:10px 12px;color:var(--t);font-size:.87rem;font-family:inherit;outline:none;transition:border .2s}
    .field input:focus,.field select:focus{border-color:var(--sec)}
    .field select option{background:var(--bg2)}
    .mode-note{margin-top:10px;padding:10px 14px;border-radius:var(--r2);background:rgba(129,140,248,.05);border:1px solid rgba(129,140,248,.08);font-size:.8rem;color:var(--t2);line-height:1.6}

    .run-row{margin-top:18px;display:flex;align-items:center;gap:14px}
    .btn-run{padding:13px 32px;border-radius:var(--r);border:none;cursor:pointer;font-weight:700;font-size:.9rem;font-family:inherit;background:linear-gradient(135deg,var(--pri),var(--sec));color:#fff;display:inline-flex;align-items:center;gap:8px;transition:opacity .2s,transform .15s}
    .btn-run:hover{transform:translateY(-1px)}
    .btn-run:disabled{opacity:.4;cursor:not-allowed;transform:none}
    .spin{width:16px;height:16px;border:2px solid rgba(255,255,255,.25);border-top-color:#fff;border-radius:50%;animation:spin .6s linear infinite;display:none}
    .spin.on{display:inline-block}
    @keyframes spin{to{transform:rotate(360deg)}}

    .results{display:none}
    .results.show{display:block}

    /* CREDIT SOURCE BANNER */
    .credit-banner{margin-top:16px;padding:12px 16px;border-radius:var(--r2);font-size:.82rem;line-height:1.6;display:flex;align-items:center;gap:10px}
    .credit-banner.retro{background:rgba(52,211,153,.07);border:1px solid rgba(52,211,153,.2);color:var(--ok)}
    .credit-banner.forward{background:rgba(245,158,11,.07);border:1px solid rgba(245,158,11,.2);color:var(--acc)}

    /* VERDICT BANNER */
    .verdict{margin-top:16px;padding:22px 24px;border-radius:var(--r);display:flex;gap:18px;align-items:flex-start}
    .verdict.success{background:rgba(52,211,153,.07);border:1px solid rgba(52,211,153,.2)}
    .verdict.failure{background:rgba(248,113,113,.07);border:1px solid rgba(248,113,113,.2)}
    .verdict.success-wrong-attr{background:rgba(245,158,11,.05);border:1px solid rgba(245,158,11,.2)}
    .verdict.success-wrong-attr .verdict-title{color:var(--acc)}
    .verdict-icon{font-size:2rem;flex-shrink:0;margin-top:2px}
    .verdict-title{font-size:1.2rem;font-weight:800;margin-bottom:6px}
    .verdict.success .verdict-title{color:var(--ok)}
    .verdict.failure .verdict-title{color:var(--err)}
    .verdict-body{font-size:.9rem;color:var(--t2);line-height:1.75}
    .verdict-body strong{color:var(--t)}

    /* METRIC CARDS — PSIA is hero */
    .metrics-row{display:grid;grid-template-columns:2fr 1fr 1fr 1fr 1fr;gap:10px;margin-top:20px}
    .mc{border-radius:var(--r);padding:16px;text-align:center;background:rgba(255,255,255,.025);border:1px solid var(--border)}
    .mc.hero-metric{background:rgba(129,140,248,.06);border-color:rgba(129,140,248,.2)}
    .mc-label{font-size:.67rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--t3);margin-bottom:6px}
    .mc-val{font-family:'JetBrains Mono',monospace;font-size:1.7rem;font-weight:700;color:#fff}
    .mc-val.ok{color:var(--ok)}.mc-val.err{color:var(--err)}.mc-val.warn{color:var(--acc)}
    .mc-explain{font-size:.7rem;color:var(--t3);margin-top:5px;line-height:1.4}
    .mc.hero-metric .mc-val{font-size:2.4rem}
    .mc.hero-metric .mc-label{color:var(--sec)}

    /* WHAT METRICS MEAN */
    .metrics-explain{margin-top:14px;padding:16px 20px;border-radius:var(--r);background:rgba(255,255,255,.02);border:1px solid var(--border)}
    .me-title{font-size:.75rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--t3);margin-bottom:10px}
    .me-row{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:8px}
    .me-item{padding:10px 12px;border-radius:var(--r2);background:rgba(255,255,255,.02);border:1px solid var(--border)}
    .me-name{font-size:.75rem;font-weight:700;color:var(--sec);margin-bottom:3px;font-family:'JetBrains Mono',monospace}
    .me-desc{font-size:.78rem;color:var(--t2);line-height:1.5}

    /* ATTRIBUTION BOX */
    .attr-box{margin-top:16px;padding:18px 20px;border-radius:var(--r);background:rgba(129,140,248,.04);border:1px solid rgba(129,140,248,.1)}
    .attr-title{font-size:.78rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--sec);margin-bottom:10px}
    .attr-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:8px}
    .attr-item{padding:10px 12px;border-radius:var(--r2);background:rgba(255,255,255,.02);border:1px solid var(--border)}
    .attr-k{font-size:.66rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--t3);margin-bottom:4px}
    .attr-v{font-size:.84rem;font-weight:600;color:var(--t);font-family:'JetBrains Mono',monospace;word-break:break-all}

    /* TIMELINE */
    .tl-item{display:grid;grid-template-columns:28px 1fr;margin-bottom:4px}
    .tl-dot-col{display:flex;flex-direction:column;align-items:center}
    .tl-dot{width:10px;height:10px;border-radius:50%;background:var(--t3);margin-top:5px;flex-shrink:0}
    .tl-dot.fail{background:var(--err)}
    .tl-dot.pivot{background:var(--acc);box-shadow:0 0 0 3px rgba(245,158,11,.2)}
    .tl-dot.success-terminal{background:var(--ok)}
    .tl-line{width:2px;flex:1;background:rgba(255,255,255,.06);margin-top:3px}
    .tl-item:last-child .tl-line{display:none}
    .tl-body{padding:0 0 14px 10px}
    .tl-head{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:4px}
    .tl-step-num{font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--t3)}
    .tl-action{font-family:'JetBrains Mono',monospace;font-size:.85rem;font-weight:600;color:var(--t)}
    .tl-tag{font-size:.66rem;font-weight:700;padding:2px 7px;border-radius:var(--rp);border:1px solid}
    .tl-tag.is-pivot{background:rgba(245,158,11,.1);color:var(--acc);border-color:rgba(245,158,11,.25)}
    .tl-tag.is-decoy{background:rgba(75,85,99,.15);color:var(--t3);border-color:rgba(75,85,99,.2)}
    .tl-tag.is-terminal{background:rgba(52,211,153,.08);color:var(--ok);border-color:rgba(52,211,153,.2)}
    .tl-tag.is-fail{background:rgba(248,113,113,.08);color:var(--err);border-color:rgba(248,113,113,.2)}
    .tl-tag.credit-hi{background:rgba(129,140,248,.12);color:var(--sec);border-color:rgba(129,140,248,.25)}
    .tl-tag.credit-lo{background:rgba(75,85,99,.1);color:var(--t3);border-color:rgba(75,85,99,.15)}
    .tl-ctx{font-size:.82rem;color:var(--t2);line-height:1.55}
    .tl-why{font-size:.78rem;color:var(--t3);margin-top:3px;font-style:italic;line-height:1.5}
    .tl-credit-bar{margin-top:5px;display:flex;align-items:center;gap:6px}
    .tl-credit-track{flex:1;height:4px;background:rgba(255,255,255,.07);border-radius:2px;max-width:120px}
    .tl-credit-fill{height:4px;border-radius:2px;transition:width .3s}
    .tl-credit-label{font-size:.72rem;color:var(--t3);font-family:'JetBrains Mono',monospace}
    .tl-legend{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:14px}
    .tl-legend-item{display:flex;align-items:center;gap:5px;font-size:.74rem;color:var(--t2)}
    .tl-legend-dot{width:9px;height:9px;border-radius:50%;flex-shrink:0}

    /* REWARD DESIGN NOTE */
    .reward-note{margin-top:10px;padding:10px 14px;border-radius:var(--r2);background:rgba(255,255,255,.02);border:1px solid var(--border);font-size:.78rem;color:var(--t3);line-height:1.6}
    .reward-note strong{color:var(--t2)}

    /* TABS */
    .tabs{display:flex;gap:3px;background:rgba(255,255,255,.03);border-radius:var(--r2);padding:4px;border:1px solid var(--border);margin-top:20px}
    .tab-btn{flex:1;padding:9px;font-size:.8rem;font-weight:600;border:none;background:transparent;color:var(--t3);cursor:pointer;border-radius:6px;font-family:inherit;transition:all .18s}
    .tab-btn.act{background:rgba(129,140,248,.12);color:var(--sec)}
    .tab-btn:hover:not(.act){background:rgba(255,255,255,.03)}
    .tab-panel{display:none;padding-top:16px}
    .tab-panel.act{display:block}

    /* HOW IT WORKS */
    .hw-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px}
    .hw-item{padding:18px;border-radius:var(--r);background:rgba(255,255,255,.02);border:1px solid var(--border)}
    .hw-num{font-family:'JetBrains Mono',monospace;font-size:.72rem;font-weight:700;color:var(--pri);margin-bottom:6px}
    .hw-title{font-size:.95rem;font-weight:700;margin-bottom:6px}
    .hw-text{font-size:.83rem;color:var(--t2);line-height:1.65}

    footer{text-align:center;padding:24px 0 0;color:var(--t3);font-size:.78rem;line-height:1.6}
    footer code{padding:2px 5px;border-radius:4px;background:rgba(255,255,255,.05);font-family:'JetBrains Mono',monospace;font-size:.72rem}

    .mt16{margin-top:16px}.mt20{margin-top:20px}
    @keyframes fadeUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}
    .fade{animation:fadeUp .35s ease both}
    @media(max-width:768px){
      .metrics-row{grid-template-columns:1fr 1fr 1fr}
      .cfg-row{grid-template-columns:1fr 1fr}
      .task-grid{grid-template-columns:1fr 1fr}
      .te-row{grid-template-columns:1fr}
    }
    @media(max-width:480px){
      .metrics-row{grid-template-columns:1fr 1fr}
      .cfg-row{grid-template-columns:1fr}
      .task-grid{grid-template-columns:1fr}
    }
  </style>
</head>
<body>
<div class="wrap">

<nav>
  <div class="logo">🧭 CreditMaze</div>
  <span class="badge-nav">Meta RL Hackathon</span>
  <div class="sp"></div>
  <a href="/tasks" class="nav-link">API</a>
  <a href="/health" class="nav-link">Health</a>
</nav>

<section class="hero fade">
  <h1>Can your AI tell which decision actually mattered?</h1>
  <p class="hero-sub">Most benchmarks ask: <em>did the agent succeed?</em> CreditMaze asks something harder: <strong>does the agent know <em>why</em> it succeeded?</strong></p>
  <div class="hero-insight">
    💡 <strong>The core idea:</strong> In a 12-step task, maybe only step 4 truly determined the outcome. Steps 1–3 and 5–12 were noise. Can the AI look back and correctly identify step 4 as the one that mattered? That's credit assignment — the bottleneck for training better RL agents.
  </div>
  <div class="key-diff">
    ⚠️ <strong>This is NOT about taking the right action.</strong> It's about knowing, <em>after the fact</em>, which action caused the outcome. An agent can succeed without understanding why — and CreditMaze detects that failure.
  </div>
  <div class="metric-strip">
    <div class="mp"><span>PSIA</span>Did the AI identify the decisive step?</div>
    <div class="mp"><span>CCE</span>How calibrated are its causal estimates?</div>
    <div class="mp"><span>TSR</span>Did it complete the task at all?</div>
    <div class="mp"><span>MPCS</span>Did it find all pivotal steps? (multi-pivot)</div>
  </div>
</section>

<div class="card fade">
  <div class="card-title">Run an AI Agent on CreditMaze</div>
  <div class="card-sub">Select a task and run an LLM agent through it. After the episode ends, the agent assigns causal credit to each step. PSIA measures whether it correctly identified which step actually determined the outcome.</div>

  <div class="task-grid" id="taskGrid"></div>

  <div class="task-explain" id="taskExplain">
    <div style="font-size:.88rem;font-weight:700;color:var(--t);margin-bottom:4px" id="teTitle"></div>
    <div style="font-size:.83rem;color:var(--t2);line-height:1.65;margin-bottom:12px" id="teWhat"></div>
    <div class="te-row">
      <div class="te-box">
        <div class="te-label">🎯 The credit challenge</div>
        <div class="te-text" id="teChallenge"></div>
      </div>
      <div class="te-box">
        <div class="te-label">📊 What each step means</div>
        <div class="te-text" id="teSteps"></div>
      </div>
    </div>
  </div>

  <div class="cfg-row mt16">
    <div class="field">
      <label>Seed</label>
      <input id="seedInput" type="number" value="42" min="0">
    </div>
    <div class="field">
      <label>Agent Mode</label>
      <select id="runMode">
        <option value="auto">LLM mode (Fallback: Random baseline)</option>
        <option value="random">Random baseline (no LLM needed)</option>
      </select>
    </div>
    <div class="field">
      <label>Model</label>
      <input id="modelInput" type="text" placeholder="e.g. gpt-4o-mini">
    </div>
    <div class="field">
      <label>API Base URL</label>
      <input id="baseUrlInput" type="text" placeholder="https://router.huggingface.co/v1">
    </div>
  </div>
  <div style="margin-top:10px">
    <div class="field">
      <label>API Key <span style="color:var(--t3);font-weight:400;text-transform:none;letter-spacing:0">(optional — only needed for LLM mode)</span></label>
      <input id="apiKeyInput" type="password" placeholder="Temporary session key — not stored">
    </div>
  </div>
  <div class="mode-note" id="modeNote"></div>

  <div class="run-row">
    <button class="btn-run" id="runBtn" disabled>
      <span class="spin" id="runSpin"></span>
      <span id="runBtnText">Select a task first</span>
    </button>
    <span id="runStatus" style="font-size:.82rem;color:var(--t3)"></span>
  </div>

  <div class="results" id="results">

    <!-- CREDIT SOURCE BANNER -->
    <div class="credit-banner" id="creditBanner" style="display:none"></div>

    <!-- VERDICT -->
    <div class="verdict" id="verdictBox">
      <div class="verdict-icon" id="verdictIcon"></div>
      <div>
        <div class="verdict-title" id="verdictTitle"></div>
        <div class="verdict-body" id="verdictBody"></div>
      </div>
    </div>

    <!-- METRIC CARDS — PSIA is the hero -->
    <div class="metrics-row">
      <div class="mc hero-metric">
        <div class="mc-label">PSIA — Did it find the decisive step?</div>
        <div class="mc-val" id="mPsia">—</div>
        <div class="mc-explain">1.0 = correct attribution · 0.0 = wrong step credited</div>
      </div>
      <div class="mc">
        <div class="mc-label">CCE</div>
        <div class="mc-val" id="mCce">—</div>
        <div class="mc-explain">Calibration error (lower = better)</div>
      </div>
      <div class="mc">
        <div class="mc-label">MPCS</div>
        <div class="mc-val" id="mMpcs">—</div>
        <div class="mc-explain">Multi-pivot coordination</div>
      </div>
      <div class="mc">
        <div class="mc-label">TSR</div>
        <div class="mc-val" id="mOutcome">—</div>
        <div class="mc-explain">Task completion (independent of PSIA)</div>
      </div>
      <div class="mc">
        <div class="mc-label">Raw reward</div>
        <div class="mc-val" id="mReward">—</div>
        <div class="mc-explain">Uniform by design — not the signal</div>
      </div>
    </div>

    <div class="metrics-explain" id="metricsExplain" style="display:none">
      <div class="me-title">📖 What these numbers mean for this run</div>
      <div class="me-row" id="metricsExplainBody"></div>
    </div>

    <div class="tabs">
      <button class="tab-btn act" data-tab="tabAttribution">Credit Attribution</button>
      <button class="tab-btn" data-tab="tabTimeline">Step Timeline</button>
      <button class="tab-btn" data-tab="tabWhy">Plain English</button>
      <button class="tab-btn" data-tab="tabTech">Technical</button>
    </div>

    <div class="tab-panel act" id="tabAttribution">
      <div id="attrContent"></div>
    </div>

    <div class="tab-panel" id="tabTimeline">
      <div class="tl-legend">
        <div class="tl-legend-item"><div class="tl-legend-dot" style="background:var(--acc)"></div>Pivotal step (revealed after episode)</div>
        <div class="tl-legend-item"><div class="tl-legend-dot" style="background:var(--ok)"></div>Terminal step (last)</div>
        <div class="tl-legend-item"><div class="tl-legend-dot" style="background:var(--t3)"></div>Decoy step</div>
        <div class="tl-legend-item"><div class="tl-legend-dot" style="background:var(--err)"></div>Failed at pivot</div>
      </div>
      <div id="timeline"></div>
    </div>

    <div class="tab-panel" id="tabWhy">
      <div id="whyContent"></div>
    </div>

    <div class="tab-panel" id="tabTech">
      <div class="attr-grid" id="techContent"></div>
    </div>

  </div>
</div>

<div class="card fade mt16">
  <div class="card-title">How CreditMaze Works</div>
  <div class="card-sub">The benchmark that separates task performance from causal understanding.</div>
  <div class="hw-grid">
    <div class="hw-item">
      <div class="hw-num">01</div>
      <div class="hw-title">Hidden pivotal decision</div>
      <div class="hw-text">Every episode has exactly one (or two) steps that truly determine the outcome. All other steps are carefully designed decoys — they look important but have zero causal effect.</div>
    </div>
    <div class="hw-item">
      <div class="hw-num">02</div>
      <div class="hw-title">Agent plays through</div>
      <div class="hw-text">The AI takes actions one by one. Per-step rewards are kept intentionally <strong>uniform</strong> — this prevents the agent from discovering the pivot just by following reward signals.</div>
    </div>
    <div class="hw-item">
      <div class="hw-num">03</div>
      <div class="hw-title">Retrospective credit call</div>
      <div class="hw-text">After the episode ends, the agent sees its full trajectory + outcome and must assign credit to each step. This is the actual credit assignment test — the core of CreditMaze.</div>
    </div>
    <div class="hw-item">
      <div class="hw-num">04</div>
      <div class="hw-title">PSIA measured</div>
      <div class="hw-text">Ground truth (pivot = 1.0, decoys = 0.0) is verified by counterfactual simulation. PSIA = did the agent assign highest credit to the truly decisive step?</div>
    </div>
  </div>
</div>

<div class="card fade mt16">
  <div class="card-title">Task Catalog</div>
  <div class="card-sub">Five distinct domains. Each tests the same core ability: finding the one decision that actually caused the outcome.</div>
  <div id="catalogList"></div>
</div>

<footer>
  <code>/reset</code> · <code>/step</code> · <code>/credit</code> · <code>/state</code> · <code>/tasks</code> · <code>/grader</code> · <code>/health</code><br>
  The <code>/credit</code> endpoint accepts retrospective credit maps after episode completion — the primary PSIA signal.
</footer>
</div>

<script>
const TASK_META = """ + task_meta_json + r""";
const TASK_ORDER = ['task_easy','task_medium','task_hard','resource_hard','triage_multipivot'];
let selectedTask = null;

function $(id){ return document.getElementById(id); }

// ── TABS ──────────────────────────────────────────────────────────────────────
document.addEventListener('click', e => {
  const btn = e.target.closest('.tab-btn');
  if(!btn) return;
  const panel = btn.dataset.tab;
  btn.closest('.tabs').querySelectorAll('.tab-btn').forEach(b => b.classList.remove('act'));
  btn.classList.add('act');
  btn.closest('.results').querySelectorAll('.tab-panel').forEach(p => p.classList.remove('act'));
  $(panel).classList.add('act');
});

// ── TASK GRID ─────────────────────────────────────────────────────────────────
function buildTaskGrid(tasks){
  const order = TASK_ORDER;
  const byId  = {};
  tasks.forEach(t => byId[t.id] = t);
  const sorted = order.map(id => byId[id]).filter(Boolean);
  const grid = $('taskGrid');
  grid.innerHTML = '';
  sorted.forEach(t => {
    const m = TASK_META[t.id] || {};
    const diff = (t.difficulty||'').toLowerCase();
    const diffClass = diff.includes('hard') ? 'diff-hard' : diff === 'medium' ? 'diff-medium' : 'diff-easy';
    const btn = document.createElement('button');
    btn.className = 'task-btn';
    btn.dataset.taskId = t.id;
    btn.innerHTML = `<div class="tb-emoji">${m.emoji||'🎯'}</div><div class="tb-name">${m.name||t.id}</div><span class="tb-diff ${diffClass}">${t.difficulty||diff}</span>`;
    btn.addEventListener('click', () => selectTask(t.id, m));
    grid.appendChild(btn);
  });
  // build catalog
  const cat = $('catalogList');
  cat.innerHTML = '';
  sorted.forEach(t => {
    const m = TASK_META[t.id] || {};
    const div = document.createElement('div');
    div.style = 'padding:14px 0;border-bottom:1px solid var(--border)';
    div.innerHTML = `<div style="font-weight:700;margin-bottom:4px">${m.emoji||''} ${m.name||t.id} <span style="font-size:.75rem;color:var(--t3);font-weight:400">· ${t.difficulty}</span></div><div style="font-size:.83rem;color:var(--t2);line-height:1.6">${m.what||''}</div>`;
    cat.appendChild(div);
  });
}

function selectTask(taskId, meta){
  selectedTask = taskId;
  document.querySelectorAll('.task-btn').forEach(b => b.classList.toggle('selected', b.dataset.taskId === taskId));
  const ex = $('taskExplain');
  ex.className = 'task-explain show';
  $('teTitle').textContent = (meta.emoji||'') + ' ' + (meta.name||taskId);
  $('teWhat').textContent = meta.what||'';
  $('teChallenge').textContent = meta.challenge||'';
  $('teSteps').textContent = meta.steps_explain||'';
  $('runBtn').disabled = false;
  $('runBtnText').textContent = 'Run Agent on ' + (meta.name||taskId);
  updateModeNote();
}

// ── MODE NOTE ─────────────────────────────────────────────────────────────────
function updateModeNote(){
  const mode = $('runMode').value;
  const note = $('modeNote');
  if(mode === 'random'){
    note.innerHTML = '🎲 <strong>Random baseline:</strong> Agent picks actions at random. Retrospective credits are uniform — PSIA will reflect random chance (~1/N steps). Use this to verify the benchmark runs correctly without needing an API key.';
  } else {
    note.innerHTML = '🤖 <strong>LLM mode:</strong> The agent uses the model you specify to take actions. After the episode ends, the model is asked to review the full trajectory and assign causal credit to each step. PSIA measures whether it correctly identified the decisive step. Falls back to random if the LLM is unavailable.';
  }
}
$('runMode').addEventListener('change', updateModeNote);
updateModeNote();

// ── RUN ───────────────────────────────────────────────────────────────────────
async function runEval(){
  if(!selectedTask) return;
  $('runBtn').disabled = true;
  $('runSpin').classList.add('on');
  $('runBtnText').textContent = 'Running…';
  $('runStatus').textContent = '';
  $('results').className = 'results';

  const payload = {
    task_id:      selectedTask,
    seed:         parseInt($('seedInput').value)||42,
    mode:         $('runMode').value,
    model_name:   $('modelInput').value.trim()||null,
    api_base_url: $('baseUrlInput').value.trim()||null,
    api_key:      $('apiKeyInput').value.trim()||null,
  };

  try {
    const r = await fetch('/demo/run', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
    if(!r.ok){ const t = await r.text(); throw new Error(t); }
    const data = await r.json();
    renderResults(data);
    $('runStatus').textContent = '';
  } catch(e){
    $('runStatus').textContent = '❌ Error: ' + e.message;
  } finally {
    $('runBtn').disabled = false;
    $('runSpin').classList.remove('on');
    $('runBtnText').textContent = 'Run Again';
  }
}

$('runBtn').addEventListener('click', runEval);

// ── RENDER RESULTS ────────────────────────────────────────────────────────────
function renderResults(p) {
  $('results').className = 'results show';
  const m = TASK_META[p.task_id] || {};
  const isSuccess = p.outcome === 'success';
  const g = p.grader || {};
  const creditSource = p.credit_source || 'forward';

  // ── Per-episode values for metric cards (never use session averages here) ─
  // episode_psia/cce come from the retrospective credit call for THIS run only.
  // session_* accumulate across runs and are shown only in the Technical tab.
  // Use typeof check (not != null) to correctly handle 0.0 as a valid score.
  const psia = typeof p.episode_psia === 'number' ? p.episode_psia : null;
  const cce  = typeof p.episode_cce  === 'number' ? p.episode_cce  : null;
  const isMultiPivot = p.is_multi_pivot === true;
  const mpcs = (isMultiPivot && typeof p.episode_mpcs === 'number') ? p.episode_mpcs : null;
  const sessionMpcs = p.session_mpcs != null ? Number(p.session_mpcs) : null;
  // Flag: did we get real per-episode values?
  const hasEpisodeMetrics = psia !== null;

  // ── CREDIT SOURCE BANNER ──────────────────────────────────────────────────
  const banner = $('creditBanner');
  banner.style.display = 'flex';
  if(creditSource === 'retrospective'){
    banner.className = 'credit-banner retro';
    banner.innerHTML = '✅ <div><strong>Retrospective credit collected.</strong> After the episode ended, the LLM reviewed the full trajectory and assigned causal credit to each step. PSIA/CCE below reflect genuine credit assignment ability.</div>';
  } else {
    banner.className = 'credit-banner forward';
    banner.innerHTML = '⚠️ <div><strong>Random baseline mode — no LLM credit assignment.</strong> Actions and credit estimates are random (uniform). PSIA reflects chance (~1/N) rather than genuine credit assignment ability. To measure real credit assignment, use LLM mode with an API key.</div>';
  }

  // ── VERDICT ───────────────────────────────────────────────────────────────
  // Only show full green success if BOTH task completed AND attribution correct.
  // If task succeeded but credit was wrong, show a mixed/muted banner.
  const pivots = g.pivotal_step_indices;
  const attributionCorrect = psia !== null && psia >= 0.8;
  const wrongAttrSuccess = isSuccess && psia !== null && psia < 0.8;

  const vbox = $('verdictBox');
  if(isSuccess && attributionCorrect){
    vbox.className = 'verdict success';
  } else if(isSuccess && wrongAttrSuccess){
    vbox.className = 'verdict success-wrong-attr';  // muted success
  } else {
    vbox.className = 'verdict ' + (isSuccess ? 'success' : 'failure');
  }

  $('verdictIcon').textContent = isSuccess && attributionCorrect ? '✅' : isSuccess ? '⚠️' : '❌';
  if(isSuccess && attributionCorrect){
    $('verdictTitle').textContent = 'Task Completed — Credit Correctly Attributed';
  } else if(isSuccess && wrongAttrSuccess){
    $('verdictTitle').textContent = 'Task Completed — But Wrong Step Credited';
  } else if(isSuccess){
    $('verdictTitle').textContent = 'Task Completed';
  } else {
    $('verdictTitle').textContent = 'Task Failed';
  }

  let verdictText = isSuccess ? (m.why_succeed||'Agent completed the task.') : (m.why_fail||'Agent failed the task.');
  if(isSuccess && wrongAttrSuccess){
    verdictText = `<span style="color:var(--acc)">The task was completed, but the agent misidentified which step caused it.</span> ` + verdictText;
  }
  if(pivots && pivots.length){
    const pWord = pivots.length > 1 ? 'Steps ' + pivots.map(i=>i+1).join(' and ') + ' were' : 'Step ' + (pivots[0]+1) + ' was';
    verdictText += ` <strong>${pWord} the decisive moment</strong> — the action taken there determined the final outcome.`;
  }
  $('verdictBody').innerHTML = verdictText;

  // ── METRIC CARDS ──────────────────────────────────────────────────────────
  const mPsia = $('mPsia');
  if(psia !== null){
    mPsia.textContent = psia.toFixed(2);
    mPsia.className = 'mc-val ' + (psia >= 0.8 ? 'ok' : psia >= 0.4 ? 'warn' : 'err');
  } else {
    mPsia.textContent = '—';
    mPsia.className = 'mc-val';
  }

  const mCce = $('mCce');
  if(cce !== null){
    mCce.textContent = cce.toFixed(2);
    mCce.className = 'mc-val ' + (cce <= 0.15 ? 'ok' : cce <= 0.30 ? 'warn' : 'err');
  } else {
    mCce.textContent = '—';
    mCce.className = 'mc-val';
  }

  const mMpcs = $('mMpcs');
  if(mpcs !== null){
    mMpcs.textContent = mpcs.toFixed(2);
    mMpcs.className = 'mc-val ' + (mpcs >= 0.8 ? 'ok' : mpcs >= 0.4 ? 'warn' : 'err');
  } else {
    mMpcs.textContent = 'N/A';
    mMpcs.className = 'mc-val';
  }

  $('mOutcome').textContent = isSuccess ? '1.00' : '0.00';
  $('mOutcome').className = 'mc-val ' + (isSuccess ? 'ok' : 'err');
  $('mReward').textContent = Number(p.raw_reward||0).toFixed(3);
  $('mReward').className = 'mc-val';

  // ── METRICS EXPLAIN ───────────────────────────────────────────────────────
  const meBox = $('metricsExplain');
  meBox.style.display = 'block';
  const psiaRating = psia === null ? '⚪ Not measured' : psia >= 0.8 ? '🟢 Correct attribution' : psia >= 0.5 ? '🟡 Partial' : '🔴 Wrong attribution';
  const cceRating  = cce === null ? '⚪ Not measured' : cce <= 0.15 ? '🟢 Well calibrated' : cce <= 0.30 ? '🟡 Moderate' : '🔴 Poorly calibrated';
  const psiaExplain = m.what_psia_means || 'Did the agent rate the pivotal step as most causally important?';
  const uniformWarning = creditSource !== 'retrospective'
    ? ' <em style="color:var(--acc)">(⚠️ uniform fallback credits — LLM rate-limited. Run again for real measurement.)</em>' : '';
  const psiaVal = psia !== null ? psia.toFixed(2) : '—';
  const cceVal  = cce  !== null ? cce.toFixed(2)  : '—';
  const psiaDesc = psia === null ? 'No retrospective credit data available for this run.' :
    psia >= 0.8 ? 'The agent correctly identified which step caused the outcome.' :
    psia >= 0.5 ? 'The agent partially identified the decisive step.' :
    'The agent failed to attribute credit to the decisive step — it was fooled by decoys.';
  const cceDesc = cce === null ? 'No retrospective credit data available.' :
    cce <= 0.15 ? 'Very accurate.' : cce <= 0.30 ? 'Reasonably calibrated.' : 'High error — credit estimates were scattered.';
  $('metricsExplainBody').innerHTML = `
    <div class="me-item"><div class="me-name">PSIA = ${psiaVal} — ${psiaRating}</div><div class="me-desc">${psiaExplain} ${psiaDesc}${uniformWarning}</div></div>
    <div class="me-item"><div class="me-name">CCE = ${cceVal} — ${cceRating}</div><div class="me-desc">Calibration error between the agent's credit estimates and ground truth across all steps. Lower = better. ${cceDesc}${uniformWarning}</div></div>
    <div class="me-item"><div class="me-name">TSR — ${isSuccess ? 'Completed ✅' : 'Failed ❌'}</div><div class="me-desc">${isSuccess ? 'Agent completed the task.' : 'Agent failed the task.'} <em>TSR and PSIA are independent — an agent can succeed without correctly attributing why.</em></div></div>
    ${mpcs !== null ? `<div class="me-item"><div class="me-name">MPCS = ${mpcs.toFixed(2)}</div><div class="me-desc">Multi-pivot: both pivotal steps must be identified. 1.0 = found all, 0.5 = found one, 0.0 = found neither.</div></div>` : (isMultiPivot ? '' : '<div class="me-item"><div class="me-name">MPCS — N/A</div><div class="me-desc">Not a multi-pivot task. MPCS only applies to the Causal Triage task.</div></div>')}
    <div class="me-item"><div class="me-name">Reward = ${Number(p.raw_reward||0).toFixed(3)}</div><div class="me-desc">Per-step rewards are intentionally uniform (0.06 each) — this is by design. If the pivot gave a higher reward, the agent could cheat by following reward signals instead of reasoning about causation.</div></div>
  `;

  // ── ATTRIBUTION TAB (now the hero tab) ────────────────────────────────────
  const steps     = p.steps || [];
  const topStep   = g.top_attributed_step;        // 0-based
  const pivotRank = g.pivotal_step_rank;
  const falsePos  = g.false_positive_steps || [];
  const attrGap   = g.attribution_gap;
  const wrongAttr = g.success_with_wrong_attribution;

  let attrHtml = `<div class="attr-box">
    <div class="attr-title">🔍 Credit Assignment Result — What the agent thought caused the outcome</div>
    <div style="font-size:.85rem;color:var(--t2);line-height:1.7;margin-bottom:16px">
      After the episode ended, the agent was shown its full trajectory and the outcome, then asked:
      <em>"Looking back, which step actually caused this result?"</em>
      The table below compares its answer to the ground truth verified by counterfactual simulation.
    </div>
    <div class="attr-grid">`;

  const attrItems = [
    ['Agent credited most to', topStep !== null && topStep !== undefined ? 'Step ' + (topStep+1) : 'Unknown'],
    ['Actually decisive step', pivots ? 'Step ' + pivots.map(i=>i+1).join(' & ') : 'Unknown'],
    ['Attribution correct?', topStep !== null && pivots && pivots.includes(topStep) ? '✅ Yes' : '❌ No — fooled by decoy'],
    ['Pivotal step ranked', pivotRank ? '#' + pivotRank + ' of ' + steps.length : 'Unknown'],
    ['Attribution gap', attrGap !== null && attrGap !== undefined ? Number(attrGap).toFixed(3) + (attrGap > 0.2 ? ' ✓ good' : ' ✗ poor') : 'Unknown'],
    ['Decoys incorrectly credited', falsePos.length ? 'Steps ' + falsePos.map(i=>i+1).join(', ') : 'None'],
    ['Succeeded with wrong attribution?', wrongAttr ? '⚠️ Yes — succeeded by chance' : isSuccess ? 'No — correctly attributed' : 'N/A (failed)'],
    ['Credit source', creditSource === 'retrospective' ? '✅ Retrospective LLM call' : '⚠️ Forward guesses (fallback)'],
  ];
  attrItems.forEach(([k, v]) => {
    attrHtml += `<div class="attr-item"><div class="attr-k">${k}</div><div class="attr-v">${v}</div></div>`;
  });
  attrHtml += `</div></div>`;

  if(wrongAttr && isSuccess){
    attrHtml += `<div style="margin-top:12px;padding:12px 16px;border-radius:8px;background:rgba(245,158,11,.06);border:1px solid rgba(245,158,11,.18);font-size:.85rem;color:var(--t2);line-height:1.7">
      ⚠️ <strong style="color:var(--acc)">Success with wrong attribution:</strong> The agent completed the task but misidentified which step caused it. This is like a student passing an exam by guessing correctly — the right answer, but for the wrong reason. In RL training, this would reinforce the wrong behaviour.
    </div>`;
  }

  attrHtml += `<div style="margin-top:14px;padding:16px 18px;border-radius:10px;background:rgba(255,255,255,.02);border:1px solid var(--border)">
    <div style="font-size:.78rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--t3);margin-bottom:10px">Why this matters for RL training</div>
    <p style="font-size:.85rem;color:var(--t2);line-height:1.7">
      RL algorithms like GRPO and PPO assign credit backwards through time — deciding which past action caused the current reward.
      If the algorithm credits the wrong step, it <strong style="color:var(--t)">reinforces the wrong behaviour</strong>, even when the agent succeeds by luck.
      PSIA and CCE directly measure this failure mode. A benchmark that only tracks TSR (task success) completely misses it.
    </p>
  </div>`;

  $('attrContent').innerHTML = attrHtml;

  // ── TIMELINE TAB ──────────────────────────────────────────────────────────
  const tl = $('timeline');
  tl.innerHTML = '';

  // Build agent credit map from retrospective data if available
  const agentCreditMap = p.agent_credits || {};  // step_index (str) -> normalised credit

  steps.forEach((s, i) => {
    // s.step_index = 0-based canonical index, s.step = 1-based display label
    const stepIdx = s.step_index !== undefined ? s.step_index : i;
    const isPiv = pivots && pivots.includes(stepIdx);
    const isTerminal = s.done && !isPiv;
    const isFailPiv  = s.done && s.reward === 0 && isPiv;
    const isFailTerm = s.done && s.reward === 0 && !isPiv;

    let dotClass = 'tl-dot';
    if(isPiv && !isFailPiv) dotClass = 'tl-dot pivot';
    else if(isFailPiv || isFailTerm) dotClass = 'tl-dot fail';
    else if(isTerminal && isSuccess) dotClass = 'tl-dot success-terminal';

    const roleLabel = isPiv
      ? '🔑 Decisive step'
      : (isFailPiv ? '💥 Failed here' : (s.done ? (isSuccess ? '🏁 Terminal' : '❌ Terminal') : '· Decoy'));
    const roleClass = isPiv ? 'tl-tag is-pivot' : (s.done ? 'tl-tag is-terminal' : 'tl-tag is-decoy');

    const agentCredit = agentCreditMap[String(stepIdx)];
    // Detect uniform fallback: all credits equal means LLM rate-limited
    const allCreditVals = Object.values(agentCreditMap).map(Number);
    const isUniform = allCreditVals.length > 0 && allCreditVals.every(v => Math.abs(v - allCreditVals[0]) < 0.001);
    let creditHtml = '';
    if(agentCredit !== undefined){
      const barColor = isPiv ? 'var(--acc)' : (agentCredit > 0.6 ? 'var(--sec)' : 'var(--t3)');
      const creditClass = agentCredit > 0.5 ? 'tl-tag credit-hi' : 'tl-tag credit-lo';
      const creditLabel = isUniform
        ? `<span class="tl-tag credit-lo" style="font-style:italic">uniform (fallback)</span>`
        : `<span class="${creditClass}">credit ${(agentCredit*100).toFixed(0)}%</span>`;
      const pivotNote = !isUniform && isPiv ? '← agent credited this most' : (isPiv ? '' : '');
      creditHtml = `${creditLabel}
        <div class="tl-credit-bar">
          <div class="tl-credit-track"><div class="tl-credit-fill" style="width:${(agentCredit*100).toFixed(1)}%;background:${barColor}"></div></div>
          <span class="tl-credit-label">${pivotNote}</span>
        </div>`;
    }

    const isTerminalStep = s.done && !isFailPiv;
    const why = isPiv && !isFailPiv
      ? 'Causal ground truth: this step determined the final outcome. Every other step was a decoy with no causal effect.'
      : (isFailPiv ? 'Wrong action at the pivotal step — episode failed immediately.'
      : (isTerminalStep && isSuccess
          ? '⚠️ The large reward here (0.67) is NOT a signal of importance — it fires on the last step regardless of where the pivot was. This is by design: uniform per-step rewards prevent the agent from finding the pivot by following rewards.'
          : (isTerminalStep ? 'Episode ended here.' : 'Decoy step — looked important, had zero causal effect on the outcome.')));

    // For terminal success step, show reward differently to avoid confusion
    const rewardBadge = isTerminalStep && isSuccess
      ? `<span class="tl-tag is-terminal" title="Terminal reward — fires here by design, NOT a causal signal">terminal reward ${s.reward.toFixed(2)}</span>`
      : isFailPiv || (!isSuccess && s.done)
        ? `<span class="tl-tag is-fail">reward ${s.reward.toFixed(2)}</span>`
        : `<span class="tl-tag is-decoy">reward ${s.reward.toFixed(2)}</span>`;

    const div = document.createElement('div');
    div.className = 'tl-item';
    div.innerHTML = `<div class="tl-dot-col"><div class="${dotClass}"></div><div class="tl-line"></div></div>
      <div class="tl-body">
        <div class="tl-head">
          <span class="tl-step-num">Step ${s.step}</span>
          <span class="tl-action">${s.action}</span>
          <span class="${roleClass}">${roleLabel}</span>
          ${rewardBadge}
        </div>
        ${creditHtml}
        <div class="tl-ctx">${(s.context_snippet||'').slice(0,200)}</div>
        <div class="tl-why">${why}</div>
      </div>`;
    tl.appendChild(div);
  });

  if(steps.length){
    const note = document.createElement('div');
    note.className = 'reward-note';
    note.innerHTML = `<strong>Why do all steps show reward 0.06?</strong> Per-step rewards are intentionally uniform across decoys AND the decisive step. If the pivot gave a higher reward, the agent could cheat by following reward signals instead of reasoning causally. The large terminal reward (${steps[steps.length-1]?.reward?.toFixed(2)||'—'}) fires at the last step regardless — this is the design.`;
    tl.appendChild(note);
  }

  // ── WHY TAB ───────────────────────────────────────────────────────────────
  let whyHtml = `<div style="padding:18px 20px;border-radius:10px;background:rgba(255,255,255,.02);border:1px solid var(--border);margin-bottom:14px">
    <div style="font-size:.78rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--t3);margin-bottom:10px">🎯 What happened — plain English</div>
    <p style="font-size:.9rem;line-height:1.75;color:var(--t);margin-bottom:12px">
      The agent <strong style="color:${isSuccess?'var(--ok)':'var(--err)'}">${isSuccess?'succeeded':'failed'}</strong> after <strong>${steps.length} step${steps.length!==1?'s':''}</strong>. ${isSuccess?(m.why_succeed||''):(m.why_fail||'')}
    </p>`;

  if(pivots && pivots.length){
    const pStr = pivots.length > 1 ? 'Steps ' + pivots.map(i=>i+1).join(' and ') : 'Step ' + (pivots[0]+1);
    whyHtml += `<p style="font-size:.88rem;line-height:1.7;color:var(--t2);margin-bottom:10px">
      📍 <strong style="color:var(--t)">The decisive moment${pivots.length>1?'s':''}:</strong> ${pStr} ${pivots.length>1?'were':'was'} the causal step${pivots.length>1?'s':''}. Every other step was a decoy — it contributed to the episode narrative but had zero effect on the outcome.
    </p>`;
  }

  if(topStep !== undefined && topStep !== null && pivots && pivots.length){
    const agentCorrect = pivots.includes(topStep);
    whyHtml += `<p style="font-size:.88rem;line-height:1.7;color:var(--t2);margin-bottom:10px">
      🧠 <strong style="color:var(--t)">The agent retrospectively credited Step ${topStep+1} most.</strong>
      ${agentCorrect
        ? '<strong style="color:var(--ok)">✓ Correct.</strong> It identified the right pivotal step.'
        : `<strong style="color:var(--err)">✗ Wrong.</strong> The actually decisive step was Step ${pivots[0]+1}. The agent was fooled by a decoy.`}
    </p>`;
  }

  if(attrGap !== undefined && attrGap !== null){
    const gapGood = attrGap > 0.2;
    whyHtml += `<p style="font-size:.85rem;line-height:1.65;color:var(--t2)">
      📊 <strong style="color:var(--t)">Attribution gap: ${Number(attrGap).toFixed(3)}</strong>
      ${gapGood ? ' — The agent gave noticeably higher credit to the decisive step vs decoys. Good separation.' : ' — The agent barely distinguished the decisive step from decoys. Poor causal understanding.'}
    </p>`;
  }

  whyHtml += '</div>';
  $('whyContent').innerHTML = whyHtml;

  // ── TECH TAB ──────────────────────────────────────────────────────────────
  const techItems = [
    ['Episode ID', p.episode_id],
    ['Task ID', p.task_id],
    ['Mode used', p.mode_used],
    ['Model', p.model_label||'random-baseline'],
    ['Credit source', creditSource],
    ['LLM fallback?', p.used_fallback ? 'Yes — ' + (p.fallback_reason||'') : 'No'],
    ['Outcome', p.outcome],
    ['Raw reward', Number(p.raw_reward||0).toFixed(4)],
    ['Normalised score', Number(p.score||0).toFixed(4)],
    ['Steps taken', steps.length],
    ['Decisive step(s)', pivots ? pivots.map(i=>i+1).join(', ') : '—'],
    ['Agent top-credited', topStep !== null ? 'Step '+(topStep+1) : '—'],
    ['── This episode ──', ''],
    ['PSIA (this run)', psia.toFixed(4)],
    ['CCE (this run)', cce.toFixed(4)],
    ['MPCS (this run)', isMultiPivot && p.episode_mpcs != null ? Number(p.episode_mpcs).toFixed(4) : 'N/A — not a multi-pivot task'],
    ['Attribution gap', attrGap !== null ? Number(attrGap).toFixed(4) : '—'],
    ['── Session averages ──', ''],
    ['Session PSIA', p.session_psia != null ? Number(p.session_psia).toFixed(4) : '—'],
    ['Session CCE', p.session_cce != null ? Number(p.session_cce).toFixed(4) : '—'],
    ['Session MPCS', sessionMpcs !== null ? sessionMpcs.toFixed(4) + ' (avg across multi-pivot runs)' : 'N/A — no multi-pivot runs yet'],
    ['Causal faithfulness', g.causal_faithfulness != null ? Number(g.causal_faithfulness).toFixed(3) : '—'],
  ];
  $('techContent').innerHTML = techItems.map(([k,v]) => {
    if(k.startsWith('──')) return `<div class="attr-item" style="grid-column:1/-1;background:transparent;border-color:transparent;padding:4px 0 0"><div class="attr-k" style="color:var(--sec)">${k}</div></div>`;
    return `<div class="attr-item"><div class="attr-k">${k}</div><div class="attr-v">${v??'—'}</div></div>`;
  }).join('');

  // Switch to Attribution tab
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('act'));
  document.querySelectorAll('.tab-panel').forEach(p2 => p2.classList.remove('act'));
  document.querySelector('[data-tab="tabAttribution"]').classList.add('act');
  $('tabAttribution').classList.add('act');
}

// ── LOAD TASKS ────────────────────────────────────────────────────────────────
async function loadTasks() {
  try {
    const r = await fetch('/tasks');
    const p = await r.json();
    buildTaskGrid(p.tasks || []);
  } catch(e) {
    $('taskGrid').innerHTML = '<p style="color:var(--err);font-size:.85rem">Failed to load tasks: ' + e.message + '</p>';
  }
}
loadTasks();
</script>
</body>
</html>
"""


def _make_demo_client(req: "DemoRunRequest") -> tuple:
    try:
        from openai import OpenAI
    except ImportError:
        return None, "random-baseline", "openai_not_installed", "random"

    default_key = req.api_key or ""
    if not default_key:
        default_key = (
            os.getenv("API_KEY") or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or ""
        )
    default_base  = req.api_base_url or "https://router.huggingface.co/v1"
    default_model = req.model_name   or "Qwen/Qwen2.5-72B-Instruct"

    if req.mode == "random":
        return None, "random-baseline", None, "random"
    # "auto" mode: try LLM, fall back to random if no key or unavailable
    if not default_key:
        return None, "random-baseline", "no_credentials", "random"

    client = OpenAI(base_url=default_base, api_key=default_key, max_retries=0)
    return client, default_model, None, "llm"


def _choose_demo_action(client, model_name: str, obs: dict, task_id: str) -> tuple:
    """Choose next action. Does NOT ask for credit_estimate — that's retrospective."""
    if client is None:
        return ({"action_id": random.choice(obs["available_actions"]), "reasoning": "Random baseline"},
                "random_fallback:no_credentials", True)
    prompt = (
        f"Task: {task_id}\nDomain: {obs['domain']}\nTier: {obs['tier']}\n"
        f"Step: {obs['step_count'] + 1}/{obs['t_total']}\n"
        f"Context:\n{obs['context']}\n\nAvailable actions: {obs['available_actions']}"
    )
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": UI_SYSTEM_PROMPT},
                      {"role": "user", "content": prompt}],
            temperature=0, stream=False,
        )
        content = (completion.choices[0].message.content or "").strip()
        parsed  = json.loads(content)
        return parsed, None, False
    except Exception as exc:
        return ({"action_id": random.choice(obs["available_actions"]), "reasoning": "Random fallback"},
                f"random_fallback:{type(exc).__name__}", True)


def _collect_retrospective_credits(
    client,
    model_name: str,
    episode_id: str,
    steps: list,
    outcome: str,
    task_id: str,
) -> tuple[Dict[str, float], bool]:
    """
    After episode ends: ask the LLM to assign retrospective credit to each step.
    Returns (credits_dict, used_fallback).
    credits_dict maps str(step_index) -> float [0,1].
    """
    if client is None:
        # Uniform fallback for random mode — signals no real credit assignment
        return {str(i): 1.0/max(len(steps),1) for i in range(len(steps))}, True

    # Build trajectory summary
    traj_lines = []
    for s in steps:
        traj_lines.append(f"  Step {s['step']}: action={s['action']}")
        if s.get('context_snippet'):
            traj_lines.append(f"    Context: {s['context_snippet'][:120]}")

    prompt = (
        f"Task: {task_id}\n"
        f"Final outcome: {outcome.upper()}\n\n"
        f"Trajectory ({len(steps)} steps):\n" + "\n".join(traj_lines) + "\n\n"
        + RETRO_CREDIT_PROMPT
    )

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0, stream=False,
        )
        raw_content = (completion.choices[0].message.content or "").strip()
        # Strip markdown code fences if present
        if raw_content.startswith("```"):
            raw_content = raw_content.split("```")[1]
            if raw_content.startswith("json"):
                raw_content = raw_content[4:]
        raw = json.loads(raw_content.strip())
        n_steps = len(steps)
        credits: Dict[str, float] = {}
        for k, v in raw.items():
            try:
                key_int = int(k)
                val = float(max(0.0, min(1.0, v)))
                # Prompt uses 1-based step numbers. Translate to 0-based indices.
                # If LLM returned 1-based (1..N), subtract 1.
                # If LLM returned 0-based (0..N-1), use as-is.
                # Heuristic: if max key >= n_steps, it must be 1-based.
                credits[k] = val
            except (ValueError, TypeError):
                pass
        # Detect and translate 1-based keys: if keys run 1..N instead of 0..N-1
        if credits:
            max_key = max(int(k) for k in credits)
            min_key = min(int(k) for k in credits)
            if max_key == n_steps and min_key >= 1:
                # LLM used 1-based numbering — translate to 0-based
                credits = {str(int(k) - 1): v for k, v in credits.items()}
            elif max_key >= n_steps:
                # Keys out of range — clamp to valid indices
                credits = {str(min(int(k), n_steps-1)): v for k, v in credits.items()}
        return credits, False
    except Exception as exc:
        # Fallback: uniform
        return {str(i): 1.0/max(len(steps),1) for i in range(len(steps))}, True


# ── Request schemas ───────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    tier:   str = "easy"
    domain: Optional[str] = None
    seed:   Optional[int] = None


class StepRequest(BaseModel):
    episode_id:      str
    action_id:       str
    reasoning:       Optional[str]   = None
    credit_estimate: Optional[float] = None


class CreditRequest(BaseModel):
    episode_id: str
    credits:    Dict[str, float]  # {step_index_str: credit_float}


class GraderRequest(BaseModel):
    episode_id: str


class DemoRunRequest(BaseModel):
    task_id:      str
    seed:         Optional[int] = 42
    mode:         str = "auto"
    api_base_url: Optional[str] = None
    model_name:   Optional[str] = None
    api_key:      Optional[str] = None


ResetRequest.model_rebuild()
StepRequest.model_rebuild()
CreditRequest.model_rebuild()
GraderRequest.model_rebuild()
DemoRunRequest.model_rebuild()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/reset")
def reset(req: ResetRequest):
    valid_tiers = ["easy", "medium", "hard", "multi-pivot"]
    if req.tier not in valid_tiers:
        raise HTTPException(400, f"tier must be one of {valid_tiers}")
    return env.reset(tier=req.tier, domain=req.domain, seed=req.seed).model_dump()


@app.post("/step")
def step(req: StepRequest):
    try:
        result = env.step(req.episode_id, Action(
            action_id=req.action_id, reasoning=req.reasoning,
            credit_estimate=req.credit_estimate,
        ))
        return result.model_dump()
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/credit")
def credit(req: CreditRequest):
    """
    Submit retrospective credit assignments after episode completion.
    This is the PRIMARY credit signal for PSIA/CCE/MPCS.
    Call once after done=True with a map of {step_index_str: credit_float}.
    """
    try:
        result = env.submit_retrospective_credits(req.episode_id, req.credits)
        return result
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/state")
def state(episode_id: str):
    try:
        return env.state(episode_id).model_dump()
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.get("/tasks")
def tasks():
    def grader_meta(task_id: str, summary: str) -> dict:
        return {
            "type": "programmatic",
            "description": (
                f"Score 0.01–0.99 based on PSIA (primary) and TSR (secondary). "
                f"PSIA measures retrospective credit assignment accuracy. {summary}"
            ),
        }
    return {
        "tasks": [
            {"id": "task_easy",        "difficulty": "easy",        "max_steps": 15, "grader": grader_meta("task_easy",        "Corridor navigation — one junction is decisive.")},
            {"id": "task_medium",      "difficulty": "medium",      "max_steps": 20, "grader": grader_meta("task_medium",      "Research synthesis — one contradiction must be resolved correctly.")},
            {"id": "task_hard",        "difficulty": "hard",        "max_steps": 18, "grader": grader_meta("task_hard",        "Code debugging — fix order has a hidden dependency.")},
            {"id": "resource_hard",    "difficulty": "hard",        "max_steps": 18, "grader": grader_meta("resource_hard",    "Resource allocation — one resource has an irreversible commitment window.")},
            {"id": "triage_multipivot","difficulty": "multi-pivot", "max_steps": 18, "grader": grader_meta("triage_multipivot","Multi-pivot triage — two causal signals must both be identified.")},
        ]
    }


@app.post("/grader")
def grader(req: GraderRequest):
    try:
        s = env.state(req.episode_id)
    except KeyError as e:
        raise HTTPException(404, str(e))
    if not s.episode_complete:
        raise HTTPException(400, "Episode not complete.")
    return {
        "episode_id":                     req.episode_id,
        "outcome":                        s.outcome,
        "score":                          env.normalized_score(req.episode_id),
        "raw_reward":                     round(s.cumulative_reward, 4),
        "session_psia":                   s.session_psia,
        "session_cce":                    s.session_cce,
        "session_tsr":                    s.session_tsr,
        "session_mpcs":                   s.session_mpcs,
        "episodes_completed":             s.episodes_completed,
        "causal_faithfulness":            s.causal_faithfulness,
        "pivotal_step_indices":           s.pivotal_step_indices,
        "top_attributed_step":            s.top_attributed_step,
        "top_attributed_action":          s.top_attributed_action,
        "top_attributed_credit":          s.top_attributed_credit,
        "pivotal_step_rank":              s.pivotal_step_rank,
        "false_positive_steps":           s.false_positive_steps,
        "attribution_gap":                s.attribution_gap,
        "success_with_wrong_attribution": s.success_with_wrong_attribution,
        "credit_source":                  s.credit_source,
    }


@app.post("/baseline")
def baseline():
    try:
        result = subprocess.run(
            ["python", "baseline.py", "--n", "3"],
            capture_output=True, timeout=600, text=True,
        )
        if result.returncode != 0:
            raise HTTPException(500, f"Baseline script failed:\n{result.stderr[:800]}")
        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        return json.loads(lines[-1])
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "Baseline script timed out")
    except json.JSONDecodeError:
        raise HTTPException(500, "Baseline did not produce valid JSON")


@app.post("/demo/run")
def demo_run(req: DemoRunRequest):
    if req.task_id not in TASK_CONFIG:
        raise HTTPException(400, f"Unknown task_id '{req.task_id}'")

    client, model_label, startup_reason, mode_used = _make_demo_client(req)
    cfg = TASK_CONFIG[req.task_id]
    obs = env.reset(tier=cfg["tier"], domain=cfg["domain"], seed=req.seed)
    episode_id    = obs.episode_id
    steps         = []
    used_fallback = startup_reason is not None
    fallback_reason = startup_reason

    # ── Play through episode ──────────────────────────────────────────────────
    while True:
        context_before = obs.context
        obs_dict = obs.model_dump()
        decision, step_error, step_fallback = _choose_demo_action(client, model_label, obs_dict, req.task_id)
        action_id = decision.get("action_id", obs.available_actions[0])
        if action_id not in obs.available_actions:
            action_id = obs.available_actions[0]
        if step_fallback:
            used_fallback   = True
            fallback_reason = step_error
            mode_used       = "random-fallback"

        step_index = obs.step_count   # 0-based index BEFORE incrementing

        result = env.step(episode_id, Action(
            action_id=action_id,
            reasoning=decision.get("reasoning"),
            # No credit_estimate here — credit is collected retrospectively
        ))
        steps.append({
            "step":            result.observation.step_count,   # 1-based display
            "step_index":      step_index,                      # 0-based for matching
            "action":          action_id,
            "reward":          result.reward,
            "done":            result.done,
            "error":           step_error,
            "context_snippet": context_before[:220],
        })
        obs = result.observation
        if result.done:
            break

    outcome = obs.episode_outcome or "failure"

    # ── Retrospective credit call ─────────────────────────────────────────────
    retro_credits, retro_fallback = _collect_retrospective_credits(
        client, model_label, episode_id, steps, outcome, req.task_id
    )
    if retro_fallback:
        used_fallback   = True
        fallback_reason = (fallback_reason or "") + "+retro_fallback"

    credit_result = env.submit_retrospective_credits(episode_id, retro_credits)
    credit_source = credit_result.get("credit_source", "forward")

    grader_payload = grader(GraderRequest(episode_id=episode_id))

    # Normalise retro credits for display. When all equal (uniform fallback),
    # keep at 0.5 so bars are visible rather than collapsing to 0.
    credits_vals = list(retro_credits.values())
    if not credits_vals:
        agent_credits_normalised = {}
    else:
        mn, mx = min(credits_vals), max(credits_vals)
        if mx == mn:
            agent_credits_normalised = {k: 0.5 for k in retro_credits}
        else:
            agent_credits_normalised = {
                k: round((v - mn) / (mx - mn), 4) for k, v in retro_credits.items()
            }

    cfg_tier = TASK_CONFIG[req.task_id]["tier"]
    is_multi_pivot = (cfg_tier == "multi-pivot")

    return {
        "task_id":         req.task_id,
        "episode_id":      episode_id,
        "mode_used":       mode_used,
        "model_label":     model_label,
        "used_fallback":   used_fallback,
        "fallback_reason": fallback_reason,
        "outcome":         grader_payload["outcome"],
        "score":           grader_payload["score"],
        "raw_reward":      grader_payload["raw_reward"],
        # Per-episode metrics (what this run scored)
        "episode_psia":    credit_result.get("psia_score"),
        "episode_cce":     credit_result.get("cce"),
        "episode_mpcs":    credit_result.get("mpcs") if is_multi_pivot else None,
        # Session averages (across all runs this session)
        "session_psia":    grader_payload["session_psia"],
        "session_cce":     grader_payload["session_cce"],
        "session_mpcs":    grader_payload.get("session_mpcs"),
        "is_multi_pivot":  is_multi_pivot,
        "credit_source":   credit_source,
        "agent_credits":   agent_credits_normalised,  # for timeline bars
        "final_context":   obs.context,
        "steps":           steps,
        "grader":          grader_payload,
    }


@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    accept     = request.headers.get("accept", "")
    user_agent = request.headers.get("user-agent", "").lower()
    if "text/html" in accept or "mozilla" in user_agent:
        return HTMLResponse(_homepage_html())
    return JSONResponse({
        "name": "CreditMaze", "status": "ok",
        "message": "CreditMaze — credit assignment benchmark.",
        "endpoints": ["/health", "/tasks", "/reset", "/step", "/credit", "/state", "/grader", "/baseline"],
    })


@app.get("/ui", response_class=HTMLResponse)
def ui():
    return HTMLResponse(_homepage_html())


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.1.0", "environment": "CreditMaze"}


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()