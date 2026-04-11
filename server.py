"""
CreditMaze — FastAPI Server
Compatible with: FastAPI >=0.115, pydantic >=2.7, uvicorn >=0.31

Endpoints (all required by OpenEnv spec):
  POST /reset     — start new episode
  POST /step      — submit action
  GET  /state     — get episode state (causal labels revealed after done)
  GET  /tasks     — list tasks + action schema
  POST /grader    — get grader metrics for completed episode
  POST /baseline  — trigger baseline inference script
  GET  /health    — health check
"""
from __future__ import annotations
import subprocess, json, random
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from openai import OpenAI

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

TASK_CONFIG = {
    "task_easy": {"tier": "easy", "domain": "corridor"},
    "task_medium": {"tier": "medium", "domain": "research"},
    "task_hard": {"tier": "hard", "domain": "debugging"},
    "resource_hard": {"tier": "hard", "domain": "resource"},
    "triage_multipivot": {"tier": "multi-pivot", "domain": "triage"},
}

UI_SYSTEM_PROMPT = (
    "You are interacting with the CreditMaze benchmark. "
    "Choose exactly one valid action_id from the list you are given, and estimate "
    "how important the current step is to final success. "
    'Respond with JSON only: {"action_id":"...", "reasoning":"...", "credit_estimate":0.0}'
)


def _homepage_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CreditMaze — Causal Decision Attribution Benchmark</title>
  <meta name="description" content="CreditMaze: A long-horizon benchmark for evaluating causal decision attribution in AI agents.">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    :root{
      --bg:#06080f;--bg2:#0c1020;--card:#111827;--card-b:rgba(99,102,241,.08);
      --card-h:rgba(99,102,241,.14);--t:#e2e8f0;--t2:#94a3b8;--t3:#475569;
      --pri:#14b8a6;--sec:#818cf8;--acc:#f59e0b;--ok:#34d399;--err:#fb7185;
      --r:14px;--r2:10px;--r3:20px;
      --sh:0 1px 3px rgba(0,0,0,.4);
    }
    body{font-family:'Inter',system-ui,sans-serif;background:var(--bg);color:var(--t);min-height:100vh;overflow-x:hidden}
    body::before{content:"";position:fixed;inset:0;background:radial-gradient(ellipse 70% 50% at 20% 0%,rgba(99,102,241,.06),transparent),radial-gradient(ellipse 50% 40% at 80% 100%,rgba(20,184,166,.04),transparent);pointer-events:none}
    .app{position:relative;z-index:1;max-width:1120px;margin:0 auto;padding:20px 20px 40px}

    /* ── NAV BAR ── */
    .nav{display:flex;align-items:center;gap:14px;padding:14px 0;margin-bottom:8px}
    .nav .logo{font-size:1.25rem;font-weight:900;letter-spacing:-.03em;background:linear-gradient(135deg,var(--pri),var(--sec));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
    .nav .badge{font-size:.7rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;padding:4px 10px;border-radius:999px;background:rgba(99,102,241,.1);border:1px solid rgba(99,102,241,.18);color:var(--sec)}
    .nav .spacer{flex:1}
    .nav .link{font-size:.82rem;color:var(--t2);text-decoration:none;padding:6px 12px;border-radius:8px;transition:background .2s}
    .nav .link:hover{background:rgba(255,255,255,.04)}

    /* ── HERO ── */
    .hero{text-align:center;padding:40px 20px 32px}
    .hero h1{font-size:clamp(1.8rem,4vw,3rem);font-weight:900;letter-spacing:-.04em;line-height:1.1;background:linear-gradient(135deg,#f1f5f9,var(--pri) 60%,var(--sec));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
    .hero p{max-width:680px;margin:14px auto 0;color:var(--t2);font-size:.95rem;line-height:1.7}
    .hero-metrics{display:flex;justify-content:center;gap:28px;margin-top:28px;flex-wrap:wrap}
    .hero-m{text-align:center}
    .hero-m .hv{font-family:'JetBrains Mono',monospace;font-size:1.3rem;font-weight:700;color:#fff}
    .hero-m .hl{font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:var(--t3);margin-top:4px}

    /* ── CARDS ── */
    .card{background:var(--card);border:1px solid var(--card-b);border-radius:var(--r3);padding:24px;box-shadow:var(--sh)}
    .card h2{font-size:1.15rem;font-weight:800;letter-spacing:-.02em;margin-bottom:4px}
    .card .sub{color:var(--t2);font-size:.85rem;margin-bottom:16px}

    /* ── CONFIG SECTION ── */
    .config-grid{display:grid;grid-template-columns:1fr 100px 1fr 1fr;gap:12px;align-items:end}
    .field{display:flex;flex-direction:column;gap:5px}
    .field label{font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--t3)}
    .field select,.field input{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:var(--r2);padding:10px 12px;color:var(--t);font-size:.88rem;font-family:inherit;outline:none;transition:border .2s,box-shadow .2s;width:100%}
    .field select:focus,.field input:focus{border-color:var(--sec);box-shadow:0 0 0 2px rgba(129,140,248,.15)}
    .field select{cursor:pointer}
    .field select option{background:var(--bg2)}
    .field input:disabled,.field select:disabled{opacity:.35;cursor:not-allowed}

    .config-row2{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;margin-top:12px}
    .mode-hint{grid-column:1/-1;font-size:.8rem;color:var(--t3);padding:10px 14px;background:rgba(129,140,248,.05);border:1px solid rgba(129,140,248,.08);border-radius:var(--r2);line-height:1.5}

    .run-row{display:flex;gap:12px;margin-top:16px;align-items:center}
    .btn{border:none;border-radius:var(--r2);cursor:pointer;font-family:inherit;font-weight:700;font-size:.9rem;padding:12px 28px;transition:transform .15s,box-shadow .15s,opacity .15s;display:inline-flex;align-items:center;gap:8px}
    .btn-go{background:linear-gradient(135deg,var(--pri),var(--sec));color:#fff;box-shadow:0 4px 16px rgba(20,184,166,.25)}
    .btn-go:hover{transform:translateY(-1px);box-shadow:0 6px 24px rgba(20,184,166,.3)}
    .btn-go:disabled{opacity:.45;cursor:not-allowed;transform:none}
    .spinner{width:16px;height:16px;border:2px solid rgba(255,255,255,.2);border-top-color:#fff;border-radius:50%;animation:spin .6s linear infinite;display:none}
    .spinner.on{display:inline-block}
    @keyframes spin{to{transform:rotate(360deg)}}

    /* ── METRICS ROW ── */
    .metrics-row{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:20px}
    .m-card{background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.05);border-radius:var(--r);padding:16px;text-align:center;transition:border-color .2s}
    .m-card:hover{border-color:rgba(129,140,248,.15)}
    .m-label{font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--t3)}
    .m-val{font-family:'JetBrains Mono',monospace;font-size:1.6rem;font-weight:700;color:#fff;margin-top:8px;transition:color .3s}
    .m-val.ok{color:var(--ok)}.m-val.err{color:var(--err)}

    /* ── TABS ── */
    .tabs{display:flex;gap:4px;margin-top:20px;background:rgba(255,255,255,.03);border-radius:var(--r2);padding:4px;border:1px solid rgba(255,255,255,.05)}
    .tab-btn{flex:1;padding:10px;font-size:.82rem;font-weight:600;border:none;background:transparent;color:var(--t3);cursor:pointer;border-radius:8px;transition:all .2s;font-family:inherit}
    .tab-btn.active{background:rgba(129,140,248,.12);color:var(--sec)}
    .tab-btn:hover:not(.active){background:rgba(255,255,255,.03)}
    .tab-panel{display:none;margin-top:16px}
    .tab-panel.active{display:block}

    /* ── TIMELINE ── */
    .tl-item{display:grid;grid-template-columns:36px 1fr;gap:0;margin-bottom:2px}
    .tl-dot-col{display:flex;flex-direction:column;align-items:center}
    .tl-dot{width:10px;height:10px;border-radius:50%;background:var(--pri);margin-top:6px;flex-shrink:0;box-shadow:0 0 0 3px rgba(20,184,166,.15)}
    .tl-dot.fail{background:var(--err);box-shadow:0 0 0 3px rgba(251,113,133,.15)}
    .tl-line{width:2px;flex:1;background:rgba(255,255,255,.06);margin-top:4px}
    .tl-item:last-child .tl-line{display:none}
    .tl-body{padding:0 0 16px}
    .tl-head{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
    .tl-step{font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--t3)}
    .tl-act{font-family:'JetBrains Mono',monospace;font-size:.85rem;font-weight:600;color:var(--t)}
    .tl-rw{font-family:'JetBrains Mono',monospace;font-size:.72rem;font-weight:600;padding:2px 7px;border-radius:5px}
    .tl-rw.pos{background:rgba(52,211,153,.1);color:var(--ok);border:1px solid rgba(52,211,153,.15)}
    .tl-rw.zero{background:rgba(251,113,133,.1);color:var(--err);border:1px solid rgba(251,113,133,.15)}
    .tl-ctx{font-size:.82rem;color:var(--t2);line-height:1.5;margin-top:4px}

    /* ── SUMMARY GRID ── */
    .sg{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:8px}
    .sg-item{padding:10px 12px;border-radius:var(--r2);background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.04)}
    .sg-k{font-size:.65rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--t3);margin-bottom:4px}
    .sg-v{font-family:'JetBrains Mono',monospace;font-size:.82rem;font-weight:600;color:var(--t);word-break:break-all}

    /* ── CONTEXT BOX ── */
    .ctx-box{background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.05);border-radius:var(--r2);padding:14px;margin-top:12px}
    .ctx-box h3{font-size:.85rem;font-weight:700;margin-bottom:8px}
    .ctx-box pre{font-family:'JetBrains Mono',monospace;font-size:.8rem;line-height:1.6;color:var(--t2);white-space:pre-wrap;word-break:break-word;margin:0}

    /* ── TASK CATALOG ── */
    .task-row{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px;margin-top:16px}
    .t-card{padding:16px;border-radius:var(--r);background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.05);transition:border-color .2s}
    .t-card:hover{border-color:rgba(129,140,248,.15)}
    .t-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
    .t-name{font-family:'JetBrains Mono',monospace;font-size:.88rem;font-weight:700}
    .pill{font-size:.65rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;padding:3px 8px;border-radius:999px}
    .pill-easy{background:rgba(52,211,153,.1);color:var(--ok);border:1px solid rgba(52,211,153,.18)}
    .pill-medium{background:rgba(245,158,11,.1);color:var(--acc);border:1px solid rgba(245,158,11,.18)}
    .pill-hard{background:rgba(251,113,133,.1);color:var(--err);border:1px solid rgba(251,113,133,.18)}
    .t-meta{font-size:.75rem;color:var(--t3)}
    .t-desc{font-size:.8rem;color:var(--t2);line-height:1.5;margin-top:6px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}

    /* ── FOOTER ── */
    .footer{text-align:center;padding:24px 0 0;color:var(--t3);font-size:.78rem;line-height:1.6}
    .footer code{padding:2px 5px;border-radius:4px;background:rgba(255,255,255,.04);font-family:'JetBrains Mono',monospace;font-size:.72rem}

    .mt12{margin-top:12px}.mt20{margin-top:20px}
    .hidden{display:none!important}

    @keyframes fadeIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
    .fade{animation:fadeIn .4s ease both}
    .fd1{animation-delay:.05s}.fd2{animation-delay:.1s}.fd3{animation-delay:.15s}

    @media(max-width:768px){
      .config-grid{grid-template-columns:1fr 1fr}
      .config-row2{grid-template-columns:1fr 1fr}
      .metrics-row{grid-template-columns:1fr 1fr}
      .task-row{grid-template-columns:1fr}
      .hero-metrics{gap:16px}
    }
    @media(max-width:480px){
      .config-grid{grid-template-columns:1fr}
      .config-row2{grid-template-columns:1fr}
      .app{padding:12px 12px 32px}
    }
  </style>
</head>
<body>
<div class="app">

  <!-- NAV -->
  <nav class="nav fade">
    <div class="logo">🧭 CreditMaze</div>
    <span class="badge">Meta RL Hackathon</span>
    <div class="spacer"></div>
    <a href="/tasks" class="link">API</a>
    <a href="/health" class="link">Health</a>
  </nav>

  <!-- HERO -->
  <section class="hero fade fd1">
    <h1>Can your agent tell which decision actually mattered?</h1>
    <p>A benchmark for causal decision attribution in long-horizon AI&nbsp;workflows. Measures whether agents identify the pivotal step — not just whether they succeed.</p>
    <div class="hero-metrics">
      <div class="hero-m"><div class="hv" id="heroTaskCount">5</div><div class="hl">Tasks</div></div>
      <div class="hero-m"><div class="hv">PSIA</div><div class="hl">Step ID Accuracy</div></div>
      <div class="hero-m"><div class="hv">CCE</div><div class="hl">Calibration Error</div></div>
      <div class="hero-m"><div class="hv">MPCS</div><div class="hl">Multi-Pivot Score</div></div>
      <div class="hero-m"><div class="hv">TSR</div><div class="hl">Task Success Rate</div></div>
    </div>
  </section>

  <!-- EVALUATION CARD -->
  <section class="card fade fd2">
    <h2>Run Evaluation</h2>
    <div class="sub">Configure and launch a benchmark episode. The agent plays to completion automatically.</div>

    <div class="config-grid">
      <div class="field"><label for="taskSelect">Task</label><select id="taskSelect"></select></div>
      <div class="field"><label for="seedInput">Seed</label><input id="seedInput" type="number" value="42"></div>
      <div class="field"><label for="runMode">Agent Mode</label>
        <select id="runMode">
          <option value="auto">Auto (LLM → random fallback)</option>
          <option value="random">Random baseline</option>
          <option value="llm">LLM only</option>
        </select>
      </div>
      <div class="field"><label for="modelInput">Model</label><input id="modelInput" type="text" placeholder="e.g. gpt-4o-mini"></div>
    </div>
    <div class="config-row2">
      <div class="field"><label for="baseUrlInput">API Base URL</label><input id="baseUrlInput" type="text" placeholder="https://router.huggingface.co/v1"></div>
      <div class="field"><label for="apiKeyInput">API Key</label><input id="apiKeyInput" type="password" placeholder="Temporary session key"></div>
      <div class="mode-hint" id="modeNote" style="grid-column:span 2">Auto mode uses the configured LLM when available and falls back to a random baseline if the call fails.</div>
    </div>
    <div class="run-row">
      <button class="btn btn-go" id="autoRunBtn"><span class="spinner" id="runSpinner"></span><span id="runBtnText">Run Agent Evaluation</span></button>
    </div>

    <!-- LIVE METRICS -->
    <div class="metrics-row">
      <div class="m-card"><div class="m-label">Outcome</div><div class="m-val" id="metricOutcome">—</div></div>
      <div class="m-card"><div class="m-label">Reward</div><div class="m-val" id="metricReward">0.00</div></div>
      <div class="m-card"><div class="m-label">PSIA</div><div class="m-val" id="metricPsia">0.00</div></div>
      <div class="m-card"><div class="m-label">CCE</div><div class="m-val" id="metricCce">0.50</div></div>
    </div>

    <!-- TABBED RESULTS -->
    <div class="tabs">
      <button class="tab-btn active" data-tab="tabTimeline">Timeline</button>
      <button class="tab-btn" data-tab="tabSummary">Summary</button>
      <button class="tab-btn" data-tab="tabAttribution">Attribution</button>
      <button class="tab-btn" data-tab="tabContext">Context</button>
    </div>

    <div class="tab-panel active" id="tabTimeline">
      <div id="timeline"><p class="sub">Run an evaluation to see the step-by-step timeline.</p></div>
    </div>
    <div class="tab-panel" id="tabSummary">
      <div class="sg" id="runModeText"></div>
      <div class="sg mt12" id="statusText"></div>
    </div>
    <div class="tab-panel" id="tabAttribution">
      <div class="sg" id="graderText"></div>
    </div>
    <div class="tab-panel" id="tabContext">
      <div class="ctx-box"><h3>Episode Snapshot</h3><pre id="contextText">No episode data yet. Run an evaluation to see context.</pre></div>
    </div>
  </section>

  <!-- TASK CATALOG -->
  <section class="card fade fd3 mt20">
    <h2>Task Catalog</h2>
    <div class="sub">The benchmark tasks exposed by this Space for evaluators.</div>
    <div class="task-row" id="taskList"></div>
  </section>

  <div class="footer">
    <code>/reset</code> · <code>/step</code> · <code>/state</code> · <code>/tasks</code> · <code>/grader</code> · <code>/health</code>
    <br>Interactive UI on top of the same evaluation API used by agents.
  </div>
</div>

<script>
const state={tasks:[],episodeId:null,done:false,timeline:[]};
const $=id=>document.getElementById(id);
const els={
  taskSelect:$('taskSelect'),seedInput:$('seedInput'),runMode:$('runMode'),
  modelInput:$('modelInput'),baseUrlInput:$('baseUrlInput'),apiKeyInput:$('apiKeyInput'),
  modeNote:$('modeNote'),autoRunBtn:$('autoRunBtn'),runSpinner:$('runSpinner'),runBtnText:$('runBtnText'),
  metricOutcome:$('metricOutcome'),metricReward:$('metricReward'),
  metricPsia:$('metricPsia'),metricCce:$('metricCce'),
  runModeText:$('runModeText'),statusText:$('statusText'),graderText:$('graderText'),
  taskList:$('taskList'),timeline:$('timeline'),contextText:$('contextText'),
  heroTaskCount:$('heroTaskCount'),
};

// tabs
document.querySelectorAll('.tab-btn').forEach(btn=>{
  btn.addEventListener('click',()=>{
    document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.tab).classList.add('active');
  });
});

async function api(path,opts={}){
  const r=await fetch(path,{headers:{'Content-Type':'application/json'},...opts});
  if(!r.ok){const t=await r.text();throw new Error(t||'Request failed: '+r.status)}
  return r.json();
}
function sg(entries){
  if(!entries||!entries.length)return'<p class="sub">Nothing yet.</p>';
  return entries.map(([k,v])=>'<div class="sg-item"><div class="sg-k">'+k+'</div><div class="sg-v">'+(v??'—')+'</div></div>').join('');
}
function pillClass(d){
  if(!d)return'pill-easy';const l=d.toLowerCase();
  return l==='easy'?'pill-easy':l==='medium'?'pill-medium':'pill-hard';
}
function renderCatalog(){
  els.taskList.innerHTML='';
  els.heroTaskCount.textContent=String(state.tasks.length||0);
  state.tasks.forEach(t=>{
    const d=document.createElement('div');d.className='t-card';
    d.innerHTML='<div class="t-top"><span class="t-name">'+t.id+'</span><span class="pill '+pillClass(t.difficulty)+'">'+t.difficulty+'</span></div>'
      +'<div class="t-meta">max_steps = '+t.max_steps+'</div>'
      +'<div class="t-desc">'+(t.grader?.prompt_template||'')+'</div>';
    els.taskList.appendChild(d);
  });
}
function syncBtns(busy){
  els.autoRunBtn.disabled=busy;
  els.runSpinner.className=busy?'spinner on':'spinner';
  els.runBtnText.textContent=busy?'Running…':'Run Agent Evaluation';
}
function renderRunSummary(p){
  els.runModeText.innerHTML=sg([
    ['Requested Mode',els.runMode.value],['Mode Used',p?.mode_used||'—'],
    ['Model',p?.model_label||'random-baseline'],['Fallback',p?String(!!p.used_fallback):'—'],
    ['Fallback Reason',p?.fallback_reason||'none'],['Score',p?.score!=null?Number(p.score).toFixed(3):'—'],
  ]);
}
function renderEpisodeSummary(p){
  els.statusText.innerHTML=sg([
    ['Episode ID',p?.episode_id||'—'],['Task',p?.task_id||els.taskSelect.value||'—'],
    ['Outcome',p?.outcome||'—'],['Raw Reward',p?.raw_reward!=null?Number(p.raw_reward).toFixed(3):'—'],
    ['Steps',p?.steps?p.steps.length:'—'],
  ]);
}
function renderGrader(g){
  if(!g){els.graderText.innerHTML='<p class="sub">Run an evaluation to see attribution diagnostics.</p>';return}
  els.graderText.innerHTML=sg([
    ['Top Step',g.top_attributed_step??'—'],['Top Action',g.top_attributed_action??'—'],
    ['Pivotal Rank',g.pivotal_step_rank??'—'],['Attribution Gap',g.attribution_gap??'—'],
    ['False Positives',g.false_positive_steps?.length?g.false_positive_steps.join(', '):'none'],
    ['Wrong Attribution',String(!!g.success_with_wrong_attribution)],
  ]);
}
function updateMode(){
  const m=els.runMode.value,r=m==='random';
  els.modelInput.disabled=r;els.baseUrlInput.disabled=r;els.apiKeyInput.disabled=r;
  els.modeNote.textContent=r?'Random baseline mode ignores model settings.':m==='llm'?'LLM-only mode requires a valid API key. Fails instead of falling back.':'Auto mode uses the configured LLM when available and falls back to a random baseline if the call fails.';
}
function renderTimeline(){
  if(!state.timeline.length){els.timeline.innerHTML='<p class="sub">Run an evaluation to see the step-by-step timeline.</p>';return}
  els.timeline.innerHTML='';
  state.timeline.forEach(it=>{
    const fail=it.done&&it.reward===0;
    const d=document.createElement('div');d.className='tl-item';
    d.innerHTML='<div class="tl-dot-col"><div class="tl-dot'+(fail?' fail':'')+'"></div><div class="tl-line"></div></div>'
      +'<div class="tl-body"><div class="tl-head"><span class="tl-step">Step '+it.step+'</span>'
      +'<span class="tl-act">'+it.action+'</span>'
      +'<span class="tl-rw '+(it.reward>0?'pos':'zero')+'">'+it.reward.toFixed(2)+'</span></div>'
      +'<div class="tl-ctx">'+it.contextSnippet+'</div></div>';
    els.timeline.appendChild(d);
  });
}
async function loadTasks(){
  const p=await api('/tasks');state.tasks=p.tasks||[];
  els.taskSelect.innerHTML='';
  state.tasks.forEach(t=>{const o=document.createElement('option');o.value=t.id;o.textContent=t.id;els.taskSelect.appendChild(o)});
  renderCatalog();
}
async function runEval(){
  syncBtns(true);
  try{
    const p=await api('/demo/run',{method:'POST',body:JSON.stringify({
      task_id:els.taskSelect.value,seed:Number(els.seedInput.value||42),mode:els.runMode.value,
      model_name:els.modelInput.value||null,api_base_url:els.baseUrlInput.value||null,api_key:els.apiKeyInput.value||null,
    })});
    state.timeline=(p.steps||[]).map(s=>({step:s.step,action:s.action,reward:s.reward,done:s.done,error:s.error,contextSnippet:s.context_snippet||''}));
    state.done=true;state.episodeId=p.episode_id;
    const o=els.metricOutcome;o.textContent=p.outcome||'—';
    o.className='m-val'+(p.outcome==='success'?' ok':p.outcome==='failure'?' err':'');
    els.metricReward.textContent=Number(p.raw_reward||0).toFixed(2);
    els.metricPsia.textContent=Number(p.session_psia||0).toFixed(2);
    els.metricCce.textContent=Number(p.session_cce||0.5).toFixed(2);
    els.contextText.textContent=p.final_context||'Episode complete.';
    renderEpisodeSummary(p);renderGrader(p.grader);renderRunSummary(p);renderTimeline();
    // auto-flip to timeline tab
    document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p2=>p2.classList.remove('active'));
    document.querySelector('[data-tab="tabTimeline"]').classList.add('active');
    document.getElementById('tabTimeline').classList.add('active');
  }catch(err){
    els.runModeText.innerHTML=sg([['Requested Mode',els.runMode.value],['Result','Failed'],['Error',err.message]]);
  }finally{syncBtns(false)}
}
els.autoRunBtn.addEventListener('click',runEval);
els.runMode.addEventListener('change',updateMode);
loadTasks().catch(err=>{els.statusText.innerHTML=sg([['Status','Failed to load tasks'],['Error',err.message]])});
updateMode();renderRunSummary(null);renderEpisodeSummary(null);renderGrader(null);
</script>
</body>
</html>
"""


def _make_demo_client(req: DemoRunRequest) -> tuple[Optional[OpenAI], str, Optional[str], str]:
    default_key = req.api_key or ""
    if not default_key:
        import os
        default_key = (
            os.getenv("API_KEY")
            or os.getenv("HF_TOKEN")
            or os.getenv("OPENAI_API_KEY")
            or ""
        )
    default_base = req.api_base_url or "https://router.huggingface.co/v1"
    default_model = req.model_name or "Qwen/Qwen2.5-72B-Instruct"

    if req.mode == "random":
        return None, "random-baseline", None, "random"
    if req.mode == "llm" and not default_key:
        raise HTTPException(400, "LLM mode requested but no API key was provided in the form or server environment.")
    if not default_key:
        return None, "random-baseline", "no_credentials", "random"
    client = OpenAI(base_url=default_base, api_key=default_key, max_retries=0)
    return client, default_model, None, "llm"


def _choose_demo_action(client: Optional[OpenAI], model_name: str, obs: dict, task_id: str) -> tuple[dict, Optional[str], bool]:
    if client is None:
        return ({
            "action_id": random.choice(obs["available_actions"]),
            "reasoning": "Random baseline",
            "credit_estimate": 0.5,
        }, "random_fallback:no_credentials", True)

    prompt = (
        f"Task: {task_id}\n"
        f"Domain: {obs['domain']}\n"
        f"Tier: {obs['tier']}\n"
        f"Step: {obs['step_count'] + 1}/{obs['t_total']}\n"
        f"Context:\n{obs['context']}\n\n"
        f"Available actions: {obs['available_actions']}"
    )
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": UI_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            stream=False,
        )
        content = (completion.choices[0].message.content or "").strip()
        parsed = json.loads(content)
        credit = float(parsed.get("credit_estimate", 0.5))
        parsed["credit_estimate"] = min(max(credit, 0.0), 1.0)
        return parsed, None, False
    except Exception as exc:
        return ({
            "action_id": random.choice(obs["available_actions"]),
            "reasoning": "Random fallback",
            "credit_estimate": 0.5,
        }, f"random_fallback:model_call_failed:{type(exc).__name__}", True)


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


class GraderRequest(BaseModel):
    episode_id: str


class DemoRunRequest(BaseModel):
    task_id: str
    seed: Optional[int] = 42
    mode: str = "auto"
    api_base_url: Optional[str] = None
    model_name: Optional[str] = None
    api_key: Optional[str] = None


DemoRunRequest.model_rebuild()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    req = req or ResetRequest()
    valid_tiers = ["easy", "medium", "hard", "multi-pivot"]
    if req.tier not in valid_tiers:
        raise HTTPException(400, f"tier must be one of {valid_tiers}")
    obs = env.reset(tier=req.tier, domain=req.domain, seed=req.seed)
    return obs.model_dump()


@app.post("/step")
def step(req: StepRequest):
    try:
        action = Action(
            action_id=req.action_id,
            reasoning=req.reasoning,
            credit_estimate=req.credit_estimate,
        )
        result = env.step(req.episode_id, action)
        return result.model_dump()
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
            "type": "llm",
            "prompt_template": (
                f"Score the agent's performance on {task_id} from 0.01 to 0.99 "
                f"based on whether it completed the task correctly, identified the "
                f"decisive step(s), and avoided over-crediting decoys. {summary}"
            ),
        }

    return {
        "tasks": [
            {"id": "task_easy",   "difficulty": "easy",   "max_steps": 15, "grader": grader_meta("task_easy", "This task maps to the environment's canonical easy benchmark episode.")},
            {"id": "task_medium", "difficulty": "medium", "max_steps": 15, "grader": grader_meta("task_medium", "This task maps to the environment's canonical medium benchmark episode.")},
            {"id": "task_hard",   "difficulty": "hard",   "max_steps": 15, "grader": grader_meta("task_hard", "This task maps to the environment's canonical hard benchmark episode.")},
            {"id": "resource_hard", "difficulty": "hard", "max_steps": 15, "grader": grader_meta("resource_hard", "This task maps to the environment's resource allocation benchmark episode.")},
            {"id": "triage_multipivot", "difficulty": "hard", "max_steps": 15, "grader": grader_meta("triage_multipivot", "This task maps to the environment's multi-pivot triage benchmark episode.")},
        ],
    }


@app.post("/grader")
def grader(req: GraderRequest):
    try:
        s = env.state(req.episode_id)
    except KeyError as e:
        raise HTTPException(404, str(e))

    if not s.episode_complete:
        raise HTTPException(400, "Episode not complete. Keep calling /step until done=True.")

    return {
        "episode_id":           req.episode_id,
        "outcome":              s.outcome,
        "score":                env.normalized_score(req.episode_id),
        "raw_reward":           round(s.cumulative_reward, 4),
        "session_psia":         s.session_psia,
        "session_cce":          s.session_cce,
        "session_tsr":          s.session_tsr,
        "session_mpcs":         s.session_mpcs,
        "episodes_completed":   s.episodes_completed,
        "causal_faithfulness":  s.causal_faithfulness,
        "pivotal_step_indices": s.pivotal_step_indices,
        "top_attributed_step":  s.top_attributed_step,
        "top_attributed_action": s.top_attributed_action,
        "top_attributed_credit": s.top_attributed_credit,
        "pivotal_step_rank":    s.pivotal_step_rank,
        "false_positive_steps": s.false_positive_steps,
        "attribution_gap":      s.attribution_gap,
        "success_with_wrong_attribution": s.success_with_wrong_attribution,
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
        raise HTTPException(500, "Baseline script did not produce valid JSON")


@app.post("/demo/run")
def demo_run(req: DemoRunRequest):
    if req.task_id not in TASK_CONFIG:
        raise HTTPException(400, f"Unknown task_id '{req.task_id}'")

    client, model_label, startup_reason, mode_used = _make_demo_client(req)
    cfg = TASK_CONFIG[req.task_id]
    obs = env.reset(tier=cfg["tier"], domain=cfg["domain"], seed=req.seed)
    episode_id = obs.episode_id
    steps = []
    used_fallback = startup_reason is not None
    fallback_reason = startup_reason

    while True:
        obs_dict = obs.model_dump()
        decision, step_error, step_fallback = _choose_demo_action(client, model_label, obs_dict, req.task_id)
        action_id = decision.get("action_id", obs.available_actions[0])
        if action_id not in obs.available_actions:
            action_id = obs.available_actions[0]
        if step_fallback:
            used_fallback = True
            fallback_reason = step_error
            mode_used = "random-fallback"
        result = env.step(
            episode_id,
            Action(
                action_id=action_id,
                reasoning=decision.get("reasoning"),
                credit_estimate=float(decision.get("credit_estimate", 0.5)),
            ),
        )
        steps.append({
            "step": result.observation.step_count,
            "action": action_id,
            "reward": result.reward,
            "done": result.done,
            "error": step_error,
            "context_snippet": result.observation.context[:220],
        })
        obs = result.observation
        if result.done:
            break

    grader_payload = grader(GraderRequest(episode_id=episode_id))
    return {
        "task_id": req.task_id,
        "episode_id": episode_id,
        "mode_used": mode_used,
        "model_label": model_label,
        "used_fallback": used_fallback,
        "fallback_reason": fallback_reason,
        "outcome": grader_payload["outcome"],
        "score": grader_payload["score"],
        "raw_reward": grader_payload["raw_reward"],
        "session_psia": grader_payload["session_psia"],
        "session_cce": grader_payload["session_cce"],
        "final_context": obs.context,
        "steps": steps,
        "grader": grader_payload,
    }


@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    accept = request.headers.get("accept", "")
    user_agent = request.headers.get("user-agent", "").lower()
    if "text/html" in accept or "mozilla" in user_agent:
        return HTMLResponse(_homepage_html())
    return JSONResponse({
        "name": "CreditMaze",
        "status": "ok",
        "message": "CreditMaze is running.",
        "endpoints": [
            "/health",
            "/tasks",
            "/reset",
            "/step",
            "/state",
            "/grader",
            "/baseline",
        ],
    })


@app.get("/ui", response_class=HTMLResponse)
def ui():
    return HTMLResponse(_homepage_html())


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0", "environment": "CreditMaze"}


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
