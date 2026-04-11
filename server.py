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
  <title>CreditMaze</title>
  <style>
    :root {
      --bg: #f4efe3;
      --paper: #fffaf0;
      --panel: rgba(255, 250, 240, 0.84);
      --ink: #172238;
      --muted: #596579;
      --line: rgba(23, 34, 56, 0.12);
      --accent: #0f766e;
      --accent-2: #d97706;
      --accent-3: #1d4ed8;
      --success: #166534;
      --danger: #b91c1c;
      --shadow: 0 26px 60px rgba(23, 34, 56, 0.12);
      --radius-xl: 28px;
      --radius-lg: 20px;
      --radius-md: 14px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      font-family: "Georgia", "Palatino Linotype", serif;
      background:
        radial-gradient(circle at top left, rgba(29, 78, 216, 0.14), transparent 32%),
        radial-gradient(circle at top right, rgba(217, 119, 6, 0.18), transparent 30%),
        linear-gradient(180deg, #f7f2e8 0%, #efe6d7 100%);
      min-height: 100vh;
    }
    .shell {
      width: min(1220px, calc(100vw - 32px));
      margin: 24px auto 40px;
      display: grid;
      gap: 18px;
    }
    .hero {
      position: relative;
      overflow: hidden;
      border-radius: var(--radius-xl);
      background:
        linear-gradient(135deg, rgba(15,118,110,0.94), rgba(29,78,216,0.92) 58%, rgba(217,119,6,0.86));
      color: #f8fafc;
      padding: 34px 34px 30px;
      box-shadow: var(--shadow);
      isolation: isolate;
    }
    .hero::after {
      content: "";
      position: absolute;
      inset: auto -12% -28% 38%;
      height: 280px;
      background: radial-gradient(circle, rgba(255,255,255,0.18), transparent 62%);
      transform: rotate(-8deg);
      z-index: -1;
    }
    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 14px;
      border-radius: 999px;
      background: rgba(255,255,255,0.16);
      font: 600 13px/1 "Trebuchet MS", "Segoe UI", sans-serif;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }
    h1 {
      margin: 18px 0 12px;
      font-size: clamp(2.3rem, 5vw, 4.2rem);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }
    .hero p {
      max-width: 760px;
      margin: 0;
      font-size: 1.03rem;
      line-height: 1.65;
      color: rgba(248, 250, 252, 0.92);
    }
    .hero-grid {
      margin-top: 24px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
    }
    .hero-stat {
      border-radius: 18px;
      padding: 16px 18px;
      background: rgba(255,255,255,0.12);
      border: 1px solid rgba(255,255,255,0.16);
      backdrop-filter: blur(10px);
    }
    .hero-stat strong {
      display: block;
      font-size: 1.45rem;
      margin-bottom: 4px;
    }
    .layout {
      display: grid;
      grid-template-columns: 1.18fr 0.82fr;
      gap: 18px;
    }
    .card {
      border-radius: var(--radius-xl);
      background: var(--panel);
      border: 1px solid rgba(255,255,255,0.42);
      box-shadow: var(--shadow);
      backdrop-filter: blur(16px);
      padding: 24px;
    }
    .card h2, .card h3 {
      margin: 0 0 10px;
      line-height: 1.1;
    }
    .card h2 { font-size: 1.7rem; }
    .card h3 { font-size: 1.18rem; }
    .muted, .card p, .card li, label, .hint {
      color: var(--muted);
      font-family: "Trebuchet MS", "Segoe UI", sans-serif;
    }
    .stack { display: grid; gap: 14px; }
    .metrics {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
    }
    .metric {
      padding: 16px;
      border-radius: 18px;
      background: rgba(255,255,255,0.62);
      border: 1px solid var(--line);
    }
    .metric .label {
      font: 700 12px/1 "Trebuchet MS", "Segoe UI", sans-serif;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .metric .value {
      margin-top: 10px;
      font-size: 1.48rem;
      font-weight: 700;
    }
    .task-list {
      display: grid;
      gap: 12px;
      margin-top: 16px;
    }
    .task {
      padding: 16px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.56);
    }
    .task-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 8px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border-radius: 999px;
      padding: 6px 10px;
      font: 700 12px/1 "Trebuchet MS", "Segoe UI", sans-serif;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: #fff;
      background: var(--accent);
    }
    .pill.alt { background: var(--accent-2); }
    .pill.blue { background: var(--accent-3); }
    .controls {
      display: grid;
      gap: 14px;
      margin-top: 18px;
    }
    .field {
      display: grid;
      gap: 8px;
    }
    .field-row {
      display: grid;
      grid-template-columns: 1fr 130px;
      gap: 12px;
    }
    select, input, textarea, button {
      width: 100%;
      border: 1px solid rgba(23,34,56,0.12);
      border-radius: 14px;
      background: rgba(255,255,255,0.86);
      color: var(--ink);
      padding: 13px 14px;
      font-size: 0.98rem;
      font-family: "Trebuchet MS", "Segoe UI", sans-serif;
    }
    textarea {
      min-height: 90px;
      resize: vertical;
    }
    button {
      cursor: pointer;
      font-weight: 700;
      background: linear-gradient(135deg, var(--accent), var(--accent-3));
      color: #fff;
      border: none;
      box-shadow: 0 16px 30px rgba(15,118,110,0.18);
      transition: transform .18s ease, box-shadow .18s ease, opacity .18s ease;
    }
    button.secondary {
      background: rgba(255,255,255,0.72);
      color: var(--ink);
      border: 1px solid var(--line);
      box-shadow: none;
    }
    button:hover { transform: translateY(-1px); }
    button:disabled { opacity: 0.58; cursor: not-allowed; transform: none; }
    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 14px;
    }
    .action-chip {
      border-radius: 999px;
      padding: 10px 14px;
      border: 1px solid rgba(29,78,216,0.16);
      background: rgba(29,78,216,0.08);
      color: var(--ink);
      font: 700 14px/1 "Trebuchet MS", "Segoe UI", sans-serif;
      cursor: pointer;
      transition: all .18s ease;
    }
    .action-chip.active {
      background: linear-gradient(135deg, rgba(15,118,110,0.92), rgba(29,78,216,0.92));
      color: #fff;
      border-color: transparent;
      box-shadow: 0 12px 22px rgba(29,78,216,0.18);
    }
    .context-panel, .timeline, .json-panel {
      border-radius: 18px;
      padding: 18px;
      background: rgba(255,255,255,0.62);
      border: 1px solid var(--line);
    }
    .timeline-item {
      position: relative;
      padding: 0 0 14px 18px;
      margin-left: 4px;
      border-left: 2px solid rgba(23,34,56,0.12);
    }
    .timeline-item:last-child {
      border-left-color: transparent;
      padding-bottom: 0;
    }
    .timeline-item::before {
      content: "";
      position: absolute;
      left: -7px;
      top: 4px;
      width: 12px;
      height: 12px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 0 4px rgba(15,118,110,0.12);
    }
    .timeline-item.fail::before {
      background: var(--danger);
      box-shadow: 0 0 0 4px rgba(185,28,28,0.12);
    }
    .timeline-meta {
      font: 700 12px/1 "Trebuchet MS", "Segoe UI", sans-serif;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 6px;
    }
    .status-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 12px;
    }
    code, pre {
      font-family: Consolas, "SFMono-Regular", monospace;
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      color: var(--ink);
      font-size: 0.92rem;
      line-height: 1.5;
    }
    .footer {
      padding: 0 4px;
      color: var(--muted);
      font: 500 13px/1.6 "Trebuchet MS", "Segoe UI", sans-serif;
    }
    @media (max-width: 980px) {
      .layout { grid-template-columns: 1fr; }
      .metrics { grid-template-columns: repeat(2, 1fr); }
      .field-row { grid-template-columns: 1fr; }
    }
    @media (max-width: 640px) {
      .shell { width: min(100vw - 18px, 1220px); margin-top: 12px; }
      .hero, .card { padding: 20px; border-radius: 22px; }
      .metrics { grid-template-columns: 1fr 1fr; }
      .status-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="eyebrow">CreditMaze Demo Surface</div>
      <h1>Can an AI agent tell which decision actually mattered?</h1>
      <p>
        CreditMaze is a long-horizon benchmark for causal decision attribution. It does not just
        ask whether an agent reaches the right outcome. It asks whether the agent identifies the
        step that truly caused success or failure in workflows like research synthesis, debugging,
        resource allocation, and triage.
      </p>
      <div class="hero-grid">
        <div class="hero-stat"><strong id="heroTaskCount">5</strong><span>Evaluator-facing tasks</span></div>
        <div class="hero-stat"><strong>PSIA</strong><span>Did the agent credit the right step?</span></div>
        <div class="hero-stat"><strong>CCE</strong><span>How calibrated was its attribution?</span></div>
        <div class="hero-stat"><strong>MPCS</strong><span>Can it find multiple jointly causal steps?</span></div>
      </div>
    </section>

    <div class="layout">
      <section class="card">
        <h2>Interactive Episode Lab</h2>
        <p class="muted">
          Launch a benchmark episode, inspect the live context, choose actions, and see how the
          environment distinguishes local progress from the truly pivotal step.
        </p>
        <div class="controls">
          <div class="field-row">
            <div class="field">
              <label for="taskSelect">Task</label>
              <select id="taskSelect"></select>
            </div>
            <div class="field">
              <label for="seedInput">Seed</label>
              <input id="seedInput" type="number" value="42">
            </div>
          </div>
          <div class="field-row">
            <div class="field">
              <label for="runMode">Agent Mode</label>
              <select id="runMode">
                <option value="auto">LLM if configured, else random fallback</option>
                <option value="llm">Force LLM</option>
                <option value="random">Random baseline</option>
              </select>
            </div>
            <div class="field">
              <label for="modelInput">Model Name</label>
              <input id="modelInput" type="text" placeholder="Optional override, e.g. gpt-4o-mini">
            </div>
          </div>
          <div class="field-row">
            <div class="field">
              <label for="baseUrlInput">API Base URL</label>
              <input id="baseUrlInput" type="text" placeholder="Optional override, e.g. https://router.huggingface.co/v1">
            </div>
            <div class="field">
              <label for="apiKeyInput">API Key</label>
              <input id="apiKeyInput" type="password" placeholder="Optional temporary key for this browser session">
            </div>
          </div>
          <div class="hint">Use server-side environment variables, or paste temporary values here to test an LLM path. If the model call fails and mode is <code>auto</code>, the demo falls back to random actions.</div>
          <div class="field-row">
            <button id="resetBtn">Start Episode</button>
            <button id="graderBtn" class="secondary" disabled>Run Grader</button>
          </div>
          <button id="autoRunBtn" class="secondary">Run Agent Automatically</button>
          <div class="context-panel">
            <h3>Current Context</h3>
            <pre id="contextText">Start an episode to load the benchmark state.</pre>
          </div>
          <div class="field">
            <label for="reasoningInput">Reasoning Note</label>
            <textarea id="reasoningInput" placeholder="Optional note for this step."></textarea>
          </div>
          <div class="field">
            <label for="creditInput">Credit Estimate</label>
            <input id="creditInput" type="number" min="0" max="1" step="0.01" value="0.50">
          </div>
          <div>
            <div class="hint">Available Actions</div>
            <div id="actionChips" class="actions"></div>
          </div>
          <button id="stepBtn" disabled>Submit Step</button>
        </div>
      </section>

      <section class="stack">
        <section class="card">
          <h2>Live Scoreboard</h2>
          <div class="metrics">
            <div class="metric"><div class="label">Outcome</div><div class="value" id="metricOutcome">-</div></div>
            <div class="metric"><div class="label">Reward</div><div class="value" id="metricReward">0.00</div></div>
            <div class="metric"><div class="label">PSIA</div><div class="value" id="metricPsia">0.00</div></div>
            <div class="metric"><div class="label">CCE</div><div class="value" id="metricCce">0.50</div></div>
          </div>
          <div class="json-panel" style="margin-top:14px;">
            <h3>Execution Mode</h3>
            <pre id="runModeText">Manual interaction. Use "Run Agent Automatically" to test an LLM or random fallback.</pre>
          </div>
          <div class="status-grid" style="margin-top:14px;">
            <div class="json-panel">
              <h3>Episode Status</h3>
              <pre id="statusText">No episode loaded.</pre>
            </div>
            <div class="json-panel">
              <h3>Attribution Diagnostics</h3>
              <pre id="graderText">Run the grader after the episode completes to inspect attribution quality.</pre>
            </div>
          </div>
        </section>

        <section class="card">
          <h2>Task Catalog</h2>
          <p class="muted">
            These are the evaluator-facing benchmark tasks exposed by the Space.
          </p>
          <div id="taskList" class="task-list"></div>
        </section>

        <section class="card">
          <h2>Episode Timeline</h2>
          <div id="timeline" class="timeline">
            <p class="muted">No steps yet.</p>
          </div>
        </section>
      </section>
    </div>

    <div class="footer">
      CreditMaze keeps the benchmark API available at <code>/reset</code>, <code>/step</code>,
      <code>/state</code>, <code>/tasks</code>, <code>/grader</code>, and <code>/health</code>.
      This UI is an interactive layer on top of the same evaluation surface used by agents.
    </div>
  </div>

  <script>
    const state = {
      tasks: [],
      episodeId: null,
      selectedAction: null,
      done: false,
      lastObservation: null,
      timeline: []
    };

    const els = {
      taskSelect: document.getElementById('taskSelect'),
      seedInput: document.getElementById('seedInput'),
      resetBtn: document.getElementById('resetBtn'),
      graderBtn: document.getElementById('graderBtn'),
      contextText: document.getElementById('contextText'),
      reasoningInput: document.getElementById('reasoningInput'),
      creditInput: document.getElementById('creditInput'),
      runMode: document.getElementById('runMode'),
      modelInput: document.getElementById('modelInput'),
      baseUrlInput: document.getElementById('baseUrlInput'),
      apiKeyInput: document.getElementById('apiKeyInput'),
      autoRunBtn: document.getElementById('autoRunBtn'),
      actionChips: document.getElementById('actionChips'),
      stepBtn: document.getElementById('stepBtn'),
      metricOutcome: document.getElementById('metricOutcome'),
      metricReward: document.getElementById('metricReward'),
      metricPsia: document.getElementById('metricPsia'),
      metricCce: document.getElementById('metricCce'),
      runModeText: document.getElementById('runModeText'),
      statusText: document.getElementById('statusText'),
      graderText: document.getElementById('graderText'),
      taskList: document.getElementById('taskList'),
      timeline: document.getElementById('timeline'),
      heroTaskCount: document.getElementById('heroTaskCount'),
    };

    async function api(path, options = {}) {
      const response = await fetch(path, {
        headers: { 'Content-Type': 'application/json' },
        ...options,
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || ('Request failed: ' + response.status));
      }
      return response.json();
    }

    function renderTaskCatalog() {
      els.taskList.innerHTML = '';
      els.heroTaskCount.textContent = String(state.tasks.length || 0);
      state.tasks.forEach((task, idx) => {
        const div = document.createElement('div');
        div.className = 'task';
        div.innerHTML = `
          <div class="task-top">
            <strong>${task.id}</strong>
            <span class="pill ${idx % 3 === 1 ? 'alt' : idx % 3 === 2 ? 'blue' : ''}">${task.difficulty}</span>
          </div>
          <div class="muted">max_steps=${task.max_steps}</div>
          <p style="margin:10px 0 0;">${task.grader?.prompt_template || 'No grader metadata available.'}</p>
        `;
        els.taskList.appendChild(div);
      });
    }

    function renderActionChips(actions = []) {
      els.actionChips.innerHTML = '';
      state.selectedAction = actions[0] || null;
      actions.forEach((action, idx) => {
        const chip = document.createElement('button');
        chip.type = 'button';
        chip.className = 'action-chip' + (idx === 0 ? ' active' : '');
        chip.textContent = action;
        chip.onclick = () => {
          state.selectedAction = action;
          [...els.actionChips.children].forEach(node => node.classList.toggle('active', node === chip));
        };
        els.actionChips.appendChild(chip);
      });
      els.stepBtn.disabled = !(state.episodeId && state.selectedAction && !state.done);
    }

    function renderObservation(obs) {
      state.lastObservation = obs;
      els.contextText.textContent = obs.context || 'No context available.';
      renderActionChips(obs.available_actions || []);
      els.metricOutcome.textContent = obs.episode_outcome || 'in_progress';
      els.statusText.textContent = JSON.stringify({
        episode_id: obs.episode_id,
        domain: obs.domain,
        tier: obs.tier,
        step_count: obs.step_count,
        max_steps: obs.max_steps,
        cumulative_reward: obs.cumulative_reward,
        episode_outcome: obs.episode_outcome,
      }, null, 2);
    }

    function syncRunButtons(busy) {
      els.resetBtn.disabled = busy;
      els.stepBtn.disabled = busy || !(state.episodeId && state.selectedAction && !state.done);
      els.autoRunBtn.disabled = busy;
    }

    function renderTimeline() {
      if (!state.timeline.length) {
        els.timeline.innerHTML = '<p class="muted">No steps yet.</p>';
        return;
      }
      els.timeline.innerHTML = '';
      state.timeline.forEach(item => {
        const div = document.createElement('div');
        div.className = 'timeline-item' + (item.done && item.reward === 0 ? ' fail' : '');
        div.innerHTML = `
          <div class="timeline-meta">Step ${item.step}</div>
          <strong>${item.action}</strong>
          <p class="muted" style="margin:8px 0 4px;">reward=${item.reward.toFixed(2)} | done=${String(item.done)} | error=${item.error || 'null'}</p>
          <p style="margin:0;">${item.contextSnippet}</p>
        `;
        els.timeline.appendChild(div);
      });
    }

    async function loadTasks() {
      const payload = await api('/tasks');
      state.tasks = payload.tasks || [];
      els.taskSelect.innerHTML = '';
      state.tasks.forEach(task => {
        const opt = document.createElement('option');
        opt.value = task.id;
        opt.textContent = task.id;
        els.taskSelect.appendChild(opt);
      });
      renderTaskCatalog();
    }

    async function startEpisode() {
      const taskId = els.taskSelect.value;
      const task = state.tasks.find(t => t.id === taskId);
      if (!task) return;

      const mappings = {
        task_easy: { tier: 'easy', domain: 'corridor' },
        task_medium: { tier: 'medium', domain: 'research' },
        task_hard: { tier: 'hard', domain: 'debugging' },
        resource_hard: { tier: 'hard', domain: 'resource' },
        triage_multipivot: { tier: 'multi-pivot', domain: 'triage' },
      };
      const cfg = mappings[taskId] || { tier: task.difficulty, domain: null };
      const obs = await api('/reset', {
        method: 'POST',
        body: JSON.stringify({ ...cfg, seed: Number(els.seedInput.value || 42) }),
      });
      state.episodeId = obs.episode_id;
      state.done = false;
      state.timeline = [];
      els.metricReward.textContent = '0.00';
      els.metricPsia.textContent = '0.00';
      els.metricCce.textContent = '0.50';
      els.graderText.textContent = 'Run the grader after the episode completes to inspect attribution quality.';
      els.reasoningInput.value = '';
      els.creditInput.value = '0.50';
      els.graderBtn.disabled = true;
      els.runModeText.textContent = 'Manual interaction. Current mode: browser-driven step selection.';
      renderObservation(obs);
      renderTimeline();
    }

    async function submitStep() {
      if (!state.episodeId || !state.selectedAction || state.done) return;
      const result = await api('/step', {
        method: 'POST',
        body: JSON.stringify({
          episode_id: state.episodeId,
          action_id: state.selectedAction,
          reasoning: els.reasoningInput.value || null,
          credit_estimate: Number(els.creditInput.value || 0.5),
        }),
      });
      const obs = result.observation;
      state.done = !!result.done;
      state.timeline.push({
        step: obs.step_count,
        action: state.selectedAction,
        reward: result.reward || 0,
        done: !!result.done,
        error: null,
        contextSnippet: (obs.context || '').slice(0, 170) + ((obs.context || '').length > 170 ? '...' : ''),
      });
      els.metricReward.textContent = Number(obs.cumulative_reward || 0).toFixed(2);
      els.metricPsia.textContent = Number(result.info?.psia_running || 0).toFixed(2);
      els.metricCce.textContent = Number(result.info?.cce_running || 0.5).toFixed(2);
      renderObservation(obs);
      renderTimeline();
      els.graderBtn.disabled = !state.done;
      els.stepBtn.disabled = state.done;
    }

    async function autoRunEpisode() {
      syncRunButtons(true);
      try {
        const payload = await api('/demo/run', {
          method: 'POST',
          body: JSON.stringify({
            task_id: els.taskSelect.value,
            seed: Number(els.seedInput.value || 42),
            mode: els.runMode.value,
            model_name: els.modelInput.value || null,
            api_base_url: els.baseUrlInput.value || null,
            api_key: els.apiKeyInput.value || null,
          }),
        });
        state.timeline = (payload.steps || []).map(item => ({
          step: item.step,
          action: item.action,
          reward: item.reward,
          done: item.done,
          error: item.error,
          contextSnippet: item.context_snippet || '',
        }));
        state.done = true;
        state.episodeId = payload.episode_id;
        els.metricOutcome.textContent = payload.outcome || '-';
        els.metricReward.textContent = Number(payload.raw_reward || 0).toFixed(2);
        els.metricPsia.textContent = Number(payload.session_psia || 0).toFixed(2);
        els.metricCce.textContent = Number(payload.session_cce || 0.5).toFixed(2);
        els.contextText.textContent = payload.final_context || 'Episode complete.';
        els.statusText.textContent = JSON.stringify({
          episode_id: payload.episode_id,
          task_id: payload.task_id,
          mode_used: payload.mode_used,
          model_label: payload.model_label,
          outcome: payload.outcome,
          raw_reward: payload.raw_reward,
          score: payload.score,
        }, null, 2);
        els.graderText.textContent = JSON.stringify(payload.grader, null, 2);
        els.runModeText.textContent = JSON.stringify({
          requested_mode: els.runMode.value,
          mode_used: payload.mode_used,
          model_label: payload.model_label,
          used_fallback: payload.used_fallback,
          fallback_reason: payload.fallback_reason,
        }, null, 2);
        renderActionChips([]);
        renderTimeline();
        els.graderBtn.disabled = false;
      } catch (err) {
        els.runModeText.textContent = 'Automatic run failed: ' + err.message;
      } finally {
        syncRunButtons(false);
      }
    }

    async function runGrader() {
      if (!state.episodeId || !state.done) return;
      const result = await api('/grader', {
        method: 'POST',
        body: JSON.stringify({ episode_id: state.episodeId }),
      });
      els.graderText.textContent = JSON.stringify(result, null, 2);
      els.metricOutcome.textContent = result.outcome || '-';
      els.metricPsia.textContent = Number(result.session_psia || 0).toFixed(2);
      els.metricCce.textContent = Number(result.session_cce || 0.5).toFixed(2);
    }

    els.resetBtn.addEventListener('click', startEpisode);
    els.stepBtn.addEventListener('click', submitStep);
    els.graderBtn.addEventListener('click', runGrader);
    els.autoRunBtn.addEventListener('click', autoRunEpisode);

    loadTasks().catch(err => {
      els.statusText.textContent = 'Failed to load task catalog: ' + err.message;
    });
  </script>
</body>
</html>"""


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
