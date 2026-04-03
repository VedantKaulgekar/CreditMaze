"""
CreditMaze — Template Engine
Generates episodes from parameterised domain templates.
Each episode has a known causal structure with verified ground-truth labels.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


# ── Episode dataclass ─────────────────────────────────────────────────────────

@dataclass
class Episode:
    episode_id: str
    domain: str
    tier: str
    t_total: int
    max_steps: int
    steps: List[Dict]           # [{context, available_actions, default_action}, ...]
    pivotal_indices: List[int]
    pivotal_actions: List[str]
    decoy_steps: List[int]
    causal_chain: List[str]
    counterfactuals: Dict[str, str]
    initial_state: Dict[str, Any]
    faithfulness_score: float = 0.0
    min_steps: int = 1


# ── Tier configuration ────────────────────────────────────────────────────────

TIER_CONFIG = {
    "easy":        dict(t_total=10,  n_pivot=1, pivot_lo=5,  pivot_hi=9,  sim=0.2, max_s=15),
    "medium":      dict(t_total=14,  n_pivot=1, pivot_lo=1,  pivot_hi=5,  sim=0.5, max_s=20),
    "hard":        dict(t_total=12,  n_pivot=1, pivot_lo=1,  pivot_hi=3,  sim=0.8, max_s=18),
    "multi-pivot": dict(t_total=12,  n_pivot=2, pivot_lo=2,  pivot_hi=10, sim=0.9, max_s=18),
}

DOMAINS = ["corridor", "research", "debugging", "resource", "triage"]


# ── Main engine ───────────────────────────────────────────────────────────────

class TemplateEngine:
    def __init__(self):
        self._generators = {
            "corridor":  CorridorGenerator(),
            "research":  ResearchGenerator(),
            "debugging": DebuggingGenerator(),
            "resource":  ResourceGenerator(),
            "triage":    TriageGenerator(),
        }

    def generate(
        self,
        tier: str = "easy",
        domain: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Episode:
        rng = random.Random(seed)
        cfg = TIER_CONFIG[tier]

        # Multi-pivot requires triage domain — it is the only domain
        # built to generate N>1 independent pivotal actions at different steps.
        # All other domains have a single correct resolution per episode.
        if cfg["n_pivot"] > 1:
            domain = "triage"
        else:
            domain = domain or rng.choice(DOMAINS)

        # Draw pivot positions — unique indices within allowed range
        lo, hi = cfg["pivot_lo"], min(cfg["pivot_hi"], cfg["t_total"] - 1)
        population = list(range(lo, hi + 1))
        n = min(cfg["n_pivot"], len(population))
        pivot_positions = sorted(rng.sample(population, n))

        gen = self._generators[domain]
        episode = gen.generate(
            t_total=cfg["t_total"],
            pivot_positions=pivot_positions,
            decoy_similarity=cfg["sim"],
            rng=rng,
            tier=tier,
        )
        episode.max_steps = cfg["max_s"]
        return episode


# ════════════════════════════════════════════════════════════════════════════════
# DOMAIN 1 — Corridor Navigation
# ════════════════════════════════════════════════════════════════════════════════

class CorridorGenerator:
    BRANCHES = ["north", "south", "east", "west", "left", "right", "forward", "back"]
    LOOP_DESCS = [
        "leads to a dead end after 20 metres.",
        "circles back to the junction you came from.",
        "descends to a lower level with no exit.",
        "is blocked by a locked gate.",
        "loops through a maintenance corridor and returns here.",
    ]

    def generate(self, t_total, pivot_positions, decoy_similarity, rng, tier) -> Episode:
        pivot_idx = pivot_positions[0]
        branches  = rng.sample(self.BRANCHES, 3)
        correct   = branches[0]
        wrong     = branches[1:]

        steps = []
        causal_chain = []
        decoys = [t for t in range(t_total) if t not in pivot_positions and t > 0 and t < t_total - 1]

        for t in range(t_total):
            if t == 0:
                steps.append({
                    "context": (
                        "You enter a large underground facility. Your goal is to reach the emergency exit. "
                        "Multiple corridors branch from the central hub. Most loop back; only one path leads out."
                    ),
                    "available_actions": [f"enter_{b}" for b in branches],
                    "default_action": f"enter_{wrong[0]}",
                })
                causal_chain.append("Step 0: Entry point. No causal effect on outcome — all corridors accessible from hub.")

            elif t == pivot_idx:
                loop_desc = rng.choice(self.LOOP_DESCS)
                steps.append({
                    "context": (
                        f"You reach the critical junction. Three passages branch ahead: {', '.join(branches)}. "
                        f"You recall that passage '{wrong[0]}' {loop_desc} "
                        f"Passage '{correct}' is the only route to the exit level."
                    ),
                    "available_actions": [f"take_{b}" for b in branches],
                    "default_action": f"take_{wrong[0]}",
                })
                causal_chain.append(
                    f"Step {t}: PIVOTAL JUNCTION. take_{correct} → success. "
                    f"Any other choice → failure (loops or dead ends)."
                )

            elif t == t_total - 1:
                steps.append({
                    "context": "You emerge into daylight. The emergency exit is directly ahead.",
                    "available_actions": ["exit_facility", "go_back"],
                    "default_action": "exit_facility",
                })
                causal_chain.append(f"Step {t}: Exit reached. Outcome already determined at step {pivot_idx}.")

            else:
                n = t
                loop_note = rng.choice(self.LOOP_DESCS)
                steps.append({
                    "context": (
                        f"Corridor section {n}. You traverse a long passage. "
                        f"A side door on the {'left' if n % 2 == 0 else 'right'} {loop_note} "
                        "Continue forward or investigate?"
                    ),
                    "available_actions": ["continue_forward", "investigate_side_door"],
                    "default_action": "continue_forward",
                })
                causal_chain.append(
                    f"Step {t}: Decoy choice. investigate_side_door loops back. "
                    "continue_forward advances — neither affects pivotal outcome."
                )

        cf = {
            f"take_{w}_at_step_{pivot_idx}": "failure (loops back)"
            for w in wrong
        }
        cf[f"take_{correct}_at_step_{pivot_idx}"] = "success"

        return Episode(
            episode_id="", domain="corridor", tier=tier,
            t_total=t_total, max_steps=15,
            steps=steps,
            pivotal_indices=pivot_positions,
            pivotal_actions=[f"take_{correct}"],
            decoy_steps=decoys,
            causal_chain=causal_chain,
            counterfactuals=cf,
            initial_state={"location": "hub", "visited": []},
            min_steps=pivot_idx + 2,
        )


# ════════════════════════════════════════════════════════════════════════════════
# DOMAIN 2 — Multi-Document Research
# ════════════════════════════════════════════════════════════════════════════════

class ResearchGenerator:
    TOPICS = [
        {
            "question": "Does caffeine improve working memory?",
            "source_a": "Caffeine significantly improves working memory in sleep-deprived subjects (d=0.82, n=240).",
            "source_d": "Caffeine shows no significant effect on working memory in well-rested subjects (d=0.09, p=0.61).",
            "correct_synthesis": "Caffeine improves working memory specifically in sleep-deprived individuals, not universally.",
            "wrong_synthesis_a": "Caffeine universally improves working memory in all subjects.",
            "wrong_synthesis_b": "Caffeine has no meaningful effect on working memory.",
            "decoy_sources": [
                "Caffeine increases heart rate and blood pressure dose-dependently.",
                "Working memory capacity correlates with fluid intelligence (r=0.67).",
                "Sleep deprivation reduces prefrontal cortex activation by 18-24%.",
            ],
        },
        {
            "question": "Is remote work more productive than office work?",
            "source_a": "Remote workers show 13% higher output in structured task completion studies (n=500).",
            "source_d": "Remote work reduces creative collaboration output by 25% compared to in-person teams.",
            "correct_synthesis": "Remote work increases individual structured-task productivity but reduces collaborative creativity.",
            "wrong_synthesis_a": "Remote work universally increases productivity across all task types.",
            "wrong_synthesis_b": "Remote work is less productive than office work overall.",
            "decoy_sources": [
                "Remote workers save an average of 72 minutes per day in commute time.",
                "Office temperature preferences vary significantly among workers.",
                "Video conferencing adoption increased 300% since 2020.",
            ],
        },
    ]

    def generate(self, t_total, pivot_positions, decoy_similarity, rng, tier) -> Episode:
        topic    = rng.choice(self.TOPICS)
        piv_idx  = pivot_positions[0]
        steps    = []
        causal_chain = []
        decoys   = [t for t in range(t_total) if t not in pivot_positions and t > 0 and t < t_total - 1]

        for t in range(t_total):
            if t == 0:
                steps.append({
                    "context": (
                        f"Research task: '{topic['question']}'\n"
                        f"You have access to 5 sources. Begin by reviewing them systematically."
                    ),
                    "available_actions": ["read_source_A", "read_source_B", "read_source_C", "read_source_D", "read_source_E"],
                    "default_action": "read_source_B",
                })
                causal_chain.append("Step 0: Starting review. No causal effect — all sources available throughout.")

            elif t == piv_idx:
                steps.append({
                    "context": (
                        f"Critical finding: Source A states: '{topic['source_a']}'\n"
                        f"However, Source D contradicts: '{topic['source_d']}'\n"
                        f"You must decide how to synthesise these contradictory findings."
                    ),
                    "available_actions": [
                        "acknowledge_contradiction_and_qualify",
                        "trust_source_A_ignore_D",
                        "trust_source_D_ignore_A",
                        "flag_as_inconclusive",
                    ],
                    "default_action": "trust_source_A_ignore_D",
                })
                causal_chain.append(
                    f"Step {t}: PIVOTAL — resolving A vs D contradiction. "
                    "'acknowledge_contradiction_and_qualify' → qualified correct answer (success). "
                    "Any other choice → incorrect synthesis (failure)."
                )

            elif t == t_total - 1:
                steps.append({
                    "context": "Compile your synthesis into the final research report.",
                    "available_actions": ["submit_qualified_answer", "submit_strong_claim", "request_more_time"],
                    "default_action": "submit_qualified_answer",
                })
                causal_chain.append(f"Step {t}: Submission. Outcome determined at step {piv_idx}.")

            else:
                decoy_src = rng.choice(topic["decoy_sources"])
                steps.append({
                    "context": (
                        f"Reviewing supporting literature. Found: '{decoy_src}' "
                        f"This is relevant background information."
                    ),
                    "available_actions": ["note_and_continue", "deep_dive_this_source", "skip_to_next"],
                    "default_action": "note_and_continue",
                })
                causal_chain.append(
                    f"Step {t}: Decoy — gathering background. No path to outcome node. "
                    "Any action here is causally irrelevant."
                )

        cf = {
            f"trust_source_A_at_step_{piv_idx}": "failure — overclaims universality",
            f"trust_source_D_at_step_{piv_idx}": "failure — underclaims, ignores conditions",
            f"flag_inconclusive_at_step_{piv_idx}": "failure — no synthesis produced",
            f"acknowledge_contradiction_at_step_{piv_idx}": "success",
        }

        return Episode(
            episode_id="", domain="research", tier=tier,
            t_total=t_total, max_steps=20,
            steps=steps,
            pivotal_indices=pivot_positions,
            pivotal_actions=["acknowledge_contradiction_and_qualify"],
            decoy_steps=decoys,
            causal_chain=causal_chain,
            counterfactuals=cf,
            initial_state={"sources_read": [], "notes": []},
            min_steps=piv_idx + 2,
        )


# ════════════════════════════════════════════════════════════════════════════════
# DOMAIN 3 — Code Debugging Chain
# ════════════════════════════════════════════════════════════════════════════════

class DebuggingGenerator:
    BUG_SETS = [
        {
            "title": "compute_stats() — statistical analysis function",
            "bugs": [
                {"id": "A", "desc": "Off-by-one in loop range (line 3)", "symptom": "Wrong mean value", "fix": "fix_bug_A", "severity": "high"},
                {"id": "B", "desc": "Division by zero when n=1 (line 7)", "symptom": "ZeroDivisionError on single-item list", "fix": "fix_bug_B", "severity": "critical"},
                {"id": "C", "desc": "Wrong variable name: total vs _sum (line 12)", "symptom": "NameError in specific code path", "fix": "fix_bug_C", "severity": "medium"},
                {"id": "D", "desc": "Float precision in comparison (line 18)", "symptom": "Wrong conditional branch", "fix": "fix_bug_D", "severity": "low"},
            ],
            "dependency": "C",   # Must fix C first — A depends on C being resolved
            "correct_first": "fix_bug_C",
            "wrong_first": ["fix_bug_A", "fix_bug_B", "fix_bug_D"],
        },
        {
            "title": "parse_config() — configuration parser",
            "bugs": [
                {"id": "A", "desc": "Missing null check on optional field (line 5)", "symptom": "AttributeError on empty config", "fix": "fix_bug_A", "severity": "high"},
                {"id": "B", "desc": "Type coercion fails for boolean strings (line 11)", "symptom": "'true' not parsed as True", "fix": "fix_bug_B", "severity": "high"},
                {"id": "C", "desc": "Encoding issue in file reader (line 2)", "symptom": "UnicodeDecodeError on non-ASCII configs", "fix": "fix_bug_C", "severity": "critical"},
                {"id": "D", "desc": "Cache invalidation not triggered (line 19)", "symptom": "Stale config served after update", "fix": "fix_bug_D", "severity": "medium"},
            ],
            "dependency": "C",
            "correct_first": "fix_bug_C",
            "wrong_first": ["fix_bug_A", "fix_bug_B", "fix_bug_D"],
        },
    ]

    def generate(self, t_total, pivot_positions, decoy_similarity, rng, tier) -> Episode:
        bug_set = rng.choice(self.BUG_SETS)
        piv_idx = pivot_positions[0]
        steps   = []
        causal_chain = []
        decoys  = [t for t in range(t_total) if t not in pivot_positions and t > 0 and t < t_total - 1]
        all_fixes = [b["fix"] for b in bug_set["bugs"]]

        for t in range(t_total):
            if t == 0:
                bug_list = "\n".join(
                    f"  Bug {b['id']}: {b['desc']} — symptom: {b['symptom']} [severity: {b['severity']}]"
                    for b in bug_set["bugs"]
                )
                steps.append({
                    "context": (
                        f"You are debugging {bug_set['title']}.\n"
                        f"Test suite reveals 4 bugs:\n{bug_list}\n"
                        "Fix order matters due to internal dependencies. Choose which bug to fix first."
                    ),
                    "available_actions": all_fixes,
                    "default_action": bug_set["wrong_first"][0],
                })
                causal_chain.append(
                    f"Step {t}: Initial diagnosis. No fix applied yet."
                )

            elif t == piv_idx:
                dep_bug = next(b for b in bug_set["bugs"] if b["id"] == bug_set["dependency"])
                steps.append({
                    "context": (
                        f"Dependency analysis: Bug {dep_bug['id']} ({dep_bug['desc']}) "
                        f"must be resolved before the root cause can be properly addressed. "
                        f"Fixing other bugs first will create an irresolvable dependency cycle. "
                        f"Current test failures: {[b['symptom'] for b in bug_set['bugs'][:2]]}"
                    ),
                    "available_actions": all_fixes,
                    "default_action": bug_set["wrong_first"][0],
                })
                causal_chain.append(
                    f"Step {t}: PIVOTAL — fix order decision. "
                    f"{bug_set['correct_first']} → dependency resolved (success path). "
                    "Any other fix first → dependency cycle, failure."
                )

            elif t == t_total - 1:
                steps.append({
                    "context": "All remaining bugs fixed. Run final test suite.",
                    "available_actions": ["run_full_test_suite", "run_unit_tests_only", "deploy_without_tests"],
                    "default_action": "run_full_test_suite",
                })
                causal_chain.append(f"Step {t}: Final validation. Outcome set at step {piv_idx}.")

            else:
                remaining = rng.choice([b["fix"] for b in bug_set["bugs"][1:]])
                steps.append({
                    "context": (
                        f"Continuing debugging session. Bug still visible: "
                        f"'{rng.choice([b['symptom'] for b in bug_set['bugs']])}'. "
                        "Address next highest-severity issue."
                    ),
                    "available_actions": all_fixes + ["write_regression_test", "add_logging"],
                    "default_action": remaining,
                })
                causal_chain.append(
                    f"Step {t}: Decoy fix. Resolves a real bug but doesn't affect dependency structure."
                )

        cf = {
            f"{bug_set['correct_first']}_at_step_{piv_idx}": "success",
            **{f"{w}_at_step_{piv_idx}": "failure (dependency cycle)" for w in bug_set["wrong_first"]},
        }

        return Episode(
            episode_id="", domain="debugging", tier=tier,
            t_total=t_total, max_steps=18,
            steps=steps,
            pivotal_indices=pivot_positions,
            pivotal_actions=[bug_set["correct_first"]],
            decoy_steps=decoys,
            causal_chain=causal_chain,
            counterfactuals=cf,
            initial_state={"bugs_fixed": [], "test_failures": 4},
            min_steps=piv_idx + 2,
        )


# ════════════════════════════════════════════════════════════════════════════════
# DOMAIN 4 — Resource Allocation Puzzle
# ════════════════════════════════════════════════════════════════════════════════

class ResourceGenerator:
    SCENARIOS = [
        {
            "title": "Cloud infrastructure allocation for product launch",
            "resources": ["compute_cluster_A", "compute_cluster_B", "database_primary", "cache_layer", "cdn_nodes"],
            "pivotal_resource": "database_primary",
            "reason": "Database is the only resource that cannot be re-provisioned after initial allocation. All others can be scaled.",
            "wrong_first": ["compute_cluster_A", "cache_layer", "cdn_nodes"],
        },
        {
            "title": "Budget allocation for Q4 marketing campaign",
            "resources": ["paid_search", "social_media", "influencer_contracts", "event_sponsorship", "email_campaign"],
            "pivotal_resource": "influencer_contracts",
            "reason": "Influencer contracts require 8-week lead time and cannot be secured after week 2. All other channels remain available.",
            "wrong_first": ["paid_search", "email_campaign", "social_media"],
        },
    ]

    def generate(self, t_total, pivot_positions, decoy_similarity, rng, tier) -> Episode:
        scenario = rng.choice(self.SCENARIOS)
        piv_idx  = pivot_positions[0]
        steps    = []
        causal_chain = []
        decoys   = [t for t in range(t_total) if t not in pivot_positions and t > 0 and t < t_total - 1]

        allocate_actions = [f"allocate_{r}" for r in scenario["resources"]]
        correct_action   = f"allocate_{scenario['pivotal_resource']}"

        for t in range(t_total):
            if t == 0:
                res_list = ", ".join(scenario["resources"])
                steps.append({
                    "context": (
                        f"Task: {scenario['title']}.\n"
                        f"Available resources to allocate: {res_list}.\n"
                        "Note: resource availability windows differ. Prioritise accordingly."
                    ),
                    "available_actions": allocate_actions,
                    "default_action": allocate_actions[0],
                })
                causal_chain.append("Step 0: Planning phase. No allocation yet — all options open.")

            elif t == piv_idx:
                steps.append({
                    "context": (
                        f"Resource availability warning: {scenario['reason']} "
                        f"Current allocation budget: 40% committed. "
                        "Choose next resource to commit."
                    ),
                    "available_actions": allocate_actions,
                    "default_action": scenario["wrong_first"][0],
                })
                causal_chain.append(
                    f"Step {t}: PIVOTAL — irreversibility window. "
                    f"{correct_action} → secures critical resource (success). "
                    "Other allocations → critical resource window closes (failure)."
                )

            elif t == t_total - 1:
                steps.append({
                    "context": "Finalise allocation plan and submit for approval.",
                    "available_actions": ["submit_plan", "request_extension", "partial_submit"],
                    "default_action": "submit_plan",
                })
                causal_chain.append(f"Step {t}: Submission. Outcome set at step {piv_idx}.")

            else:
                other = rng.choice([r for r in scenario["resources"] if r != scenario["pivotal_resource"]])
                steps.append({
                    "context": (
                        f"Routine allocation step. {other.replace('_', ' ').title()} "
                        "is available and within budget. Standard approval process applies."
                    ),
                    "available_actions": allocate_actions + ["defer_decision", "request_cost_analysis"],
                    "default_action": f"allocate_{other}",
                })
                causal_chain.append(
                    f"Step {t}: Decoy allocation. Re-allocatable resource — no irreversibility."
                )

        cf = {
            f"{correct_action}_at_step_{piv_idx}": "success",
            **{f"allocate_{w}_at_step_{piv_idx}": "failure (pivotal resource window closes)"
               for w in scenario["wrong_first"]},
        }

        return Episode(
            episode_id="", domain="resource", tier=tier,
            t_total=t_total, max_steps=18,
            steps=steps,
            pivotal_indices=pivot_positions,
            pivotal_actions=[correct_action],
            decoy_steps=decoys,
            causal_chain=causal_chain,
            counterfactuals=cf,
            initial_state={"allocated": [], "budget_remaining": 1.0},
            min_steps=piv_idx + 2,
        )


# ════════════════════════════════════════════════════════════════════════════════
# DOMAIN 5 — Information Triage (Multi-Pivot capable)
# ════════════════════════════════════════════════════════════════════════════════

class TriageGenerator:
    SCENARIOS = [
        {
            "title": "Hospital outbreak investigation",
            "target": "root cause of infection cluster",
            "signals": [
                {"id": "A", "name": "cafeteria_usage",       "corr": 0.87, "causal": False, "desc": "High correlation — patients who visited cafeteria test positive more often."},
                {"id": "B", "name": "hand_sanitiser_access", "corr": 0.71, "causal": True,  "desc": "Negative correlation — wards with low sanitiser availability have higher rates."},
                {"id": "C", "name": "patient_age",           "corr": 0.52, "causal": False, "desc": "Moderate correlation — older patients at higher risk."},
                {"id": "D", "name": "ward_ventilation",      "corr": 0.68, "causal": True,  "desc": "Negative correlation — poor ventilation ratings predict higher rates."},
                {"id": "E", "name": "visitor_frequency",     "corr": 0.79, "causal": False, "desc": "High correlation — correlate of cafeteria_usage (confound)."},
            ],
            "causal_signals": ["B", "D"],
            "correct_actions": ["flag_B_as_causal", "flag_D_as_causal"],
            "wrong_actions": ["flag_A_as_causal", "flag_C_as_causal", "flag_E_as_causal"],
        },
        {
            "title": "E-commerce conversion rate investigation",
            "target": "direct cause of conversion drop",
            "signals": [
                {"id": "A", "name": "page_load_time",     "corr": 0.83, "causal": False, "desc": "High correlation — slow pages correlate with drop. But caused by same infrastructure issue."},
                {"id": "B", "name": "payment_gateway_lag","corr": 0.91, "causal": True,  "desc": "Checkout timeout errors directly cause cart abandonment."},
                {"id": "C", "name": "mobile_traffic_share","corr": 0.44, "causal": False, "desc": "More mobile users since the drop — but mobile conversion was already low."},
                {"id": "D", "name": "coupon_code_failure", "corr": 0.77, "causal": True,  "desc": "25% of coupons silently fail — directly prevents purchase completion."},
                {"id": "E", "name": "session_duration",   "corr": 0.65, "causal": False, "desc": "Sessions shorter — effect of payment failures, not a cause."},
            ],
            "causal_signals": ["B", "D"],
            "correct_actions": ["flag_B_as_causal", "flag_D_as_causal"],
            "wrong_actions": ["flag_A_as_causal", "flag_C_as_causal", "flag_E_as_causal"],
        },
    ]

    def generate(self, t_total, pivot_positions, decoy_similarity, rng, tier) -> Episode:
        scenario     = rng.choice(self.SCENARIOS)
        n_pivot      = len(pivot_positions)
        # Map pivot positions to causal signals
        causal_sigs  = scenario["causal_signals"][:n_pivot]
        correct_acts = scenario["correct_actions"][:n_pivot]
        steps        = []
        causal_chain = []
        decoys       = [t for t in range(t_total) if t not in pivot_positions and t > 0 and t < t_total - 1]

        all_flag_actions = [f"flag_{s['id']}_as_causal" for s in scenario["signals"]]
        all_flag_actions += ["flag_as_correlate", "request_more_data", "mark_inconclusive"]

        pivot_to_signal = {p: correct_acts[i] for i, p in enumerate(pivot_positions)}

        for t in range(t_total):
            if t == 0:
                signal_list = "\n".join(
                    f"  Signal {s['id']} ({s['name']}): r={s['corr']:.2f} — {s['desc']}"
                    for s in scenario["signals"]
                )
                steps.append({
                    "context": (
                        f"Investigation: {scenario['title']}\n"
                        f"Target: identify {scenario['target']}.\n"
                        f"Available signals:\n{signal_list}\n"
                        "Distinguish direct causes from correlates and confounds."
                    ),
                    "available_actions": all_flag_actions,
                    "default_action": "request_more_data",
                })
                causal_chain.append("Step 0: Investigation start. All signals still under review.")

            elif t in pivot_positions:
                pi  = pivot_positions.index(t)
                sig_id = causal_sigs[pi]
                sig    = next(s for s in scenario["signals"] if s["id"] == sig_id)
                others = [s for s in scenario["signals"] if s["id"] != sig_id]
                decoy_sig = rng.choice([s for s in others if not s["causal"]])
                steps.append({
                    "context": (
                        f"Deeper analysis of Signal {sig['id']} ({sig['name']}):\n"
                        f"  Correlation: {sig['corr']:.2f}\n"
                        f"  Mechanism: {sig['desc']}\n"
                        f"  Comparison: Signal {decoy_sig['id']} has higher correlation ({decoy_sig['corr']:.2f}) "
                        f"but {decoy_sig['desc']}\n"
                        "Is this signal a direct cause or a correlate?"
                    ),
                    "available_actions": [
                        f"flag_{sig['id']}_as_causal",
                        f"flag_{sig['id']}_as_correlate",
                        f"flag_{decoy_sig['id']}_as_causal",
                        "request_more_data",
                    ],
                    "default_action": f"flag_{decoy_sig['id']}_as_causal",
                })
                causal_chain.append(
                    f"Step {t}: PIVOT {pi+1} — classifying Signal {sig_id}. "
                    f"flag_{sig_id}_as_causal → correct identification (required for success). "
                    "Any other choice → misclassification (failure)."
                )

            elif t == t_total - 1:
                steps.append({
                    "context": "Compile findings into incident report and propose interventions.",
                    "available_actions": ["submit_causal_report", "submit_correlational_report", "escalate_investigation"],
                    "default_action": "submit_causal_report",
                })
                causal_chain.append(f"Step {t}: Report submission. Outcome set at pivot steps.")

            else:
                decoy_sig = rng.choice([s for s in scenario["signals"] if not s["causal"]])
                steps.append({
                    "context": (
                        f"Investigating Signal {decoy_sig['id']} ({decoy_sig['name']}). "
                        f"Correlation coefficient: {decoy_sig['corr']:.2f}. "
                        f"{decoy_sig['desc']} Gather more data?"
                    ),
                    "available_actions": ["collect_more_samples", "accept_current_data", "consult_domain_expert"],
                    "default_action": "accept_current_data",
                })
                causal_chain.append(
                    f"Step {t}: Decoy — investigating non-causal signal {decoy_sig['id']}. "
                    "High correlation but no causal mechanism. No path to outcome node."
                )

        cf = {**{f"{a}_at_step_{p}": "success" for p, a in pivot_to_signal.items()}}
        for sig in scenario["signals"]:
            if sig["id"] not in causal_sigs:
                for p in pivot_positions:
                    cf[f"flag_{sig['id']}_as_causal_at_step_{p}"] = "failure (misidentified correlate)"

        return Episode(
            episode_id="", domain="triage", tier=tier,
            t_total=t_total, max_steps=18,
            steps=steps,
            pivotal_indices=pivot_positions,
            pivotal_actions=correct_acts,
            decoy_steps=decoys,
            causal_chain=causal_chain,
            counterfactuals=cf,
            initial_state={"signals_reviewed": [], "flags": {}},
            min_steps=max(pivot_positions) + 2,
        )
