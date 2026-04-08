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
            "research":  ResearchGeneratorV2(),
            "debugging": DebuggingGeneratorV2(),
            "resource":  ResourceGeneratorV2(),
            "triage":    TriageGeneratorV2(),
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


class ResearchGeneratorV2(ResearchGenerator):
    TOPICS = [dict(topic) for topic in ResearchGenerator.TOPICS] + [
        {
            "question": "Do standing desks reduce long-term back pain?",
            "source_a": "Standing desks reduce self-reported lower-back pain among office workers over 12 weeks (n=310).",
            "source_d": "A 9-month follow-up found no durable pain reduction once novelty effects and compliance were controlled.",
            "correct_synthesis": "Standing desks may reduce back pain in the short term, but durable long-term effects remain unproven.",
            "wrong_synthesis_a": "Standing desks definitively reduce long-term back pain.",
            "wrong_synthesis_b": "Standing desks have no meaningful effect on back pain at any time horizon.",
            "decoy_sources": [
                "A workplace wellness pilot observed better posture scores after standing-desk adoption, but did not measure pain outcomes.",
                "An ergonomics review notes that movement breaks reduce discomfort, making standing desks look effective when paired with coaching.",
                "A facilities report found higher employee satisfaction after desk upgrades, though it bundled chairs, monitors, and sit-stand desks together.",
            ],
        },
        {
            "question": "Does intermittent fasting improve insulin sensitivity?",
            "source_a": "Intermittent fasting improves insulin sensitivity in overweight adults over 8 weeks (n=180).",
            "source_d": "In lean metabolically healthy adults, the same fasting schedule showed no significant insulin-sensitivity gain.",
            "correct_synthesis": "Intermittent fasting can improve insulin sensitivity in overweight adults, but benefits do not clearly generalize to already healthy populations.",
            "wrong_synthesis_a": "Intermittent fasting improves insulin sensitivity for everyone.",
            "wrong_synthesis_b": "Intermittent fasting does not improve insulin sensitivity.",
            "decoy_sources": [
                "A nutrition trial found intermittent fasting reduced calorie intake, which could explain metabolic benefits without isolating fasting itself.",
                "A gym cohort reported weight-loss success during fasting windows, but adherence differed sharply across participants.",
                "A review links fasting to reduced inflammation markers, which sounds supportive but does not directly answer the insulin-sensitivity question.",
            ],
        },
        {
            "question": "Do online courses teach programming as effectively as in-person bootcamps?",
            "source_a": "Self-paced online cohorts matched in-person bootcamp graduates on standardized programming assessments after 16 weeks.",
            "source_d": "The online cohorts underperformed on collaborative debugging and project-delivery exercises requiring live team interaction.",
            "correct_synthesis": "Online courses can match in-person bootcamps on core programming assessments, but may lag on collaborative project work.",
            "wrong_synthesis_a": "Online programming courses are fully equivalent to in-person bootcamps on all outcomes.",
            "wrong_synthesis_b": "Online programming courses are categorically worse than in-person bootcamps.",
            "decoy_sources": [
                "An edtech company reports higher completion rates after adding weekly reminders, which improves engagement but does not answer learning effectiveness directly.",
                "A student survey rates online courses as more convenient, a strong adoption signal that is easy to mistake for evidence of better instruction.",
                "A hiring manager panel says portfolios matter more than modality, which is relevant context but not outcome evidence from a controlled comparison.",
            ],
        },
        {
            "question": "Does AI code completion improve developer productivity?",
            "source_a": "Engineers using AI completion finished routine coding tasks 21% faster in controlled benchmark studies.",
            "source_d": "The same engineers spent more time verifying generated code on unfamiliar systems, erasing the productivity gain on complex maintenance work.",
            "correct_synthesis": "AI code completion improves productivity on routine coding tasks, but gains weaken or disappear on unfamiliar complex maintenance work.",
            "wrong_synthesis_a": "AI code completion universally improves developer productivity.",
            "wrong_synthesis_b": "AI code completion does not improve developer productivity at all.",
            "decoy_sources": [
                "A developer survey reports higher perceived flow state with AI assistance, which sounds persuasive but is subjective.",
                "A repo-wide analysis found more lines of code merged per day after tool adoption, but did not control for review latency or code quality.",
                "A case study shows fewer syntax errors in drafts, a useful signal that still does not answer end-to-end productivity across task types.",
            ],
        },
    ]
    TOPICS[0]["decoy_sources"] = [
        "A workplace field study reports faster spreadsheet completion after caffeine, but it did not test memory tasks directly.",
        "A small student sample found better recall after energy drinks, but dose and sleep status were not controlled.",
        "A review links caffeine to increased alertness, which sounds decisive but does not isolate memory effects.",
    ]
    TOPICS[1]["decoy_sources"] = [
        "A SaaS company case study reports a 19% increase in completed tickets after going remote, but measured only asynchronous support work.",
        "A consultancy found lower meeting overhead in remote teams and interpreted it as broad productivity improvement without tracking collaborative deliverables.",
        "A survey shows employees feel more productive at home, but the result is self-reported and mixes focus work with creative work.",
    ]

    def generate(self, t_total, pivot_positions, decoy_similarity, rng, tier) -> Episode:
        topic = rng.choice(self.TOPICS)
        piv_idx = pivot_positions[0]
        steps = []
        causal_chain = []
        decoys = [t for t in range(t_total) if t not in pivot_positions and t > 0 and t < t_total - 1]

        for t in range(t_total):
            if t == 0:
                steps.append({
                    "context": (
                        f"Research task: '{topic['question']}'\n"
                        "You have access to 5 sources. Several seem persuasive at first glance. Review them carefully before committing to a synthesis."
                    ),
                    "available_actions": ["read_source_A", "read_source_B", "read_source_C", "read_source_D", "read_source_E"],
                    "default_action": "read_source_B",
                })
                causal_chain.append("Step 0: Starting review. No causal effect - all sources available throughout.")
            elif t == piv_idx:
                steps.append({
                    "context": (
                        f"Source A reports: '{topic['source_a']}'\n"
                        f"Source D reports: '{topic['source_d']}'\n"
                        "Both look credible, but they were run under different conditions. Choose how to synthesise the evidence."
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
                    f"Step {t}: PIVOTAL - resolving the conditional contradiction. "
                    "'acknowledge_contradiction_and_qualify' leads to the correct qualified synthesis."
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
                if decoy_similarity >= 0.75:
                    context = (
                        f"Supporting source review. New evidence claims: '{decoy_src}' "
                        "It looks strong enough to justify an early answer, but its scope and causal relevance remain ambiguous."
                    )
                elif decoy_similarity >= 0.45:
                    context = (
                        f"Reviewing supporting literature. Found: '{decoy_src}' "
                        "This appears directly relevant and could plausibly change the final answer."
                    )
                else:
                    context = (
                        f"Reviewing supporting literature. Found: '{decoy_src}' "
                        "This is relevant background information."
                    )
                steps.append({
                    "context": context,
                    "available_actions": ["note_and_continue", "deep_dive_this_source", "skip_to_next"],
                    "default_action": "note_and_continue",
                })
                causal_chain.append(
                    f"Step {t}: Decoy - persuasive supporting evidence that still does not resolve the core contradiction."
                )

        cf = {
            f"trust_source_A_at_step_{piv_idx}": "failure - overclaims universality",
            f"trust_source_D_at_step_{piv_idx}": "failure - underclaims and ignores conditions",
            f"flag_inconclusive_at_step_{piv_idx}": "failure - no synthesis produced",
            f"acknowledge_contradiction_at_step_{piv_idx}": "success",
        }

        return Episode(
            episode_id="",
            domain="research",
            tier=tier,
            t_total=t_total,
            max_steps=20,
            steps=steps,
            pivotal_indices=pivot_positions,
            pivotal_actions=["acknowledge_contradiction_and_qualify"],
            decoy_steps=decoys,
            causal_chain=causal_chain,
            counterfactuals=cf,
            initial_state={"sources_read": [], "notes": []},
            min_steps=piv_idx + 2,
        )


class DebuggingGeneratorV2(DebuggingGenerator):
    BUG_SETS = [
        {
            "title": "compute_stats() - statistical analysis function",
            "bugs": [
                {"id": "A", "desc": "Off-by-one in loop range (line 3)", "symptom": "Wrong mean value", "fix": "fix_bug_A", "severity": "high"},
                {"id": "B", "desc": "Division by zero when n=1 (line 7)", "symptom": "ZeroDivisionError on single-item list", "fix": "fix_bug_B", "severity": "critical"},
                {"id": "C", "desc": "Wrong variable name: total vs _sum (line 12)", "symptom": "NameError in specific code path", "fix": "fix_bug_C", "severity": "medium"},
                {"id": "D", "desc": "Float precision in comparison (line 18)", "symptom": "Wrong conditional branch", "fix": "fix_bug_D", "severity": "low"},
            ],
            "dependency": "C",
            "correct_first": "fix_bug_C",
            "wrong_first": ["fix_bug_A", "fix_bug_B", "fix_bug_D"],
            "decoy_progress": "Alternative fixes clear visible tests and make the function look mostly repaired, but the upstream dependency fault remains latent.",
        },
        {
            "title": "parse_config() - configuration parser",
            "bugs": [
                {"id": "A", "desc": "Missing null check on optional field (line 5)", "symptom": "AttributeError on empty config", "fix": "fix_bug_A", "severity": "high"},
                {"id": "B", "desc": "Type coercion fails for boolean strings (line 11)", "symptom": "'true' not parsed as True", "fix": "fix_bug_B", "severity": "high"},
                {"id": "C", "desc": "Encoding issue in file reader (line 2)", "symptom": "UnicodeDecodeError on non-ASCII configs", "fix": "fix_bug_C", "severity": "critical"},
                {"id": "D", "desc": "Cache invalidation not triggered (line 19)", "symptom": "Stale config served after update", "fix": "fix_bug_D", "severity": "medium"},
            ],
            "dependency": "C",
            "correct_first": "fix_bug_C",
            "wrong_first": ["fix_bug_A", "fix_bug_B", "fix_bug_D"],
            "decoy_progress": "Superficial fixes stabilize common inputs, which makes the encoding bug easy to dismiss until the full pipeline runs again.",
        },
        {
            "title": "render_invoice() - billing pipeline renderer",
            "bugs": [
                {"id": "A", "desc": "VAT rounding uses display precision instead of accounting precision (line 9)", "symptom": "Totals mismatch by one cent", "fix": "fix_bug_A", "severity": "medium"},
                {"id": "B", "desc": "Currency formatter silently drops negative sign in refunds (line 14)", "symptom": "Refund invoices show positive balance", "fix": "fix_bug_B", "severity": "high"},
                {"id": "C", "desc": "Template loader chooses stale schema version from cache (line 2)", "symptom": "Renderer ignores newly added fields", "fix": "fix_bug_C", "severity": "critical"},
                {"id": "D", "desc": "Optional discount block crashes when coupon missing (line 21)", "symptom": "KeyError on no-discount invoices", "fix": "fix_bug_D", "severity": "high"},
            ],
            "dependency": "C",
            "correct_first": "fix_bug_C",
            "wrong_first": ["fix_bug_B", "fix_bug_D", "fix_bug_A"],
            "decoy_progress": "Patching the visible billing bugs removes obvious failures first, making the cached schema issue look less central than it really is.",
        },
        {
            "title": "sync_profiles() - CRM account synchronizer",
            "bugs": [
                {"id": "A", "desc": "Null company field causes merge comparator mismatch (line 6)", "symptom": "Duplicate profiles created", "fix": "fix_bug_A", "severity": "high"},
                {"id": "B", "desc": "Retry loop never backs off on rate-limit responses (line 18)", "symptom": "Bursts of repeated 429 errors", "fix": "fix_bug_B", "severity": "medium"},
                {"id": "C", "desc": "Primary key mapping loads obsolete field alias from schema adapter (line 3)", "symptom": "Correct records cannot be matched across systems", "fix": "fix_bug_C", "severity": "critical"},
                {"id": "D", "desc": "Audit logger redacts IDs before dedupe summary runs (line 24)", "symptom": "Dedupe report missing record identifiers", "fix": "fix_bug_D", "severity": "low"},
            ],
            "dependency": "C",
            "correct_first": "fix_bug_C",
            "wrong_first": ["fix_bug_A", "fix_bug_B", "fix_bug_D"],
            "decoy_progress": "The duplicate-profile symptom shrinks after superficial fixes, which makes the schema adapter bug seem less urgent than it is.",
        },
    ]

    def generate(self, t_total, pivot_positions, decoy_similarity, rng, tier) -> Episode:
        bug_set = rng.choice(self.BUG_SETS)
        piv_idx = pivot_positions[0]
        steps = []
        causal_chain = []
        decoys = [t for t in range(t_total) if t not in pivot_positions and t > 0 and t < t_total - 1]
        all_fixes = [b["fix"] for b in bug_set["bugs"]]

        for t in range(t_total):
            if t == 0:
                bug_list = "\n".join(
                    f"  Bug {b['id']}: {b['desc']} - symptom: {b['symptom']} [severity: {b['severity']}]"
                    for b in bug_set["bugs"]
                )
                steps.append({
                    "context": (
                        f"You are debugging {bug_set['title']}.\n"
                        f"Test suite reveals 4 bugs:\n{bug_list}\n"
                        "Several fixes look urgent. Fix order matters because some symptoms are downstream of an upstream blocker."
                    ),
                    "available_actions": all_fixes,
                    "default_action": bug_set["wrong_first"][0],
                })
                causal_chain.append(f"Step {t}: Initial diagnosis. No fix applied yet.")
            elif t == piv_idx:
                dep_bug = next(b for b in bug_set["bugs"] if b["id"] == bug_set["dependency"])
                steps.append({
                    "context": (
                        f"Dependency analysis highlights Bug {dep_bug['id']} ({dep_bug['desc']}) as an upstream blocker. "
                        "Other fixes would remove visible failures first, but may still leave the root cause untouched. "
                        f"Current test failures: {[b['symptom'] for b in bug_set['bugs'][:2]]}"
                    ),
                    "available_actions": all_fixes,
                    "default_action": bug_set["wrong_first"][0],
                })
                causal_chain.append(
                    f"Step {t}: PIVOTAL - fix-order decision. {bug_set['correct_first']} preserves the only success path."
                )
            elif t == t_total - 1:
                steps.append({
                    "context": "All remaining bugs fixed. Run final test suite.",
                    "available_actions": ["run_full_test_suite", "run_unit_tests_only", "deploy_without_tests"],
                    "default_action": "run_full_test_suite",
                })
                causal_chain.append(f"Step {t}: Final validation. Outcome set at step {piv_idx}.")
            else:
                decoy_bug = rng.choice([b for b in bug_set["bugs"] if b["fix"] != bug_set["correct_first"]])
                steps.append({
                    "context": (
                        f"Continuing debugging session. Patch candidate {decoy_bug['id']} would address "
                        f"'{decoy_bug['symptom']}'. {bug_set['decoy_progress']}"
                    ),
                    "available_actions": all_fixes + ["write_regression_test", "add_logging"],
                    "default_action": decoy_bug["fix"],
                })
                causal_chain.append(
                    f"Step {t}: Decoy fix. It resolves a real visible bug but does not change the dependency structure."
                )

        cf = {
            f"{bug_set['correct_first']}_at_step_{piv_idx}": "success",
            **{f"{w}_at_step_{piv_idx}": "failure (dependency cycle)" for w in bug_set["wrong_first"]},
        }

        return Episode(
            episode_id="",
            domain="debugging",
            tier=tier,
            t_total=t_total,
            max_steps=18,
            steps=steps,
            pivotal_indices=pivot_positions,
            pivotal_actions=[bug_set["correct_first"]],
            decoy_steps=decoys,
            causal_chain=causal_chain,
            counterfactuals=cf,
            initial_state={"bugs_fixed": [], "test_failures": 4},
            min_steps=piv_idx + 2,
        )


class ResourceGeneratorV2(ResourceGenerator):
    SCENARIOS = [
        {
            "title": "Cloud infrastructure allocation for product launch",
            "resources": ["compute_cluster_A", "compute_cluster_B", "database_primary", "cache_layer", "cdn_nodes"],
            "pivotal_resource": "database_primary",
            "reason": "Database is the only resource that cannot be re-provisioned after initial allocation. All others can still be scaled later.",
            "wrong_first": ["compute_cluster_A", "cache_layer", "cdn_nodes"],
            "decoy_signal": "Compute and CDN spend promise immediate performance gains, so they look like the obvious high-impact moves.",
        },
        {
            "title": "Budget allocation for Q4 marketing campaign",
            "resources": ["paid_search", "social_media", "influencer_contracts", "event_sponsorship", "email_campaign"],
            "pivotal_resource": "influencer_contracts",
            "reason": "Influencer contracts require 8-week lead time and cannot be secured after week 2. All other channels remain available later.",
            "wrong_first": ["paid_search", "email_campaign", "social_media"],
            "decoy_signal": "Paid channels can boost next-week metrics quickly, making them feel more urgent than the hidden lead-time constraint.",
        },
        {
            "title": "Disaster-response logistics for a regional flood event",
            "resources": ["helicopter_fuel", "portable_generators", "water_purification_units", "medical_kits", "temporary_shelters"],
            "pivotal_resource": "portable_generators",
            "reason": "Portable generators must be loaded before the convoy route closes at dusk. The remaining supplies can be staged tomorrow.",
            "wrong_first": ["medical_kits", "water_purification_units", "temporary_shelters"],
            "decoy_signal": "Medical kits and shelters feel morally urgent, which makes generator allocation easy to postpone even though it is the irreversible bottleneck.",
        },
        {
            "title": "Semiconductor fab recovery after equipment outage",
            "resources": ["cleanroom_staff", "etcher_parts", "wafer_inventory", "vendor_diagnostics", "cooling_capacity"],
            "pivotal_resource": "vendor_diagnostics",
            "reason": "Vendor diagnostics access expires after the first maintenance window; without it, the root fault cannot be isolated in time.",
            "wrong_first": ["cleanroom_staff", "wafer_inventory", "cooling_capacity"],
            "decoy_signal": "Staffing and cooling adjustments stabilize throughput dashboards immediately, making them look more valuable than the expiring diagnostic slot.",
        },
    ]

    def generate(self, t_total, pivot_positions, decoy_similarity, rng, tier) -> Episode:
        scenario = rng.choice(self.SCENARIOS)
        piv_idx = pivot_positions[0]
        steps = []
        causal_chain = []
        decoys = [t for t in range(t_total) if t not in pivot_positions and t > 0 and t < t_total - 1]

        allocate_actions = [f"allocate_{r}" for r in scenario["resources"]]
        correct_action = f"allocate_{scenario['pivotal_resource']}"

        for t in range(t_total):
            if t == 0:
                res_list = ", ".join(scenario["resources"])
                steps.append({
                    "context": (
                        f"Task: {scenario['title']}.\n"
                        f"Available resources to allocate: {res_list}.\n"
                        "Several options offer immediate visible benefits, but not all preserve the only successful execution path."
                    ),
                    "available_actions": allocate_actions,
                    "default_action": allocate_actions[0],
                })
                causal_chain.append("Step 0: Planning phase. No allocation yet - all options open.")
            elif t == piv_idx:
                steps.append({
                    "context": (
                        f"Availability review: {scenario['reason']} "
                        "Current allocation budget: 40% committed. Choose the next resource to lock in."
                    ),
                    "available_actions": allocate_actions,
                    "default_action": f"allocate_{scenario['wrong_first'][0]}",
                })
                causal_chain.append(
                    f"Step {t}: PIVOTAL - irreversible allocation window. {correct_action} secures the only durable success path."
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
                if decoy_similarity >= 0.75:
                    context = (
                        f"Allocation review: {other.replace('_', ' ').title()} has a strong immediate business case. "
                        f"{scenario['decoy_signal']}"
                    )
                elif decoy_similarity >= 0.45:
                    context = (
                        f"Allocation review: {other.replace('_', ' ').title()} appears to unlock near-term progress. "
                        "It looks like a plausible next commitment."
                    )
                else:
                    context = (
                        f"Routine allocation step. {other.replace('_', ' ').title()} is available and within budget. "
                        "Standard approval process applies."
                    )
                steps.append({
                    "context": context,
                    "available_actions": allocate_actions + ["defer_decision", "request_cost_analysis"],
                    "default_action": f"allocate_{other}",
                })
                causal_chain.append(
                    f"Step {t}: Decoy allocation. Plausibly useful, but not the irreversible bottleneck."
                )

        cf = {
            f"{correct_action}_at_step_{piv_idx}": "success",
            **{f"allocate_{w}_at_step_{piv_idx}": "failure (pivotal resource window closes)" for w in scenario["wrong_first"]},
        }

        return Episode(
            episode_id="",
            domain="resource",
            tier=tier,
            t_total=t_total,
            max_steps=18,
            steps=steps,
            pivotal_indices=pivot_positions,
            pivotal_actions=[correct_action],
            decoy_steps=decoys,
            causal_chain=causal_chain,
            counterfactuals=cf,
            initial_state={"allocated": [], "budget_remaining": 1.0},
            min_steps=piv_idx + 2,
        )


class TriageGeneratorV2(TriageGenerator):
    SCENARIOS = TriageGenerator.SCENARIOS + [
        {
            "title": "Consumer subscription churn investigation",
            "target": "direct cause of churn spike",
            "signals": [
                {"id": "A", "name": "support_ticket_volume", "corr": 0.82, "causal": False, "desc": "Support tickets rose sharply, but mostly after failed renewals already happened."},
                {"id": "B", "name": "card_retry_failures", "corr": 0.88, "causal": True, "desc": "Retry logic stops after one failed attempt, directly preventing successful renewals."},
                {"id": "C", "name": "pricing_page_views", "corr": 0.54, "causal": False, "desc": "More users viewed pricing, but many were already in rescue flows."},
                {"id": "D", "name": "loyalty_coupon_expiry", "corr": 0.76, "causal": True, "desc": "Coupons expired one week early, directly increasing effective renewal price."},
                {"id": "E", "name": "session_length", "corr": 0.67, "causal": False, "desc": "Sessions got shorter, but largely as a downstream effect of failed billing flows."},
            ],
            "causal_signals": ["B", "D"],
            "correct_actions": ["flag_B_as_causal", "flag_D_as_causal"],
            "wrong_actions": ["flag_A_as_causal", "flag_C_as_causal", "flag_E_as_causal"],
        },
    ]

    def generate(self, t_total, pivot_positions, decoy_similarity, rng, tier) -> Episode:
        scenario = rng.choice(self.SCENARIOS)
        n_pivot = len(pivot_positions)
        causal_sigs = scenario["causal_signals"][:n_pivot]
        correct_acts = scenario["correct_actions"][:n_pivot]
        steps = []
        causal_chain = []
        decoys = [t for t in range(t_total) if t not in pivot_positions and t > 0 and t < t_total - 1]

        all_flag_actions = [f"flag_{s['id']}_as_causal" for s in scenario["signals"]]
        all_flag_actions += ["flag_as_correlate", "request_more_data", "mark_inconclusive"]
        pivot_to_signal = {p: correct_acts[i] for i, p in enumerate(pivot_positions)}

        for t in range(t_total):
            if t == 0:
                signal_list = "\n".join(
                    f"  Signal {s['id']} ({s['name']}): r={s['corr']:.2f} - {s['desc']}"
                    for s in scenario["signals"]
                )
                steps.append({
                    "context": (
                        f"Investigation: {scenario['title']}\n"
                        f"Target: identify {scenario['target']}.\n"
                        f"Available signals:\n{signal_list}\n"
                        "Several non-causal signals are highly persuasive. Distinguish direct causes from correlates and confounds."
                    ),
                    "available_actions": all_flag_actions,
                    "default_action": "request_more_data",
                })
                causal_chain.append("Step 0: Investigation start. All signals still under review.")
            elif t in pivot_positions:
                pi = pivot_positions.index(t)
                sig_id = causal_sigs[pi]
                sig = next(s for s in scenario["signals"] if s["id"] == sig_id)
                others = [s for s in scenario["signals"] if s["id"] != sig_id]
                decoy_sig = rng.choice([s for s in others if not s["causal"]])
                steps.append({
                    "context": (
                        f"Deeper analysis of Signal {sig['id']} ({sig['name']}):\n"
                        f"  Correlation: {sig['corr']:.2f}\n"
                        f"  Mechanism: {sig['desc']}\n"
                        f"  Comparison: Signal {decoy_sig['id']} has comparable or higher surface evidence ({decoy_sig['corr']:.2f}) but may still be downstream.\n"
                        "Decide which signal is actually causal."
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
                    f"Step {t}: PIVOT {pi+1} - classify Signal {sig_id}. Correct classification is required for success."
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
                if decoy_similarity >= 0.75:
                    context = (
                        f"Investigating Signal {decoy_sig['id']} ({decoy_sig['name']}). "
                        f"Correlation coefficient: {decoy_sig['corr']:.2f}. {decoy_sig['desc']} "
                        "This could easily be mistaken for the main driver."
                    )
                elif decoy_similarity >= 0.45:
                    context = (
                        f"Investigating Signal {decoy_sig['id']} ({decoy_sig['name']}). "
                        f"Correlation coefficient: {decoy_sig['corr']:.2f}. {decoy_sig['desc']} "
                        "It remains a plausible causal candidate."
                    )
                else:
                    context = (
                        f"Investigating Signal {decoy_sig['id']} ({decoy_sig['name']}). "
                        f"Correlation coefficient: {decoy_sig['corr']:.2f}. {decoy_sig['desc']} Gather more data?"
                    )
                steps.append({
                    "context": context,
                    "available_actions": ["collect_more_samples", "accept_current_data", "consult_domain_expert"],
                    "default_action": "accept_current_data",
                })
                causal_chain.append(
                    f"Step {t}: Decoy - high-correlation signal that looks persuasive without providing the true causal mechanism."
                )

        cf = {**{f"{a}_at_step_{p}": "success" for p, a in pivot_to_signal.items()}}
        for sig in scenario["signals"]:
            if sig["id"] not in causal_sigs:
                for p in pivot_positions:
                    cf[f"flag_{sig['id']}_as_causal_at_step_{p}"] = "failure (misidentified correlate)"

        return Episode(
            episode_id="",
            domain="triage",
            tier=tier,
            t_total=t_total,
            max_steps=18,
            steps=steps,
            pivotal_indices=pivot_positions,
            pivotal_actions=correct_acts,
            decoy_steps=decoys,
            causal_chain=causal_chain,
            counterfactuals=cf,
            initial_state={"signals_reviewed": [], "flags": {}},
            min_steps=max(pivot_positions) + 2,
        )
