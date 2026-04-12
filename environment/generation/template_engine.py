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
                wrong_loop_desc = rng.choice(self.LOOP_DESCS)
                stale_branch = wrong[0]
                dead_branch = wrong[1]
                steps.append({
                    "context": (
                        f"You reach the critical junction. Three passages branch ahead: {', '.join(branches)}. "
                        f"One branch ('{stale_branch}') carries your old chalk mark and likely {wrong_loop_desc} "
                        f"Another ('{dead_branch}') is still, dusty, and unusually warm. "
                        f"The remaining branch has faint airflow and distant mechanical noise from above."
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
        {
            "question": "Do stricter fraud models reduce payment losses without harming checkout approval rates?",
            "source_a": "A tighter fraud model cut chargeback losses by 19% in high-risk segments during a six-week pilot.",
            "source_d": "The same pilot reduced approval rates for legitimate cross-border customers by 11%, erasing most revenue gains outside the highest-risk segment.",
            "correct_synthesis": "Stricter fraud models reduce payment losses in high-risk segments, but can hurt approval rates enough to offset gains in broader traffic.",
            "wrong_synthesis_a": "Stricter fraud models are uniformly better because they reduce payment losses.",
            "wrong_synthesis_b": "Stricter fraud models are ineffective because approval rates fell in the pilot.",
            "decoy_sources": [
                "A merchant survey reports stronger trust in the checkout flow after fraud controls were tightened, but it does not measure approval rates or losses directly.",
                "An operations review found fewer manual review escalations after the new model shipped, which sounds supportive without answering the tradeoff question.",
                "A dashboard shows lower false-positive counts in one geography, but that slice excludes the cross-border cohort where the main disagreement appears.",
            ],
        },
        {
            "question": "Does adding retrieval improve enterprise support-chat accuracy?",
            "source_a": "Support agents using retrieval-backed responses answered policy questions correctly 24% more often in benchmarked ticket simulations.",
            "source_d": "On tickets involving outdated internal docs, retrieval amplified stale answers and reduced resolution quality unless document freshness checks were added.",
            "correct_synthesis": "Retrieval improves enterprise support-chat accuracy when documents are fresh, but can degrade performance when stale documentation is retrieved without freshness controls.",
            "wrong_synthesis_a": "Retrieval always improves enterprise support-chat accuracy.",
            "wrong_synthesis_b": "Retrieval does not improve support-chat accuracy in practice.",
            "decoy_sources": [
                "Customer satisfaction rose after the support team rolled out quicker first-response messages, but the change bundled retrieval with UI improvements.",
                "A platform vendor benchmark advertises lower hallucination rates with retrieval, though it does not discuss stale-document failure modes.",
                "An internal note says agents felt more confident with retrieval, which sounds persuasive but is not the same as answer correctness.",
            ],
        },
        {
            "question": "Do four-day workweeks improve team productivity?",
            "source_a": "Teams piloting a four-day week maintained output on individually owned deliverables while reporting lower burnout.",
            "source_d": "Cross-team dependency work slipped in organizations where coordination windows shrank, reducing throughput on shared projects.",
            "correct_synthesis": "Four-day workweeks can preserve productivity on individually owned work, but coordination-heavy projects may slow when shared availability shrinks.",
            "wrong_synthesis_a": "Four-day workweeks universally improve productivity.",
            "wrong_synthesis_b": "Four-day workweeks reduce productivity across the board.",
            "decoy_sources": [
                "Employee-retention surveys improved sharply during the pilot, which is relevant context but not direct productivity evidence.",
                "Managers reported better meeting discipline after the schedule change, a plausible mechanism that does not fully answer the output question.",
                "A recruiting team saw more inbound applicants after the policy announcement, which can be mistaken for performance evidence.",
            ],
        },
        {
            "question": "Do mandatory code reviews reduce production incidents?",
            "source_a": "Teams adopting mandatory reviews saw fewer configuration-related production incidents over the next quarter.",
            "source_d": "The same teams experienced slower hotfix turnaround and no reduction in incident volume for time-critical on-call patches.",
            "correct_synthesis": "Mandatory code reviews reduce some production incidents, especially configuration mistakes, but the benefit weakens for urgent hotfix paths where speed matters.",
            "wrong_synthesis_a": "Mandatory code reviews always reduce production incidents.",
            "wrong_synthesis_b": "Mandatory code reviews do not reduce production incidents.",
            "decoy_sources": [
                "Developers reported higher confidence in merged changes after reviews became mandatory, which sounds promising but is subjective.",
                "A platform dashboard shows fewer reverted commits, though it does not isolate production-incident outcomes.",
                "A team retrospective credits the policy for improved mentoring, which matters culturally without resolving the incident-effect question.",
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

        # Build meaningful action names from the actual synthesis options
        correct_action  = "conclude_" + "_".join(topic["correct_synthesis"].split()[:5]).lower().rstrip(".,")
        wrong_action_a  = "conclude_" + "_".join(topic["wrong_synthesis_a"].split()[:5]).lower().rstrip(".,")
        wrong_action_b  = "conclude_" + "_".join(topic["wrong_synthesis_b"].split()[:5]).lower().rstrip(".,")
        wrong_action_c  = "flag_evidence_insufficient_and_withhold_conclusion"
        # Sanitise to valid action identifiers
        import re
        def clean(s): return re.sub(r'[^a-z0-9_]', '_', s)[:60]
        correct_action  = clean(correct_action)
        wrong_action_a  = clean(wrong_action_a)
        wrong_action_b  = clean(wrong_action_b)

        source_labels = ["review_source_on_experimental_design", "review_source_on_sample_population",
                         "check_for_contradicting_literature", "assess_effect_size_and_confidence_intervals",
                         "examine_methodology_and_controls", "review_meta_analysis_coverage"]

        for t in range(t_total):
            if t == 0:
                steps.append({
                    "context": (
                        f"Policy brief request: '{topic['question']}'\n"
                        f"You have retrieved 5 studies. Preliminary scan suggests the evidence is mixed. "
                        f"You need to produce a defensible synthesis before the stakeholder deadline."
                    ),
                    "available_actions": [
                        "skim_all_abstracts_before_deep_review",
                        "start_with_highest_cited_paper",
                        "start_with_most_recent_publication",
                        "request_additional_literature_search",
                        "begin_structured_evidence_table",
                    ],
                    "default_action": "skim_all_abstracts_before_deep_review",
                })
                causal_chain.append("Step 0: Initial orientation. Approach choice has no causal effect — all evidence remains accessible regardless.")
            elif t == piv_idx:
                steps.append({
                    "context": (
                        f"You have now reviewed all five studies. Two directly contradict each other on the key question.\n\n"
                        f"Study A ({topic['source_a']})\n"
                        f"Study D ({topic['source_d']})\n\n"
                        f"Both studies are well-designed. The disagreement is real. "
                        f"Three other studies provide supporting context but do not resolve this contradiction directly. "
                        f"You must now write the synthesis conclusion. Which statement most accurately captures what the combined evidence supports?"
                    ),
                    "available_actions": [
                        correct_action,
                        wrong_action_a,
                        wrong_action_b,
                        wrong_action_c,
                    ],
                    "default_action": wrong_action_a,
                })
                causal_chain.append(
                    f"Step {t}: PIVOTAL — the synthesis conclusion locks in the answer. "
                    f"Only '{correct_action}' correctly accounts for the scope conditions under which both studies hold."
                )
            elif t == t_total - 1:
                steps.append({
                    "context": (
                        "Final step: the brief is drafted. You can submit it as-is, add a limitations caveat, "
                        "or request peer review before delivery."
                    ),
                    "available_actions": [
                        "submit_brief_as_drafted",
                        "add_limitations_section_then_submit",
                        "request_internal_peer_review_before_submitting",
                    ],
                    "default_action": "add_limitations_section_then_submit",
                })
                causal_chain.append(f"Step {t}: Submission formatting. Outcome was determined at step {piv_idx}.")
            else:
                decoy_src = rng.choice(topic["decoy_sources"])
                label = rng.choice(source_labels)
                if decoy_similarity >= 0.75:
                    context = (
                        f"Deep dive — {label.replace('_', ' ')}. "
                        f"Finding: {decoy_src} "
                        f"This looks like it could settle the question on its own, but on closer inspection it does not directly address the conditions under which the effect holds."
                    )
                elif decoy_similarity >= 0.45:
                    context = (
                        f"Literature review — {label.replace('_', ' ')}. "
                        f"Finding: {decoy_src} "
                        f"Relevant context, but does not resolve the core disagreement between Study A and Study D."
                    )
                else:
                    context = (
                        f"Background scan — {label.replace('_', ' ')}. "
                        f"Finding: {decoy_src}"
                    )
                steps.append({
                    "context": context,
                    "available_actions": [
                        "record_finding_and_continue_review",
                        "flag_as_potentially_decisive_and_investigate_further",
                        "dismiss_as_out_of_scope",
                    ],
                    "default_action": "record_finding_and_continue_review",
                })
                causal_chain.append(
                    f"Step {t}: Decoy — supporting evidence that is persuasive but does not resolve the A-vs-D contradiction."
                )

        cf = {
            f"{wrong_action_a}_at_step_{piv_idx}": "failure — overclaims universality, ignores scope conditions",
            f"{wrong_action_b}_at_step_{piv_idx}": "failure — underclaims, ignores positive evidence",
            f"{wrong_action_c}_at_step_{piv_idx}": "failure — no synthesis produced, brief cannot be delivered",
            f"{correct_action}_at_step_{piv_idx}": "success",
        }

        return Episode(
            episode_id="",
            domain="research",
            tier=tier,
            t_total=t_total,
            max_steps=20,
            steps=steps,
            pivotal_indices=pivot_positions,
            pivotal_actions=[correct_action],
            decoy_steps=decoys,
            causal_chain=causal_chain,
            counterfactuals=cf,
            initial_state={"sources_read": [], "notes": []},
            min_steps=piv_idx + 2,
        )


class DebuggingGeneratorV2(DebuggingGenerator):
    BUG_SETS = [
        {
            "title": "compute_stats() — statistical analysis function",
            "bugs": [
                {"id": "A", "desc": "Off-by-one in loop range (line 3)", "symptom": "Mean value is wrong on all inputs", "fix": "patch_loop_boundary_off_by_one", "severity": "high"},
                {"id": "B", "desc": "Division by zero when n=1 (line 7)", "symptom": "ZeroDivisionError on single-item lists", "fix": "add_guard_for_single_item_list", "severity": "high"},
                {"id": "C", "desc": "Wrong variable name: total vs _sum (line 12)", "symptom": "NameError crashes specific downstream code path", "fix": "rename_accumulator_variable_to_match_reference", "severity": "medium"},
                {"id": "D", "desc": "Float precision in comparison (line 18)", "symptom": "Wrong conditional branch on boundary values", "fix": "replace_float_equality_with_tolerance_check", "severity": "low"},
            ],
            "dependency": "C",
            "correct_first": "rename_accumulator_variable_to_match_reference",
            "wrong_first": ["patch_loop_boundary_off_by_one", "add_guard_for_single_item_list", "replace_float_equality_with_tolerance_check"],
            "decoy_progress": "Fixing the loop and division bugs clears the most visible test failures and makes the function appear stable — but the variable name mismatch is the upstream fault that the other fixes depend on.",
        },
        {
            "title": "parse_config() — configuration parser",
            "bugs": [
                {"id": "A", "desc": "Missing null check on optional field (line 5)", "symptom": "AttributeError when config file omits optional keys", "fix": "add_null_check_for_optional_fields", "severity": "high"},
                {"id": "B", "desc": "Type coercion fails for boolean strings (line 11)", "symptom": "String 'true' and 'false' not parsed as booleans", "fix": "add_string_to_bool_coercion", "severity": "high"},
                {"id": "C", "desc": "File reader uses wrong encoding (line 2)", "symptom": "UnicodeDecodeError on configs with non-ASCII characters", "fix": "set_utf8_encoding_in_file_reader", "severity": "critical"},
                {"id": "D", "desc": "Cache not invalidated after config update (line 19)", "symptom": "Stale configuration served after hot reload", "fix": "trigger_cache_invalidation_on_write", "severity": "medium"},
            ],
            "dependency": "C",
            "correct_first": "set_utf8_encoding_in_file_reader",
            "wrong_first": ["add_null_check_for_optional_fields", "add_string_to_bool_coercion", "trigger_cache_invalidation_on_write"],
            "decoy_progress": "Null checks and boolean coercion fix the most common failure modes, which makes the encoding bug easy to overlook — but it silently corrupts all downstream parsing for non-ASCII configs.",
        },
        {
            "title": "render_invoice() — billing pipeline renderer",
            "bugs": [
                {"id": "A", "desc": "VAT rounding uses display precision instead of accounting precision (line 9)", "symptom": "Invoice totals mismatched by one cent on some orders", "fix": "use_decimal_accounting_precision_for_vat", "severity": "medium"},
                {"id": "B", "desc": "Currency formatter drops negative sign on refunds (line 14)", "symptom": "Refund invoices display positive balance instead of negative", "fix": "preserve_negative_sign_in_currency_formatter", "severity": "high"},
                {"id": "C", "desc": "Template loader reads stale schema version from cache (line 2)", "symptom": "Invoice renderer ignores all fields added after last deploy", "fix": "clear_schema_cache_on_template_load", "severity": "critical"},
                {"id": "D", "desc": "Discount block crashes when coupon field is absent (line 21)", "symptom": "KeyError on invoices with no discount applied", "fix": "add_safe_access_for_optional_discount_block", "severity": "high"},
            ],
            "dependency": "C",
            "correct_first": "clear_schema_cache_on_template_load",
            "wrong_first": ["preserve_negative_sign_in_currency_formatter", "add_safe_access_for_optional_discount_block", "use_decimal_accounting_precision_for_vat"],
            "decoy_progress": "Patching the refund sign and the coupon crash removes the most customer-visible bugs, but the schema cache is silently omitting new fields from every invoice — that fault cascades through all downstream rendering.",
        },
        {
            "title": "sync_profiles() — CRM account synchronizer",
            "bugs": [
                {"id": "A", "desc": "Null company field causes merge comparator mismatch (line 6)", "symptom": "Duplicate customer profiles created on every sync", "fix": "handle_null_company_field_in_merge_comparator", "severity": "high"},
                {"id": "B", "desc": "Retry loop does not back off on rate-limit responses (line 18)", "symptom": "Burst of 429 errors hammers the API after any transient failure", "fix": "add_exponential_backoff_to_retry_loop", "severity": "medium"},
                {"id": "C", "desc": "Primary key mapping loads obsolete field alias (line 3)", "symptom": "Records cannot be matched across systems — all treated as new", "fix": "update_primary_key_mapping_to_current_schema", "severity": "critical"},
                {"id": "D", "desc": "Audit logger redacts IDs before deduplication summary runs (line 24)", "symptom": "Deduplication report missing record identifiers", "fix": "run_dedup_summary_before_audit_redaction", "severity": "low"},
            ],
            "dependency": "C",
            "correct_first": "update_primary_key_mapping_to_current_schema",
            "wrong_first": ["handle_null_company_field_in_merge_comparator", "add_exponential_backoff_to_retry_loop", "run_dedup_summary_before_audit_redaction"],
            "decoy_progress": "Fixing the duplicate-profile and backoff bugs reduces visible noise in the logs, but the stale key mapping is silently treating every existing record as new — that fault makes all other fixes ineffective.",
        },
        {
            "title": "authorize_claim() — insurance claims adjudication engine",
            "bugs": [
                {"id": "A", "desc": "Copay parser treats blank string as zero (line 8)", "symptom": "Some claims undercharge members on co-payments", "fix": "reject_blank_copay_and_require_explicit_zero", "severity": "medium"},
                {"id": "B", "desc": "Eligibility cache misses plan overrides on renewal date (line 17)", "symptom": "Recently renewed members fail authorization", "fix": "invalidate_eligibility_cache_on_plan_renewal", "severity": "high"},
                {"id": "C", "desc": "Coverage graph loads an obsolete rule bundle before adjudication (line 3)", "symptom": "All claims evaluated against the wrong policy tree", "fix": "force_coverage_graph_to_load_current_rule_bundle", "severity": "critical"},
                {"id": "D", "desc": "Audit serializer drops denial reason codes (line 26)", "symptom": "Operators cannot reconstruct why a claim was denied", "fix": "include_denial_reason_codes_in_audit_serializer", "severity": "low"},
            ],
            "dependency": "C",
            "correct_first": "force_coverage_graph_to_load_current_rule_bundle",
            "wrong_first": ["invalidate_eligibility_cache_on_plan_renewal", "reject_blank_copay_and_require_explicit_zero", "include_denial_reason_codes_in_audit_serializer"],
            "decoy_progress": "Fixing eligibility cache and co-pay parsing restores correct outcomes on most test claims, but the stale rule bundle is silently adjudicating every claim against last month's policy — fixing the other bugs on top of that foundation produces the wrong results.",
        },
        {
            "title": "schedule_pickups() — warehouse dispatch planner",
            "bugs": [
                {"id": "A", "desc": "Timezone conversion truncates half-hour UTC offsets (line 10)", "symptom": "Regional pickups scheduled 30 minutes late", "fix": "fix_timezone_offset_to_support_half_hour_zones", "severity": "high"},
                {"id": "B", "desc": "Vehicle capacity check ignores return-route load (line 21)", "symptom": "Some routes overbook drivers on the return leg", "fix": "include_return_route_load_in_capacity_check", "severity": "high"},
                {"id": "C", "desc": "Route graph loads yesterday's depot map from cache (line 4)", "symptom": "All routing decisions use an outdated network topology", "fix": "invalidate_route_graph_cache_before_planning_run", "severity": "critical"},
                {"id": "D", "desc": "Manifest exporter strips temperature handling tags (line 27)", "symptom": "Cold-chain pickups dispatched without temperature labels", "fix": "preserve_temperature_tags_in_manifest_export", "severity": "medium"},
            ],
            "dependency": "C",
            "correct_first": "invalidate_route_graph_cache_before_planning_run",
            "wrong_first": ["include_return_route_load_in_capacity_check", "fix_timezone_offset_to_support_half_hour_zones", "preserve_temperature_tags_in_manifest_export"],
            "decoy_progress": "Fixing capacity and timezone bugs immediately improves dispatch metrics and removes obvious failures, but the stale depot map is poisoning every routing decision — all subsequent optimisations are built on incorrect network topology.",
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
                    f"  • {b['desc']} — symptom: {b['symptom']} (severity: {b['severity']})"
                    for b in bug_set["bugs"]
                )
                steps.append({
                    "context": (
                        f"Production incident — {bug_set['title']}\n\n"
                        f"On-call triage has identified four contributing defects:\n{bug_list}\n\n"
                        "All four are real bugs. Fix order matters: one of them is a root-cause fault "
                        "that other fixes depend on. Resolving downstream symptoms first will appear to "
                        "make progress but leave the system broken in production.\n\n"
                        "Which fix do you apply first?"
                    ),
                    "available_actions": all_fixes,
                    "default_action": bug_set["wrong_first"][0],
                })
                causal_chain.append(f"Step {t}: Initial triage. Reviewing available patches — no fix applied yet.")
            elif t == piv_idx:
                dep_bug = next(b for b in bug_set["bugs"] if b["fix"] == bug_set["correct_first"])
                other_bugs = [b for b in bug_set["bugs"] if b["fix"] != bug_set["correct_first"]]
                steps.append({
                    "context": (
                        f"Engineering lead has escalated. Deeper trace analysis on {bug_set['title']}:\n\n"
                        f"The three most visible symptoms — {[b['symptom'] for b in other_bugs]} — "
                        f"each have straightforward patches ready. However, the traces also show that "
                        f"{dep_bug['desc']} ({dep_bug['symptom']}) is structurally upstream: "
                        f"it determines how the rest of the code executes.\n\n"
                        f"{bug_set['decoy_progress']}\n\n"
                        "Which fix do you apply at this stage?"
                    ),
                    "available_actions": all_fixes,
                    "default_action": bug_set["wrong_first"][0],
                })
                causal_chain.append(
                    f"Step {t}: PIVOTAL — the fix applied here determines whether the repair succeeds. "
                    f"Only '{bug_set['correct_first']}' resolves the upstream fault first."
                )
            elif t == t_total - 1:
                steps.append({
                    "context": (
                        "Remaining patches applied. Choose how to validate before closing the incident."
                    ),
                    "available_actions": [
                        "run_full_regression_suite_before_deploying",
                        "run_targeted_unit_tests_for_changed_modules",
                        "deploy_to_staging_and_monitor_error_rate",
                    ],
                    "default_action": "run_full_regression_suite_before_deploying",
                })
                causal_chain.append(f"Step {t}: Final validation. Outcome determined at step {piv_idx}.")
            else:
                decoy_bug = rng.choice([b for b in bug_set["bugs"] if b["fix"] != bug_set["correct_first"]])
                if decoy_similarity >= 0.75:
                    context = (
                        f"New error report on {bug_set['title']}. "
                        f"Affected users report: '{decoy_bug['symptom']}'. "
                        f"A patch for {decoy_bug['desc']} is ready and will resolve this complaint immediately. "
                        f"{bug_set['decoy_progress']}"
                    )
                elif decoy_similarity >= 0.45:
                    context = (
                        f"Investigating {bug_set['title']} further. "
                        f"Logs confirm: {decoy_bug['symptom']}. "
                        f"Root cause traced to {decoy_bug['desc']}. Patch is straightforward."
                    )
                else:
                    context = (
                        f"Review in progress. Next candidate: {decoy_bug['desc']} "
                        f"causing {decoy_bug['symptom']}."
                    )
                steps.append({
                    "context": context,
                    "available_actions": all_fixes + [
                        "add_detailed_logging_before_fixing",
                        "write_regression_test_to_confirm_fix",
                    ],
                    "default_action": decoy_bug["fix"],
                })
                causal_chain.append(
                    f"Step {t}: Decoy — a real visible bug with a ready patch, but not the upstream fault that determines overall success."
                )

        cf = {
            f"{bug_set['correct_first']}_at_step_{piv_idx}": "success — upstream fault resolved first",
            **{f"{w}_at_step_{piv_idx}": "failure — downstream symptom patched before upstream fault, leaving system broken" for w in bug_set["wrong_first"]},
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
            "reason": "One stateful dependency cannot be re-provisioned after the first commitment window closes, while the others can still be scaled later.",
            "wrong_first": ["compute_cluster_A", "cache_layer", "cdn_nodes"],
            "decoy_signal": "Compute and CDN spend promise immediate performance gains, so they look like the obvious high-impact moves.",
        },
        {
            "title": "Budget allocation for Q4 marketing campaign",
            "resources": ["paid_search", "social_media", "influencer_contracts", "event_sponsorship", "email_campaign"],
            "pivotal_resource": "influencer_contracts",
            "reason": "One channel has an 8-week lead time and becomes unavailable after week 2, while the others remain available later.",
            "wrong_first": ["paid_search", "email_campaign", "social_media"],
            "decoy_signal": "Paid channels can boost next-week metrics quickly, making them feel more urgent than the hidden lead-time constraint.",
        },
        {
            "title": "Disaster-response logistics for a regional flood event",
            "resources": ["helicopter_fuel", "portable_generators", "water_purification_units", "medical_kits", "temporary_shelters"],
            "pivotal_resource": "portable_generators",
            "reason": "One power-critical item must be loaded before the convoy route closes at dusk. The remaining supplies can be staged tomorrow.",
            "wrong_first": ["medical_kits", "water_purification_units", "temporary_shelters"],
            "decoy_signal": "Medical kits and shelters feel morally urgent, which makes generator allocation easy to postpone even though it is the irreversible bottleneck.",
        },
        {
            "title": "Semiconductor fab recovery after equipment outage",
            "resources": ["cleanroom_staff", "etcher_parts", "wafer_inventory", "vendor_diagnostics", "cooling_capacity"],
            "pivotal_resource": "vendor_diagnostics",
            "reason": "One diagnostic dependency expires after the first maintenance window; without it, the root fault cannot be isolated in time.",
            "wrong_first": ["cleanroom_staff", "wafer_inventory", "cooling_capacity"],
            "decoy_signal": "Staffing and cooling adjustments stabilize throughput dashboards immediately, making them look more valuable than the expiring diagnostic slot.",
        },
        {
            "title": "Hospital surge planning during respiratory outbreak",
            "resources": ["icu_beds", "oxygen_supply", "staff_redeployment", "telemetry_monitors", "elective_surgery_holds"],
            "pivotal_resource": "oxygen_supply",
            "reason": "One supply contract must be confirmed before a regional allocation freeze; the other capacity moves remain possible after the next operations review.",
            "wrong_first": ["icu_beds", "staff_redeployment", "telemetry_monitors"],
            "decoy_signal": "Bed and staffing dashboards move immediately when capacity is reassigned, making them look like the first lever to pull.",
        },
        {
            "title": "Security incident containment for exposed cloud credentials",
            "resources": ["token_rotation", "customer_notice", "forensic_snapshot", "ingress_rate_limits", "dashboard_monitoring"],
            "pivotal_resource": "token_rotation",
            "reason": "One credential action closes the active access window immediately; other responses are still useful later but do not stop ongoing misuse.",
            "wrong_first": ["forensic_snapshot", "customer_notice", "ingress_rate_limits"],
            "decoy_signal": "Forensics and notifications feel operationally urgent, which makes the revocation step easy to delay even though it is the true bottleneck.",
        },
        {
            "title": "Airport disruption recovery after weather cancellation",
            "resources": ["crew_reassignment", "gate_rebooking", "deicing_slot", "hotel_vouchers", "baggage_reconciliation"],
            "pivotal_resource": "deicing_slot",
            "reason": "One recovery slot expires before the next departure bank; the remaining service actions can be executed afterward without destroying the recovery path.",
            "wrong_first": ["crew_reassignment", "hotel_vouchers", "gate_rebooking"],
            "decoy_signal": "Passenger-facing fixes calm operations immediately, making them feel more important than the expiring operational slot.",
        },
    ]

    def generate(self, t_total, pivot_positions, decoy_similarity, rng, tier) -> Episode:
        scenario = rng.choice(self.SCENARIOS)
        piv_idx = pivot_positions[0]
        steps = []
        causal_chain = []
        decoys = [t for t in range(t_total) if t not in pivot_positions and t > 0 and t < t_total - 1]

        # Build meaningful action names: "commit_to_<resource>" instead of "allocate_<resource>"
        def action_name(resource):
            return f"commit_to_{resource}"
        allocate_actions = [action_name(r) for r in scenario["resources"]]
        correct_action   = action_name(scenario["pivotal_resource"])
        wrong_actions    = [action_name(w) for w in scenario["wrong_first"]]

        for t in range(t_total):
            if t == 0:
                res_display = "\n".join(f"  • {r.replace('_', ' ').title()}" for r in scenario["resources"])
                steps.append({
                    "context": (
                        f"Planning meeting — {scenario['title']}\n\n"
                        f"Five resource commitments are available:\n{res_display}\n\n"
                        f"You have budget for all of them but must sequence commitments carefully. "
                        f"Some resources remain available throughout the planning window; others have tight lead times or closing windows. "
                        f"Which commitment do you make first?"
                    ),
                    "available_actions": allocate_actions,
                    "default_action": allocate_actions[0],
                })
                causal_chain.append("Step 0: Planning phase. Reviewing options — no commitment made yet.")
            elif t == piv_idx:
                piv_resource = scenario["pivotal_resource"].replace("_", " ")
                steps.append({
                    "context": (
                        f"Operational update — {scenario['title']}\n\n"
                        f"{scenario['reason']}\n\n"
                        f"{scenario['decoy_signal']}\n\n"
                        f"The window for some commitments is narrowing. "
                        f"Which resource do you lock in now?"
                    ),
                    "available_actions": allocate_actions,
                    "default_action": wrong_actions[0],
                })
                causal_chain.append(
                    f"Step {t}: PIVOTAL — commitment window closing. "
                    f"Only '{correct_action}' secures the resource before the window closes."
                )
            elif t == t_total - 1:
                steps.append({
                    "context": (
                        "Remaining commitments finalised. Choose how to close out the allocation plan."
                    ),
                    "available_actions": [
                        "submit_completed_allocation_plan_for_sign_off",
                        "request_one_week_extension_to_revisit_scope",
                        "submit_partial_plan_with_open_items_flagged",
                    ],
                    "default_action": "submit_completed_allocation_plan_for_sign_off",
                })
                causal_chain.append(f"Step {t}: Plan submission. Outcome determined at step {piv_idx}.")
            else:
                other = rng.choice([r for r in scenario["resources"] if r != scenario["pivotal_resource"]])
                other_display = other.replace("_", " ").title()
                if decoy_similarity >= 0.75:
                    context = (
                        f"Resource review — {other_display}\n\n"
                        f"This commitment has a clear immediate return and strong stakeholder support. "
                        f"{scenario['decoy_signal']} "
                        f"It can be committed now or later without changing the outcome."
                    )
                elif decoy_similarity >= 0.45:
                    context = (
                        f"Resource review — {other_display}\n\n"
                        f"Standard approval checklist complete. Budget available. "
                        f"This is a flexible commitment — it can be rescheduled if priorities shift."
                    )
                else:
                    context = (
                        f"Routine review — {other_display}. Within budget. No hard deadline. "
                        f"Can be deferred to next planning cycle if needed."
                    )
                steps.append({
                    "context": context,
                    "available_actions": allocate_actions + [
                        "defer_this_commitment_to_next_review_cycle",
                        "request_updated_cost_estimate_before_committing",
                    ],
                    "default_action": action_name(other),
                })
                causal_chain.append(
                    f"Step {t}: Decoy allocation — a flexible resource with no hard window. "
                    "Can be committed in any order without affecting the outcome."
                )

        cf = {
            f"{correct_action}_at_step_{piv_idx}": "success — time-critical commitment secured",
            **{f"{w}_at_step_{piv_idx}": "failure — window for critical resource closes before it is committed" for w in wrong_actions},
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
            initial_state={"committed": [], "budget_remaining": 1.0},
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
        {
            "title": "Cloud platform outage diagnosis",
            "target": "the direct cause of a cascading regional outage",
            "signals": [
                {"id": "A", "name": "api_error_rate", "corr": 0.91, "causal": False, "desc": "API errors spike sharply, but they are mostly downstream of service unavailability."},
                {"id": "B", "name": "certificate_rotation_job", "corr": 0.64, "causal": True, "desc": "A failed certificate rotation invalidates trust for inter-service calls in the affected region."},
                {"id": "C", "name": "traffic_shift_volume", "corr": 0.83, "causal": False, "desc": "Traffic shifts rise after the outage begins as failover systems react."},
                {"id": "D", "name": "config_push_backlog", "corr": 0.77, "causal": True, "desc": "A stuck config propagation queue prevents repaired instances from receiving the corrected trust bundle."},
                {"id": "E", "name": "oncall_page_count", "corr": 0.72, "causal": False, "desc": "Paging volume grows after impact is already visible, which makes it highly correlated but non-causal."},
            ],
            "causal_signals": ["B", "D"],
            "correct_actions": ["flag_B_as_causal", "flag_D_as_causal"],
            "wrong_actions": ["flag_A_as_causal", "flag_C_as_causal", "flag_E_as_causal"],
        },
        {
            "title": "Marketplace fraud-loss spike investigation",
            "target": "the direct cause of a sudden fraud-loss increase",
            "signals": [
                {"id": "A", "name": "manual_review_queue", "corr": 0.79, "causal": False, "desc": "Manual review volume rose after suspicious transactions increased, but it does not generate the fraud itself."},
                {"id": "B", "name": "device_fingerprint_drop", "corr": 0.69, "causal": True, "desc": "A device-fingerprint dependency silently failed, removing a key risk signal from scoring."},
                {"id": "C", "name": "promo_redemption_rate", "corr": 0.58, "causal": False, "desc": "Promo use rose in the same period, making it easy to over-attribute the losses to campaign traffic."},
                {"id": "D", "name": "merchant_override_rule", "corr": 0.73, "causal": True, "desc": "A merchant override accidentally bypassed high-risk declines for a subset of card-not-present traffic."},
                {"id": "E", "name": "chargeback_notifications", "corr": 0.87, "causal": False, "desc": "Chargeback alerts are highly correlated because they are the lagging consequence of the fraud spike."},
            ],
            "causal_signals": ["B", "D"],
            "correct_actions": ["flag_B_as_causal", "flag_D_as_causal"],
            "wrong_actions": ["flag_A_as_causal", "flag_C_as_causal", "flag_E_as_causal"],
        },
        {
            "title": "Patient readmission spike investigation",
            "target": "the direct cause of readmission growth after discharge",
            "signals": [
                {"id": "A", "name": "followup_call_volume", "corr": 0.74, "causal": False, "desc": "Follow-up calls increased because more patients were already re-entering the system."},
                {"id": "B", "name": "discharge_instruction_regression", "corr": 0.66, "causal": True, "desc": "A form update removed medication guidance for one discharge cohort, directly worsening adherence."},
                {"id": "C", "name": "ed_wait_time", "corr": 0.71, "causal": False, "desc": "Emergency-department waits moved with volume but are not the trigger for readmissions."},
                {"id": "D", "name": "pharmacy_delivery_delay", "corr": 0.81, "causal": True, "desc": "Delayed post-discharge medication delivery caused treatment gaps for vulnerable patients."},
                {"id": "E", "name": "survey_response_rate", "corr": 0.55, "causal": False, "desc": "Survey participation fell during the same period, which is noticeable but not causally decisive."},
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