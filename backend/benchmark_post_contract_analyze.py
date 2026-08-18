"""
ANALYSIS-ONLY SCRIPT — isolated from production code.

Computes the post-contract qwen3:4b vs qwen3.5:4b comparison from
benchmark_results/post_contract/raw_results.json and writes
comparison.json into the same isolated namespace.

PLAN-QUALITY CAVEAT (important, stated in the output too): PIPER has no
built-in automated plan-quality scorer. The coverage checks below are
defined HERE, for this benchmark, from objectively measurable Titanic
properties (verified via pandas + the project's own pandas-3-safe
is_numeric_dtype helper):

    missing values : Age 177 (19.9%), Cabin 687 (77.1%), Embarked 2
    identifier-like: PassengerId, Name  (>=90% unique)
    low-card cats  : Sex (2), Embarked (3)      -> need encoding
    high-card text : Name (891), Ticket (681), Cabin (147)  -> need dropping

They are factual coverage checks ("did the plan address this measurable
dataset property?"), NOT a validated quality score and NOT a subjective
rating. A plan can be fully deterministic-validation-VALID while scoring
low here — that distinction is the entire point: validity is PIPER's
authority, coverage is a separate, descriptive lens.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "benchmark_results" / "post_contract"

TARGET = "Survived"
COLS_WITH_MISSING = {"Age", "Cabin", "Embarked"}
IDENTIFIER_LIKE = {"PassengerId", "Name"}
LOW_CARD_CATS = {"Sex", "Embarked"}
HIGH_CARD_TEXT = {"Name", "Ticket", "Cabin"}


def _plan_coverage(steps: list[dict]) -> dict:
    """Deterministic coverage checks over one plan's tool calls."""
    dropped, encoded, imputed, scaled, converted = set(), set(), set(), set(), set()
    touched: set[str] = set()

    for s in steps:
        tool, args = s["tool_name"], s["arguments"]
        if tool == "drop_column":
            c = args.get("column")
            if isinstance(c, str):
                dropped.add(c); touched.add(c)
        elif tool == "impute_missing_values":
            c = args.get("column")
            if isinstance(c, str):
                imputed.add(c); touched.add(c)
        elif tool == "convert_column_type":
            c = args.get("column")
            if isinstance(c, str):
                converted.add(c); touched.add(c)
        elif tool == "encode_categorical_features":
            for c in args.get("columns", []) or []:
                encoded.add(c); touched.add(c)
        elif tool == "scale_features":
            for c in args.get("columns", []) or []:
                scaled.add(c); touched.add(c)

    handled_highcard = {c for c in HIGH_CARD_TEXT if c in dropped or c in encoded}
    return {
        "step_count": len(steps),
        "target_never_touched": TARGET not in touched,
        "imputes_age": "Age" in imputed,
        "encodes_sex_and_embarked": LOW_CARD_CATS.issubset(encoded),
        "handles_all_high_card_text": HIGH_CARD_TEXT.issubset(handled_highcard),
        "high_card_text_handled": sorted(handled_highcard),
        "high_card_text_left_raw": sorted(HIGH_CARD_TEXT - handled_highcard),
        "drops_identifier_like": sorted(IDENTIFIER_LIKE & dropped),
        "identifier_like_left": sorted(IDENTIFIER_LIKE - dropped),
        "applies_scaling": bool(scaled),
        "columns_dropped": sorted(dropped),
        "columns_encoded": sorted(encoded),
        "columns_imputed": sorted(imputed),
        "columns_scaled": sorted(scaled),
    }


CRITICAL_CHECKS = ["target_never_touched", "imputes_age", "encodes_sex_and_embarked",
                   "handles_all_high_card_text"]


def _summarize(model: str, trials: list[dict]) -> dict:
    n = len(trials)
    first_valid = [t for t in trials if t["first_attempt_valid"]]
    final_valid = [t for t in trials if t["final_plan_valid"]]
    replanned = [t for t in trials if t["replan_count"] > 0]
    ttv = [t["time_to_valid_plan"] for t in trials if t["time_to_valid_plan"] is not None]
    total_lat = [t["total_planning_latency"] for t in trials]
    gen_lat = [t["total_generation_latency"] for t in trials]
    pp_lat = [t["total_prompt_processing_latency"] for t in trials]

    all_attempts = [a for t in trials for a in t["attempts"]]
    tps = [a["tokens_per_sec"] for a in all_attempts if a["tokens_per_sec"]]
    out_tok = [a["output_tokens"] for a in all_attempts if a["output_tokens"]]
    viol = [a["validation_violation_count"] or 0 for a in all_attempts
            if a["validation_violation_count"] is not None]
    timeouts = [a for a in all_attempts if a["wall_clock_over_600s"]]
    tech_fail = [a for a in all_attempts if a["transport_error"] or a["parse_error"]]

    coverage = []
    for t in trials:
        a = t["attempts"][-1]
        if a["validation_passed"]:
            cov = _plan_coverage(a["tool_arguments"])
            cov["trial_id"] = t["trial_id"]
            coverage.append(cov)

    crit_pass = [c for c in coverage if all(c[k] for k in CRITICAL_CHECKS)]

    by_state = {t["cold_or_warm"]: [] for t in trials}
    for t in trials:
        by_state[t["cold_or_warm"]].append(t["total_planning_latency"])

    return {
        "model": model,
        "trials": n,
        "first_attempt_valid_rate": f"{len(first_valid)}/{n}",
        "final_valid_rate": f"{len(final_valid)}/{n}",
        "replan_rate": f"{len(replanned)}/{n}",
        "mean_replan_count": st.mean([t["replan_count"] for t in trials]),
        "mean_time_to_valid_plan_s": st.mean(ttv) if ttv else None,
        "median_time_to_valid_plan_s": st.median(ttv) if ttv else None,
        "mean_total_planning_latency_s": st.mean(total_lat),
        "median_total_planning_latency_s": st.median(total_lat),
        "mean_generation_latency_s": st.mean(gen_lat),
        "mean_prompt_processing_latency_s": st.mean(pp_lat),
        "mean_tokens_per_sec": st.mean(tps) if tps else None,
        "mean_output_tokens": st.mean(out_tok) if out_tok else None,
        "mean_validation_violations": st.mean(viol) if viol else 0.0,
        "timeout_rate": f"{len(timeouts)}/{len(all_attempts)}",
        "technical_failure_rate": f"{len(tech_fail)}/{len(all_attempts)}",
        "cold_trial1_latency_s": trials[0]["total_planning_latency"],
        "warm_trial2_latency_s": trials[1]["total_planning_latency"],
        "warm_trial3_latency_s": trials[2]["total_planning_latency"],
        "plan_coverage": coverage,
        "plans_passing_all_critical_coverage": f"{len(crit_pass)}/{len(coverage)}",
        "plan_step_counts": [c["step_count"] for c in coverage],
    }


def main():
    raw = json.loads((RESULTS_DIR / "raw_results.json").read_text())
    summaries = {m: _summarize(m, ts) for m, ts in raw["blocks"].items()}

    comparison = {
        "fingerprint": raw["fingerprint"],
        "plan_quality_caveat": (
            "Coverage checks are defined by this analysis script from objectively "
            "measurable Titanic properties; PIPER has no built-in plan-quality scorer. "
            "They are descriptive coverage facts, NOT a validated quality metric and "
            "NOT subjective scores. Deterministic validity remains PIPER's sole authority."
        ),
        "models": summaries,
    }
    (RESULTS_DIR / "comparison.json").write_text(json.dumps(comparison, indent=2, default=str))

    for m, s in summaries.items():
        print("=" * 72)
        print(m)
        print("=" * 72)
        for k, v in s.items():
            if k in ("plan_coverage", "model"):
                continue
            print(f"  {k:38s} {v}")
        print("  plan coverage per trial:")
        for c in s["plan_coverage"]:
            crit = "PASS" if all(c[k] for k in CRITICAL_CHECKS) else "FAIL"
            print(f"    {c['trial_id']:22s} steps={c['step_count']} critical={crit}")
            print(f"       target_safe={c['target_never_touched']} imputes_Age={c['imputes_age']} "
                  f"encodes_Sex+Embarked={c['encodes_sex_and_embarked']} "
                  f"all_highcard_handled={c['handles_all_high_card_text']}")
            print(f"       high_card_left_raw={c['high_card_text_left_raw']} "
                  f"identifier_left={c['identifier_like_left']} scaling={c['applies_scaling']}")
        print()

    print(f"Wrote {RESULTS_DIR / 'comparison.json'}")


if __name__ == "__main__":
    main()
