"""
ANALYSIS-ONLY SCRIPT — isolated from production code. Makes NO Ollama
calls and does not modify any recorded benchmark result.

Replays the deterministic Plan Adequacy evaluator against the SIX
already-recorded post-contract plans (qwen3:4b x3, qwen3.5:4b x3) from
benchmark_results/post_contract/raw_results.json, using the same
SanitizedLLMContext production builds for the real Titanic fixture.

Carries HARD ASSERTIONS for the effective-feature correction:

  1. OLD BEHAVIOR (historical baseline, recorded only) — under the prior
     uniform-severity rule, Cabin (77.10% missing, never an effective
     feature in any recorded plan) was MATERIAL and blocked.
  2. NEW BEHAVIOR — the same condition must now be ADVISORY and must NOT
     block adequacy.
  3. UNCHANGED BEHAVIOR — a missing column that IS an effective feature
     and is unaddressed must still be MATERIAL and still block.
  4. REGRESSION INVARIANT — no plan that previously PASSED may now FAIL.

The OLD baseline is read from the recorded adequacy_replay.json produced
before the correction, so the comparison is against real recorded data
rather than a re-derivation.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from app.agent.plan_adequacy import evaluate_plan_adequacy
from app.agent.tools.context_budget import apply_context_budget
from app.agent.tools.sanitized_llm_context import build_sanitized_llm_context
from app.storage import InMemoryDatasetStore

RESULTS_DIR = Path(__file__).parent / "benchmark_results" / "post_contract"
RAW = RESULTS_DIR / "raw_results.json"
OLD_REPLAY = RESULTS_DIR / "adequacy_replay.json"
OUT = RESULTS_DIR / "adequacy_replay_effective_feature.json"
DATASET = Path(__file__).parent.parent / "benchmark_data" / "train.csv"
TARGET = "Survived"

_FEATURE_SELECTING = {"encode_categorical_features", "scale_features"}


class _Step:
    """Minimal duck-typed stand-in (.tool_name/.arguments) for a recorded plan step."""

    def __init__(self, tool_name: str, arguments: dict):
        self.tool_name = tool_name
        self.arguments = arguments


def _effective_feature_set(steps) -> set[str]:
    """
    Mirrors train_model()'s real feature selection: only columns named by
    encode_categorical_features / scale_features become training features
    (all_feature_columns = categorical + numeric_to_scale, and
    X_train = train_df[all_feature_columns] with remainder="drop").
    """
    feats: set[str] = set()
    for s in steps:
        if s.tool_name in _FEATURE_SELECTING:
            cols = s.arguments.get("columns")
            if isinstance(cols, list):
                feats.update(c for c in cols if isinstance(c, str) and c)
    return feats


def main():
    df = pd.read_csv(io.BytesIO(DATASET.read_bytes()))
    assert df.shape == (891, 12)
    store = InMemoryDatasetStore()
    store.save("titanic", df)

    ctx_result = build_sanitized_llm_context("titanic", TARGET, store)
    assert ctx_result.success
    context, _ = apply_context_budget(ctx_result.data)

    print("Dataset evidence (from the SAME SanitizedLLMContext the planner saw):")
    for c in context.column_contexts:
        if c.missing_percentage > 0:
            print(f"   {c.name:14s} missing={c.missing_percentage:6.2f}%")
    print()

    old_by_trial = {}
    if OLD_REPLAY.exists():
        for row in json.loads(OLD_REPLAY.read_text()):
            old_by_trial[row["trial"]] = row

    raw = json.loads(RAW.read_text())
    rows = []
    old_material_total = 0
    new_material_total = 0

    for model, trials in raw["blocks"].items():
        for trial in trials:
            attempt = trial["attempts"][-1]
            steps = [_Step(s["tool_name"], s["arguments"]) for s in attempt["tool_arguments"]]
            result = evaluate_plan_adequacy(context, steps, TARGET)

            features = _effective_feature_set(steps)
            material = result.material_findings
            advisory = [f for f in result.findings
                        if f.severity == "advisory" and f.status == "NOT_ADDRESSED"]
            not_applicable = [f for f in result.findings if f.status == "NOT_APPLICABLE"]

            old = old_by_trial.get(trial["trial_id"], {})
            old_status = old.get("adequacy")
            old_material_cols = old.get("material_columns", [])
            old_material_total += len(old_material_cols)
            new_material_total += len({c for f in material for c in f.columns})

            # --- HARD ASSERTION 3: unchanged behavior ------------------
            # Any missing column that IS an effective feature and is neither
            # imputed nor dropped MUST still be material.
            addressed = set()
            for s in steps:
                if s.tool_name in ("impute_missing_values", "drop_column"):
                    col = s.arguments.get("column")
                    if isinstance(col, str) and col:
                        addressed.add(col)
            for c in context.column_contexts:
                if (c.name != TARGET and c.missing_percentage > 0
                        and c.name in features and c.name not in addressed):
                    mat_cols = {x for f in material for x in f.columns}
                    assert c.name in mat_cols, (
                        f"REGRESSION: '{c.name}' is an effective feature with "
                        f"{c.missing_percentage}% missing and is unaddressed, but is "
                        f"not material in {trial['trial_id']}"
                    )

            # --- HARD ASSERTION 2: new behavior for non-feature columns -
            for c in context.column_contexts:
                if (c.name != TARGET and c.missing_percentage > 0
                        and c.name not in features and c.name not in addressed):
                    f = next((x for x in result.findings
                              if x.condition == "missing_values" and c.name in x.columns), None)
                    assert f is not None, f"missing finding for {c.name}"
                    assert f.severity == "advisory", (
                        f"'{c.name}' is NOT an effective feature in {trial['trial_id']} "
                        f"but was severity={f.severity} (expected advisory)"
                    )

            # --- HARD ASSERTION 4: no PASS -> FAIL regression -----------
            if old_status == "PASS":
                assert result.status == "PASS", (
                    f"REGRESSION: {trial['trial_id']} previously PASSED adequacy but now "
                    f"FAILS under effective-feature semantics"
                )

            print("=" * 78)
            print(f"{trial['trial_id']}  steps={len(steps)}  OLD={old_status}  NEW={result.status}"
                  f"  {'(changed)' if old_status != result.status else '(unchanged)'}")
            print("=" * 78)
            print(f"  effective feature set: {sorted(features)}")
            for s in steps:
                print(f"    - {s.tool_name}({s.arguments})")
            print(f"  OLD material columns: {old_material_cols}")
            print(f"  NEW material:  {[(f.condition, f.columns) for f in material] or 'none'}")
            print(f"  NEW advisory:  {[(f.condition, f.columns) for f in advisory] or 'none'}")
            print(f"  NOT_APPLICABLE: {[f.condition for f in not_applicable] or 'none'}")
            print()

            rows.append({
                "model": model,
                "trial": trial["trial_id"],
                "steps": len(steps),
                "effective_feature_set": sorted(features),
                "previous_adequacy_status": old_status,
                "adequacy_status": result.status,
                "changed": old_status != result.status,
                "material_findings": [
                    {"condition": f.condition, "columns": f.columns, "evidence": f.evidence}
                    for f in material
                ],
                "advisory_findings": [
                    {"condition": f.condition, "columns": f.columns, "evidence": f.evidence}
                    for f in advisory
                ],
                "not_applicable_findings": [f.condition for f in not_applicable],
                "old_material_columns": old_material_cols,
            })

    # --- HARD ASSERTION 1 + 2 on the specific documented Cabin case ----
    cabin_rows = [r for r in rows if "Cabin" in r["old_material_columns"]]
    assert cabin_rows, "expected the historical baseline to show Cabin as material"
    for r in cabin_rows:
        assert "Cabin" not in r["effective_feature_set"], (
            f"{r['trial']}: Cabin unexpectedly IS an effective feature"
        )
        assert "Cabin" not in {c for f in r["material_findings"] for c in f["columns"]}, (
            f"{r['trial']}: Cabin must no longer be material"
        )
        assert "Cabin" in {c for f in r["advisory_findings"] for c in f["columns"]}, (
            f"{r['trial']}: Cabin must now be advisory"
        )

    changed = [r for r in rows if r["changed"]]
    now_pass = [r for r in rows if r["adequacy_status"] == "PASS"]
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'trial':24s} {'OLD':>6s} {'NEW':>6s}  material now")
    for r in rows:
        print(f"{r['trial']:24s} {str(r['previous_adequacy_status']):>6s} {r['adequacy_status']:>6s}  "
              f"{sorted({c for f in r['material_findings'] for c in f['columns']})}")
    print()
    print(f"OLD material finding count (recorded): {old_material_total}")
    print(f"NEW material finding count:            {new_material_total}")
    print(f"FALSE-POSITIVE FINDINGS REMOVED:       {old_material_total - new_material_total}")
    print(f"Plans changed status: {len(changed)}/6   now PASS: {len(now_pass)}/6")
    print("ALL HARD ASSERTIONS PASSED")

    OUT.write_text(json.dumps({
        "old_material_finding_count": old_material_total,
        "new_material_finding_count": new_material_total,
        "false_positive_findings_removed": old_material_total - new_material_total,
        "plans_changed_status": len(changed),
        "plans_now_passing": len(now_pass),
        "trials": rows,
    }, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
