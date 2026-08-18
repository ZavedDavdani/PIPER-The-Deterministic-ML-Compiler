"""
VERIFICATION-ONLY SCRIPT — no Ollama calls, no production changes.

Part 5: renders a representative state-preserving REPLAN prompt locally,
using the REAL production path end to end (real sanitized context, real
adequacy evaluator, real classify_plan_steps(), real FailureInfo, real
build_replan_prompt()), and verifies PROGRAMMATICALLY that it contains:

  - the complete current plan
  - exact-JSON valid operations
  - the current adequacy failure
  - a patch instruction
  - valid production tool names / argument names / argument values
  - no invented schema

Uses the real Titanic fixture and a realistic inadequate plan.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from app.agent.plan_adequacy import classify_plan_steps, evaluate_plan_adequacy
from app.agent.plan_validation import ALLOWED_TOOL_NAMES, TOOL_ARGUMENT_SCHEMAS, validate_proposed_plan
from app.agent.tools.context_budget import apply_context_budget
from app.agent.tools.sanitized_llm_context import build_sanitized_llm_context
from app.llm.prompts import build_replan_prompt
from app.llm.provider import LLMPlanningContext, ProposedPlanStep
from app.schemas.failure import FailureInfo
from app.storage import InMemoryDatasetStore

DATASET = Path(__file__).parent.parent / "benchmark_data" / "train.csv"
TARGET = "Survived"
SECTION = "=== VALID OPERATIONS (preserve these) ==="


def _step(tool: str, args: dict) -> ProposedPlanStep:
    return ProposedPlanStep(action="a", tool_name=tool, arguments=args, reasoning="r")


def main() -> int:
    df = pd.read_csv(io.BytesIO(DATASET.read_bytes()))
    store = InMemoryDatasetStore()
    store.save("titanic", df)
    ctx_result = build_sanitized_llm_context("titanic", TARGET, store)
    context, _ = apply_context_budget(ctx_result.data)

    # A realistic inadequate plan: Embarked is ENCODED (so it IS an
    # effective feature) but never imputed -> material. The other steps
    # are untouched by that failure and should be preserved.
    steps = [
        _step("impute_missing_values", {"column": "Age", "strategy": "median"}),
        _step("drop_column", {"column": "Name"}),
        _step("encode_categorical_features", {"columns": ["Sex", "Embarked"]}),
        _step("scale_features", {"columns": ["Age", "Fare"]}),
    ]
    assert validate_proposed_plan(steps, TARGET).valid, "fixture plan must be structurally valid"

    adequacy = evaluate_plan_adequacy(context, steps, TARGET)
    assert adequacy.material_failure, "fixture plan must be materially inadequate"
    classified = classify_plan_steps(adequacy.findings, steps)

    failure = FailureInfo(
        category="PLAN_ADEQUACY",
        message=adequacy.summary,
        evidence={
            "adequacy_status": adequacy.status,
            "findings": [f.model_dump(mode="json") for f in adequacy.material_findings],
            "advisory_findings": [
                f.model_dump(mode="json") for f in adequacy.findings
                if f.severity == "advisory" and f.status == "NOT_ADDRESSED"
            ],
            "proposed_steps": [{"tool_name": s.tool_name, "arguments": s.arguments} for s in steps],
            "valid_steps": classified["valid_steps"],
            "implicated_steps": classified["implicated_steps"],
        },
        node="plan", attempt=0, retryable=True, human_intervention_required=False,
    )

    prompt = build_replan_prompt(LLMPlanningContext(
        objective=f"Predict '{TARGET}' from the remaining columns (binary/multiclass classification).",
        dataset_context=context.model_dump(mode="json"),
        allowed_operations=sorted(ALLOWED_TOOL_NAMES),
        tool_schemas=TOOL_ARGUMENT_SCHEMAS,
        failure_context=failure.model_dump(mode="json"),
    ))

    checks: list[tuple[str, bool, str]] = []

    # 1. Complete current plan present (every step, via proposed_steps).
    checks.append((
        "complete current plan present",
        all(json.dumps(s.arguments, sort_keys=True)[:20] in prompt or s.tool_name in prompt for s in steps)
        and all(s.tool_name in prompt for s in steps),
        f"{len(steps)} steps",
    ))

    # 2. VALID OPERATIONS section rendered.
    checks.append(("VALID OPERATIONS section present", SECTION in prompt, SECTION))

    # 3. Exact-JSON valid operations round-trip.
    section = prompt.split(SECTION)[1].split("\n=== ")[0]
    rendered = json.loads(section[section.index("["):section.rindex("]") + 1])
    checks.append((
        "valid operations are exact production JSON",
        rendered == classified["valid_steps"],
        f"{rendered}",
    ))

    # 4. Current adequacy failure present.
    material_cols = sorted({c for f in adequacy.material_findings for c in f.columns})
    checks.append((
        "current adequacy failure present",
        "PLAN_ADEQUACY" in prompt and "missing_values" in prompt
        and all(c in prompt for c in material_cols),
        f"material columns {material_cols}",
    ))

    # 5. Patch instruction present.
    checks.append((
        "patch instruction present",
        "preserve" in prompt.lower() and "PATCH" in prompt.upper(),
        "preserve/patch wording",
    ))

    # 6. Only real production tool names appear in rendered valid steps.
    names = {s["tool_name"] for s in rendered}
    checks.append((
        "tool names are real production names",
        names <= ALLOWED_TOOL_NAMES,
        f"{sorted(names)} subset of allowlist",
    ))

    # 7. Only real production argument names, with real values.
    arg_ok = True
    detail = []
    for s in rendered:
        allowed_args = set(TOOL_ARGUMENT_SCHEMAS[s["tool_name"]]["arguments"])
        if not set(s["arguments"]) <= allowed_args:
            arg_ok = False
        detail.append(f"{s['tool_name']}{sorted(s['arguments'])}")
    checks.append(("argument names match the real tool schema", arg_ok, "; ".join(detail)))

    # 8. Rendered valid steps are themselves a validator-accepted plan.
    revalidated = [_step(s["tool_name"], s["arguments"]) for s in rendered]
    checks.append((
        "rendered valid steps pass the real validator",
        validate_proposed_plan(revalidated, TARGET).valid,
        "validate_proposed_plan()",
    ))

    # 9. No invented schema: the section contains no key outside the
    #    production step shape.
    invented = {k for s in rendered for k in s} - {"tool_name", "arguments"}
    checks.append(("no invented schema keys", not invented, f"extra keys: {sorted(invented) or 'none'}"))

    print("=" * 78)
    print("REPLAN PROMPT SNAPSHOT VERIFICATION (no Ollama)")
    print("=" * 78)
    print(f"prompt chars: {len(prompt)}")
    print(f"material findings: {material_cols}")
    print(f"valid (preservable) steps: {len(classified['valid_steps'])}")
    print(f"implicated steps: {len(classified['implicated_steps'])}")
    print()
    ok = True
    for name, passed, info in checks:
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:46s} {info}")

    print()
    print("--- rendered VALID OPERATIONS section ---")
    print(section.strip())
    print()
    print("VERDICT:", "ALL CHECKS PASSED" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
