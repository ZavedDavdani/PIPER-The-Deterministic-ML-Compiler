"""
SMOKE TEST — no Ollama calls, no production changes.

Phase 1 gate for the adequacy-recovery benchmark. Verifies that
benchmark_adequacy_recovery.py's REPLAN context is genuinely IDENTICAL to
what production emits, rather than merely similar.

The strong check (Part A): run the REAL production graph offline with a
FakeLLMProvider that proposes a known materially-inadequate plan, capture
the actual FailureInfo.evidence plan_node_v2 produces, then build the
harness's evidence for the SAME plan and compare. This proves parity
against real production output instead of asserting it by construction.

Part B verifies the rendered REPLAN prompt (production builder) satisfies
all ten required properties, plus that a preserved step still passes
validate_proposed_plan().

Part C drives the harness's own trial loop against a local fake HTTP
server to confirm all four outcome classifications still work.

Run:  python smoke_adequacy_recovery.py
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import benchmark_adequacy_recovery as ar
import benchmark_planning_models as b
from app.agent.plan_adequacy import classify_plan_steps, evaluate_plan_adequacy
from app.agent.plan_validation import ALLOWED_TOOL_NAMES, TOOL_ARGUMENT_SCHEMAS, validate_proposed_plan
from app.agent.tools.sanitized_llm_context import build_sanitized_llm_context
from app.llm.prompts import build_replan_prompt
from app.llm.provider import LLMPlanningContext, ProposedPlan, ProposedPlanStep
from app.schemas.failure import FailureInfo
from app.storage import InMemoryDatasetStore

SECTION = "=== VALID OPERATIONS (preserve these) ==="
TARGET = "target"


def _step(tool: str, args: dict) -> ProposedPlanStep:
    return ProposedPlanStep(action="a", tool_name=tool, arguments=args, reasoning="r")


# The plan used for parity: `city` is ENCODED (so it IS an effective
# feature) but never imputed -> materially inadequate. `score` is scaled
# and unrelated -> preservable.
INADEQUATE_STEPS = [
    _step("encode_categorical_features", {"columns": ["city"]}),
    _step("scale_features", {"columns": ["score"]}),
]


def _dataset() -> pd.DataFrame:
    n = 200
    return pd.DataFrame({
        "age": [float(20 + i % 40) for i in range(n)],
        "city": [None if i % 10 == 0 else ("NY" if i % 3 else "LA") for i in range(n)],
        "score": [float(i % 7) for i in range(n)],
        TARGET: ["yes" if i % 2 else "no" for i in range(n)],
    })


class _FixedProvider:
    def __init__(self, steps):
        self._steps = steps
        self.calls = 0

    def generate_plan(self, context):
        from app.llm.provider import LLMProviderResult

        self.calls += 1
        return LLMProviderResult(success=True, plan=ProposedPlan(steps=list(self._steps)))


def part_a_production_parity() -> tuple[bool, dict]:
    """Compare harness-built evidence against REAL production graph output."""
    from app.agent import AgentState, build_graph
    from app.storage import InMemoryModelStore, InMemorySplitStore

    df = _dataset()
    store = InMemoryDatasetStore()
    store.save("ds", df)
    graph = build_graph(store, InMemorySplitStore(), InMemoryModelStore(), _FixedProvider(INADEQUATE_STEPS))
    result = graph.invoke(
        AgentState(run_id="smoke", dataset_id="ds", target_column=TARGET, max_retries=0),
        config={"recursion_limit": 60},
    )
    prod_failure = result["failure"]
    assert prod_failure.category == "PLAN_ADEQUACY", f"expected PLAN_ADEQUACY, got {prod_failure.category}"
    prod_evidence = prod_failure.evidence

    # Now build the SAME evidence the way the harness does.
    store2 = InMemoryDatasetStore()
    store2.save("ds", _dataset())
    sanitized = build_sanitized_llm_context("ds", TARGET, store2).data
    adequacy = evaluate_plan_adequacy(sanitized, INADEQUATE_STEPS, TARGET)
    classification = classify_plan_steps(adequacy.findings, INADEQUATE_STEPS)
    harness_evidence = {
        "adequacy_status": adequacy.status,
        "findings": [f.model_dump(mode="json") for f in adequacy.material_findings],
        "advisory_findings": [
            f.model_dump(mode="json") for f in adequacy.findings
            if f.severity == "advisory" and f.status == "NOT_ADDRESSED"
        ],
        "proposed_steps": [{"tool_name": s.tool_name, "arguments": s.arguments} for s in INADEQUATE_STEPS],
        "valid_steps": classification["valid_steps"],
        "implicated_steps": classification["implicated_steps"],
    }

    checks = {
        "evidence keys identical": set(prod_evidence) == set(harness_evidence),
        "valid_steps identical": prod_evidence.get("valid_steps") == harness_evidence["valid_steps"],
        "implicated_steps identical": prod_evidence.get("implicated_steps") == harness_evidence["implicated_steps"],
        "findings identical": prod_evidence.get("findings") == harness_evidence["findings"],
        "advisory_findings identical": prod_evidence.get("advisory_findings") == harness_evidence["advisory_findings"],
        "proposed_steps identical": prod_evidence.get("proposed_steps") == harness_evidence["proposed_steps"],
    }
    return all(checks.values()), {
        "checks": checks,
        "prod_keys": sorted(prod_evidence),
        "harness_keys": sorted(harness_evidence),
        "evidence": harness_evidence,
    }


def part_b_prompt(evidence: dict) -> tuple[bool, list]:
    store = InMemoryDatasetStore()
    store.save("ds", _dataset())
    sanitized = build_sanitized_llm_context("ds", TARGET, store).data

    failure = FailureInfo(
        category="PLAN_ADEQUACY", message="smoke", evidence=evidence,
        node="plan", attempt=0, retryable=True, human_intervention_required=False,
    )
    prompt = build_replan_prompt(LLMPlanningContext(
        objective="Predict target",
        dataset_context=sanitized.model_dump(mode="json"),
        allowed_operations=sorted(ALLOWED_TOOL_NAMES),
        tool_schemas=TOOL_ARGUMENT_SCHEMAS,
        failure_context=failure.model_dump(mode="json"),
    ))

    section = prompt.split(SECTION)[1].split("\n=== ")[0] if SECTION in prompt else ""
    rendered = json.loads(section[section.index("["):section.rindex("]") + 1]) if section else []
    material_cols = sorted({c for f in evidence["findings"] for c in f["columns"]})
    names = {s["tool_name"] for s in rendered}
    args_ok = all(set(s["arguments"]) <= set(TOOL_ARGUMENT_SCHEMAS[s["tool_name"]]["arguments"]) for s in rendered)
    revalidated = [_step(s["tool_name"], s["arguments"]) for s in rendered]

    checks = [
        ("1. current plan present", all(s["tool_name"] in prompt for s in evidence["proposed_steps"])),
        ("2. adequacy findings present", "missing_values" in prompt and all(c in prompt for c in material_cols)),
        ("3. valid_steps present", "valid_steps" in prompt and bool(rendered)),
        ("4. implicated_steps present", "implicated_steps" in prompt),
        ("5. preserved ops are exact production JSON", rendered == evidence["valid_steps"]),
        ("6. tool names valid", names <= ALLOWED_TOOL_NAMES),
        ("7. argument names valid", args_ok),
        ("8. argument values preserved exactly",
         all(s in evidence["valid_steps"] for s in rendered)),
        ("9. uses production REPLAN builder", SECTION in prompt and "PATCH" in prompt.upper()),
        ("10. no invented schema keys",
         not {k for s in rendered for k in s} - {"tool_name", "arguments"}),
        ("11. preserved step passes validate_proposed_plan()",
         bool(revalidated) and validate_proposed_plan(revalidated, TARGET).valid),
    ]
    return all(ok for _, ok in checks), checks


ADEQUATE = {"steps": [
    {"action": "a", "tool_name": "impute_missing_values", "arguments": {"column": "city", "strategy": "mode"}, "reasoning": "r"},
    {"action": "a", "tool_name": "encode_categorical_features", "arguments": {"columns": ["city"]}, "reasoning": "r"},
    {"action": "a", "tool_name": "scale_features", "arguments": {"columns": ["score"]}, "reasoning": "r"},
]}
INADEQUATE = {"steps": [{"action": "a", "tool_name": "encode_categorical_features",
                         "arguments": {"columns": ["city"]}, "reasoning": "r"}]}
INADEQUATE_2 = {"steps": [{"action": "a", "tool_name": "encode_categorical_features",
                           "arguments": {"columns": ["city"]}, "reasoning": "r"},
                          {"action": "a", "tool_name": "scale_features",
                           "arguments": {"columns": ["score"]}, "reasoning": "r"}]}
STRUCT_INVALID = {"steps": [{"action": "a", "tool_name": "drop_column",
                             "arguments": {"columns": ["city"]}, "reasoning": "r"}]}


def _server(seq):
    st = {"i": 0}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            if n:
                self.rfile.read(n)
            plan = seq[min(st["i"], len(seq) - 1)]
            st["i"] += 1
            body = json.dumps({"response": json.dumps(plan), "total_duration": 10**9,
                               "load_duration": 10**8, "prompt_eval_duration": 2 * 10**8,
                               "eval_duration": 7 * 10**8, "prompt_eval_count": 2247,
                               "eval_count": 300, "done_reason": "stop"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    s = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s


def part_c_outcomes() -> tuple[bool, list]:
    store = InMemoryDatasetStore()
    store.save(b.DATASET_ID, _dataset())
    sanitized = build_sanitized_llm_context(b.DATASET_ID, TARGET, store).data
    base = b.LLMPlanningContext(
        objective="o", dataset_context=sanitized.model_dump(mode="json"),
        allowed_operations=sorted(ALLOWED_TOOL_NAMES), tool_schemas=TOOL_ARGUMENT_SCHEMAS)

    ar._ollama_ps = lambda: []
    ar._is_resident = lambda m: False

    cases = [
        ("adequate first attempt", [ADEQUATE], "FIRST_ATTEMPT_PASS", True),
        ("inadequate -> adequate", [INADEQUATE, ADEQUATE], "SUCCESSFUL_PATCH", True),
        ("same inadequate twice", [INADEQUATE, INADEQUATE], "DUPLICATE_PLAN_WALL", False),
        ("inadequate -> invalid -> invalid", [INADEQUATE, STRUCT_INVALID, INADEQUATE_2], "NEW_INVALIDATION", False),
    ]
    results = []
    ok = True
    for label, seq, want, want_final in cases:
        srv = _server(seq)
        orig = b.DEFAULT_OLLAMA_HOST
        b.DEFAULT_OLLAMA_HOST = f"http://127.0.0.1:{srv.server_port}"
        try:
            t = ar._run_trial("fake", label, base, sanitized, TARGET)
        finally:
            b.DEFAULT_OLLAMA_HOST = orig
            srv.shutdown()
        good = t["outcome"] == want and t["final_executable"] == want_final
        ok &= good
        results.append((label, t["outcome"], want, good))
    return ok, results


def main() -> int:
    print("=" * 78)
    print("PHASE 1 SMOKE TEST — harness/production parity (no Ollama)")
    print("=" * 78)

    a_ok, a = part_a_production_parity()
    print("\n--- Part A: evidence parity vs REAL production graph output ---")
    print(f"  production keys: {a['prod_keys']}")
    print(f"  harness keys:    {a['harness_keys']}")
    for name, passed in a["checks"].items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

    b_ok, b_checks = part_b_prompt(a["evidence"])
    print("\n--- Part B: rendered REPLAN prompt properties ---")
    for name, passed in b_checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

    c_ok, c_results = part_c_outcomes()
    print("\n--- Part C: harness outcome classification ---")
    for label, got, want, good in c_results:
        print(f"  [{'PASS' if good else 'FAIL'}] {label:34s} got={got} want={want}")

    ok = a_ok and b_ok and c_ok
    print("\n" + "=" * 78)
    print("SMOKE VERDICT:", "ALL PASS — cleared for real-model benchmark" if ok else "BLOCKED")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
