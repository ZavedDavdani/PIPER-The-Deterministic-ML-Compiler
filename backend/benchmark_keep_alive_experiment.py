"""
MEASUREMENT-ONLY SCRIPT — isolated from production code. Phase 2A of
the engineering-hardening investigation (see CLAUDE.md's Phase 1
findings). Tests ONE variable: Ollama's `keep_alive` request
parameter. Nothing else changes.

Uses the exact CURRENT production prompt (post planner-contract-fix,
tool_schemas included) built ONCE and reused byte-identical across
every single trial in this script — prompt content is never a variable
here, only model residency state (COLD vs WARM) and whether
`keep_alive` was set.

Sequence (6 real Ollama calls total):

  Group A — current/default behavior (no keep_alive key sent at all,
  so Ollama applies its own documented 5-minute default):
    A1: forced COLD (ollama stop beforehand) -> baseline cold cost
    A2: immediate call right after A1 (same prompt) -> short-gap warm?
    A3: forced COLD again -> fresh cold baseline for the gap test
    [wait ~330s, past the 5-minute default]
    A4: call after the gap -> does default survive a realistic gap?

  Group B — explicit long keep_alive ("30m"):
    B1: forced COLD, keep_alive="30m" -> cold cost (should match A1/A3)
    [wait ~330s, the SAME gap duration as A3->A4]
    B2: call after the gap, keep_alive="30m" -> does explicit config
        survive the same gap that (per A3->A4) may defeat the default?

`ollama ps` is polled immediately before A2/A4/B2 to get GROUND TRUTH
on residency (not just inferred from latency after the fact).
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.request

import benchmark_planning_models as b

MODEL = "qwen3:4b"
GAP_SECONDS = 330  # ~5.5 min, past Ollama's documented 5-minute default keep_alive
LONG_KEEP_ALIVE = "30m"
RESULTS_PATH = "keep_alive_experiment_results.json"


def _ollama_stop():
    subprocess.run(["ollama", "stop", MODEL], capture_output=True, text=True, timeout=30)
    time.sleep(1)


def _ollama_ps() -> dict:
    proc = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=30)
    lines = proc.stdout.strip().splitlines()
    for line in lines[1:]:
        if line.strip().startswith(MODEL):
            parts = line.split()
            return {"resident": True, "raw": line.strip()}
    return {"resident": False, "raw": proc.stdout.strip()}


def _call(prompt: str, keep_alive: str | None, label: str) -> dict:
    payload = {"model": MODEL, "prompt": prompt, "stream": False, "format": b.PLAN_JSON_SCHEMA}
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive

    request_body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url=f"{b.DEFAULT_OLLAMA_HOST.rstrip('/')}/api/generate",
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    print(f"  [{label}] sending request (keep_alive={keep_alive!r})...", flush=True)
    t_start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=b.DEFAULT_TIMEOUT_SECONDS) as response:
        raw_bytes = response.read()
    t_end = time.perf_counter()
    wall = t_end - t_start

    body = json.loads(raw_bytes.decode("utf-8"))
    content = b._extract_content(body)
    cleaned = b._strip_markdown_fences(content) if content else None
    plan_json = json.loads(cleaned) if cleaned else None
    plan = b.ProposedPlan.model_validate(plan_json) if plan_json else None
    validation = b.validate_proposed_plan(plan.steps, b.TARGET_COLUMN) if plan else None

    record = {
        "label": label,
        "keep_alive_sent": keep_alive,
        "wall_seconds": wall,
        "load_duration_s": body.get("load_duration", 0) / 1e9,
        "prompt_eval_duration_s": body.get("prompt_eval_duration", 0) / 1e9,
        "eval_duration_s": body.get("eval_duration", 0) / 1e9,
        "total_duration_s": body.get("total_duration", 0) / 1e9,
        "prompt_eval_count": body.get("prompt_eval_count"),
        "eval_count": body.get("eval_count"),
        "tokens_per_sec": (body.get("eval_count") or 0) / (body.get("eval_duration", 1) / 1e9)
        if body.get("eval_duration") else None,
        "plan_valid": validation.valid if validation else None,
        "step_count": len(plan.steps) if plan else 0,
    }
    print(
        f"    -> wall={wall:.1f}s load={record['load_duration_s']:.2f}s "
        f"prompt_eval={record['prompt_eval_duration_s']:.2f}s eval={record['eval_duration_s']:.2f}s "
        f"valid={record['plan_valid']}",
        flush=True,
    )
    return record


def main():
    store, df = b._load_dataset()
    assert df.shape == (891, 12)
    sanitized_result = b.build_sanitized_llm_context(b.DATASET_ID, b.TARGET_COLUMN, store)
    assert sanitized_result.success
    budgeted_context, _ = b.apply_context_budget(sanitized_result.data)
    context = b.LLMPlanningContext(
        objective=f"Predict '{b.TARGET_COLUMN}' from the remaining columns (binary/multiclass classification).",
        dataset_context=budgeted_context.model_dump(mode="json"),
        allowed_operations=sorted(b.ALLOWED_TOOL_NAMES),
        tool_schemas=b.TOOL_ARGUMENT_SCHEMAS,
    )
    prompt = b.build_planning_prompt(context)
    print(f"Fixed prompt: {len(prompt)} chars — reused byte-identical for every trial below.\n")

    results = {"prompt_chars": len(prompt), "gap_seconds": GAP_SECONDS, "trials": []}

    print("=== Group A: default (no explicit keep_alive) ===")
    print("-- forcing COLD --")
    _ollama_stop()
    print(f"ollama ps before A1: {_ollama_ps()}")
    results["trials"].append(_call(prompt, None, "A1_cold_default"))

    print("-- immediate follow-up (no wait) --")
    print(f"ollama ps before A2: {_ollama_ps()}")
    results["trials"].append(_call(prompt, None, "A2_immediate_default"))

    print("-- forcing COLD again for the gap test --")
    _ollama_stop()
    print(f"ollama ps before A3: {_ollama_ps()}")
    results["trials"].append(_call(prompt, None, "A3_cold_default"))

    print(f"-- waiting {GAP_SECONDS}s (past Ollama's 5-minute default) --")
    time.sleep(GAP_SECONDS)
    ps_before_a4 = _ollama_ps()
    print(f"ollama ps before A4 (after {GAP_SECONDS}s gap, default keep_alive): {ps_before_a4}")
    results["ps_before_A4"] = ps_before_a4
    results["trials"].append(_call(prompt, None, "A4_after_gap_default"))

    print("\n=== Group B: explicit long keep_alive ===")
    print("-- forcing COLD --")
    _ollama_stop()
    print(f"ollama ps before B1: {_ollama_ps()}")
    results["trials"].append(_call(prompt, LONG_KEEP_ALIVE, "B1_cold_explicit"))

    print(f"-- waiting {GAP_SECONDS}s (same gap as A3->A4) --")
    time.sleep(GAP_SECONDS)
    ps_before_b2 = _ollama_ps()
    print(f"ollama ps before B2 (after {GAP_SECONDS}s gap, explicit keep_alive={LONG_KEEP_ALIVE}): {ps_before_b2}")
    results["ps_before_B2"] = ps_before_b2
    results["trials"].append(_call(prompt, LONG_KEEP_ALIVE, "B2_after_gap_explicit"))

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
