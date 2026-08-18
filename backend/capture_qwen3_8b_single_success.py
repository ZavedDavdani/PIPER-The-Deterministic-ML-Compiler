"""
ONE genuine end-to-end PIPER run with qwen3:8b, with full evidence capture.

This is a DEMONSTRATION harness, not a benchmark. It runs exactly one trial
and stops. It does not retry, does not repair, does not intervene.

It changes NOTHING about PIPER: the model is selected through the existing
production mechanism (the PIPER_LLM_MODEL environment variable read by
OllamaProvider.__init__), the backend is the real FastAPI app, and the run
goes through the real graph. Prompts, validator, adequacy, retry budget,
routing, timeout and keep_alive are all untouched.

Evidence is written to benchmark_results/qwen3_8b_single_success/.

DEADLINE_SECONDS below is THIS SCRIPT's observation window only — it is not
a PIPER setting and does not alter PIPER's own 600s Ollama timeout. It
exists because that timeout is known not to bound total wall time.

Usage: python capture_qwen3_8b_single_success.py <base_url>
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8022"
CSV = r"C:\dev\PIPER\benchmark_data\train.csv"
TARGET = "Survived"
OUT_DIR = r"C:\dev\PIPER\benchmark_results\qwen3_8b_single_success"
DEADLINE_SECONDS = 14400  # 4h observation window (harness-only)

EXPECTED_MODEL = "qwen3:8b"

_events: list = []
_stop = threading.Event()


def _get(path: str, timeout: int = 30):
    with urllib.request.urlopen(f"{BASE}{path}", timeout=timeout) as r:
        return json.loads(r.read())


def _get_raw(path: str, timeout: int = 30):
    try:
        return _get(path, timeout)
    except Exception as e:  # endpoint may 409 until terminal
        return {"_error": str(e)}


def _post_json(path: str, body: dict):
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _upload(path: str):
    boundary = "----piper8B"
    with open(path, "rb") as fh:
        data = fh.read()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="train.csv"\r\n'
        f"Content-Type: text/csv\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{BASE}/datasets",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def _sse_reader(run_id: str):
    """Records every TraceEvent the run emits, for per-attempt evidence."""
    try:
        with urllib.request.urlopen(f"{BASE}/runs/{run_id}/events", timeout=DEADLINE_SECONDS) as r:
            for raw in r:
                if _stop.is_set():
                    return
                line = raw.decode("utf-8", "replace").strip()
                if line.startswith("data: "):
                    try:
                        _events.append(json.loads(line[6:]))
                    except Exception:
                        _events.append({"_unparsed": line[6:]})
    except Exception as e:
        _events.append({"_sse_error": str(e)})


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    started_wall = time.time()

    health = _get("/health")
    print(f"[1] health: {health}", flush=True)

    ds = _upload(CSV)
    dataset_id = ds["dataset_id"]
    print(f"[2] uploaded {dataset_id} rows={ds['rows']} cols={ds['column_count']} fmt={ds['detected_format']}", flush=True)

    profile = _get(f"/datasets/{dataset_id}")
    print(f"[3] profiled rows={profile['rows']} cols={profile['columns']}", flush=True)

    run = _post_json("/runs", {"dataset_id": dataset_id, "target_column": TARGET})
    run_id = run["run_id"]
    print(f"[4] run_id={run_id}  model={EXPECTED_MODEL}", flush=True)

    t = threading.Thread(target=_sse_reader, args=(run_id,), daemon=True)
    t.start()

    last = None
    status = None
    while True:
        elapsed = time.time() - started_wall
        if elapsed > DEADLINE_SECONDS:
            print(f"[!] harness deadline {DEADLINE_SECONDS}s reached; last={last}", flush=True)
            status = "DEADLINE"
            break
        st = _get(f"/runs/{run_id}")
        cur = (st.get("status"), st.get("current_node"), st.get("attempt"))
        if cur != last:
            print(f"    t={int(elapsed):6d}s status={cur[0]} node={cur[1]} attempt={cur[2]}", flush=True)
            last = cur
        if st.get("status") in ("completed", "failed"):
            status = st.get("status")
            break
        time.sleep(5)

    total_elapsed = time.time() - started_wall
    _stop.set()

    result = _get_raw(f"/runs/{run_id}/result")
    summary = _get_raw(f"/runs/{run_id}/summary")
    timeline = _get_raw(f"/runs/{run_id}/timeline")
    explanation = _get_raw(f"/runs/{run_id}/learn/explanation")
    final_status = _get_raw(f"/runs/{run_id}")

    evidence = {
        "demonstration": "single successful end-to-end run attempt (NOT a reliability benchmark)",
        "model": EXPECTED_MODEL,
        "model_selection_mechanism": "PIPER_LLM_MODEL environment variable (existing production config)",
        "dataset": "benchmark_data/train.csv (Titanic, 891x12)",
        "target": TARGET,
        "run_id": run_id,
        "dataset_id": dataset_id,
        "terminal_status": status,
        "total_elapsed_seconds": total_elapsed,
        "production_config": {
            "note": "unchanged; PIPER sends no options/temperature field",
            "timeout_seconds": 600.0,
            "keep_alive": "10m",
            "max_retries_default": 2,
        },
        "dataset_upload": ds,
        "dataset_profile": profile,
        "run_status_final": final_status,
        "result": result,
        "summary": summary,
        "timeline": timeline,
        "learn_explanation": explanation,
        "trace_events": _events,
    }

    out = os.path.join(OUT_DIR, "run_evidence.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(evidence, fh, indent=2, default=str)
    print(f"\n[5] evidence -> {out}", flush=True)

    print("\n=== OUTCOME ===", flush=True)
    print(f"status:   {status}", flush=True)
    print(f"elapsed:  {total_elapsed:.1f}s", flush=True)
    print(f"attempts: retry_count={summary.get('retry_count')} replanned={summary.get('replanned')}", flush=True)

    comp = (result or {}).get("comparison")
    if comp:
        wid = comp.get("recommended_model_id")
        win = next((m for m in comp.get("models", []) if m.get("model_id") == wid), {})
        print(f"winner:   {wid} ({win.get('algorithm')}) by {comp.get('selection_metric')}", flush=True)
        for m in comp.get("models", []):
            print(f"   - {m.get('algorithm')}: f1={m.get('f1')}", flush=True)
    val = (result or {}).get("validation")
    if val:
        print(f"guardrails: valid={val.get('valid')} checks={len(val.get('checks') or [])} violations={len(val.get('violations') or [])}", flush=True)
    fail = (result or {}).get("failure")
    if fail:
        print(f"failure:  {fail.get('category')} @ {fail.get('node')} attempt={fail.get('attempt')}", flush=True)
        print(f"          {str(fail.get('message'))[:300]}", flush=True)

    print(f"\nRESULT: {'SUCCESS' if status == 'completed' else 'NOT_SUCCESS(' + str(status) + ')'}", flush=True)
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
