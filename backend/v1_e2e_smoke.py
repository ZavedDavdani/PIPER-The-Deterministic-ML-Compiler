"""
V1 end-to-end smoke test — ONE real run through the real API with the real
production planner model (qwen3:4b), against the real Titanic fixture.

This is a SMOKE TEST, not a reliability measurement: n=1. The reliability
evidence is benchmark_results/v1_reliability_10/ (2/10). This script exists
to prove the demo path executes end to end, not to estimate a success rate.

Deliberately capped by DEADLINE_SECONDS because PIPER's 600s Ollama timeout
does NOT bound total wall time (urllib's timeout is per-socket-operation) —
observed calls have reached 5.8h and 16.6h. Hitting the cap is itself a
reportable outcome, not a crash.

Run against an already-started backend:  python v1_e2e_smoke.py <base_url>
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8021"
_REPO_ROOT = Path(__file__).parent.parent
CSV = str(_REPO_ROOT / "benchmark_data" / "train.csv")
TARGET = "Survived"
DEADLINE_SECONDS = 1500  # 25 min cap



def _get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=30) as r:
        return json.loads(r.read())


def _post_json(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _upload(path: str) -> dict:
    boundary = "----piperE2E"
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


def main() -> int:
    print(f"[1] health: {_get('/health')}", flush=True)

    ds = _upload(CSV)
    dataset_id = ds["dataset_id"]
    print(f"[2] uploaded {dataset_id} rows={ds['rows']} cols={ds['column_count']} fmt={ds['detected_format']}", flush=True)

    prof = _get(f"/datasets/{dataset_id}")
    print(f"[3] profiled: rows={prof['rows']} cols={prof['columns']}", flush=True)

    run = _post_json("/runs", {"dataset_id": dataset_id, "target_column": TARGET})
    run_id = run["run_id"]
    print(f"[4] run started: {run_id}", flush=True)

    start = time.time()
    last = None
    while True:
        elapsed = time.time() - start
        if elapsed > DEADLINE_SECONDS:
            print(f"[!] DEADLINE {DEADLINE_SECONDS}s reached; last status={last}", flush=True)
            print("RESULT: TIMED_OUT_AT_CAP (known unbounded-planning-latency limitation)", flush=True)
            return 2
        st = _get(f"/runs/{run_id}")
        cur = (st.get("status"), st.get("current_node"), st.get("attempt"))
        if cur != last:
            print(f"    t={int(elapsed):5d}s status={cur[0]} node={cur[1]} attempt={cur[2]}", flush=True)
            last = cur
        if st.get("status") in ("completed", "failed"):
            break
        time.sleep(5)

    print(f"[5] terminal status={last[0]} after {int(time.time()-start)}s", flush=True)

    res = _get(f"/runs/{run_id}/result")
    print("\n=== RESULT ===", flush=True)
    print(f"status:      {res.get('status')}", flush=True)

    comp = res.get("comparison")
    if comp:
        # NOTE: the field is recommended_model_id / selection_metric — see
        # app/schemas/evaluation.py:ModelComparison. There is no
        # best_model_id/best_algorithm.
        winner_id = comp.get("recommended_model_id")
        winner = next(
            (m for m in comp.get("models", []) if m.get("model_id") == winner_id), {}
        )
        print(f"winner:      {winner_id} ({winner.get('algorithm')}) by {comp.get('selection_metric')}", flush=True)
        print(f"justification: {comp.get('justification')}", flush=True)
        for e in comp.get("models", []):
            print(f"   - {e.get('algorithm')}: f1={e.get('f1')}", flush=True)

    val = res.get("validation")
    if val:
        print(f"guardrails:  valid={val.get('valid')} checks={len(val.get('checks') or [])} violations={len(val.get('violations') or [])}", flush=True)

    base = res.get("baseline")
    if base:
        print(f"baseline:    {json.dumps(base)[:200]}", flush=True)

    fail = res.get("failure")
    if fail:
        print(f"failure:     {fail.get('category')} @ {fail.get('node')} attempt={fail.get('attempt')}", flush=True)
        print(f"             {str(fail.get('message'))[:220]}", flush=True)

    summ = _get(f"/runs/{run_id}/summary")
    print(f"summary:     retry_count={summ.get('retry_count')} replanned={summ.get('replanned')}", flush=True)
    tl = _get(f"/runs/{run_id}/timeline")
    print(f"timeline:    replan_count={tl.get('replan_count')} phases={len(tl.get('phases') or [])}", flush=True)

    print(f"\nRESULT: {'SUCCESS' if res.get('status') == 'completed' else 'TERMINAL_FAILURE'}", flush=True)
    return 0 if res.get("status") == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
