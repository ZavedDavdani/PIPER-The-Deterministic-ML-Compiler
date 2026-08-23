"""
M5: API tests for the run lifecycle/status/result/SSE endpoints
(app/api/routers/runs.py).

Uses the real FastAPI app, the real agent graph, and the real Telco
CSV — the LLM provider is overridden to heuristic_llm_provider() (see
conftest.py's api_client fixture), never real Ollama.

Note on timing: Starlette's TestClient runs BackgroundTasks
synchronously (in-process, before the HTTP response is considered
complete) — so by the time client.post("/runs", ...) returns here, the
background run has ALREADY finished. This is convenient for most
assertions below (no polling needed) but means the "still running"
(409) and genuinely-concurrent-SSE cases need to be exercised through
a directly pre-seeded run_store instead of a real in-flight run — see
TestRunResultBeforeTermination and TestLiveEventStream's
still-running test for how.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from app.api.dependencies import get_run_store
from app.main import app
from app.schemas.trace_event import TraceEvent
from app.storage import InMemoryRunStore

TELCO_CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "telco_customer_churn.csv"


def _telco_csv_bytes() -> bytes:
    return TELCO_CSV_PATH.read_bytes()


def _upload_telco(api_client) -> str:
    response = api_client.post(
        "/datasets", files={"file": ("telco.csv", _telco_csv_bytes(), "text/csv")}
    )
    return response.json()["dataset_id"]


class TestCreateRun:
    def test_missing_dataset_returns_404(self, api_client):
        response = api_client.post(
            "/runs", json={"dataset_id": "dataset_does_not_exist", "target_column": "Churn"}
        )
        assert response.status_code == 404

    def test_valid_request_returns_202_with_run_id(self, api_client):
        dataset_id = _upload_telco(api_client)

        response = api_client.post(
            "/runs", json={"dataset_id": dataset_id, "target_column": "Churn"}
        )

        assert response.status_code == 202
        body = response.json()
        assert body["run_id"].startswith("run_")
        assert body["status"] == "running"

    def test_max_retries_out_of_bounds_is_rejected(self, api_client):
        dataset_id = _upload_telco(api_client)

        response = api_client.post(
            "/runs", json={"dataset_id": dataset_id, "target_column": "Churn", "max_retries": 999}
        )

        assert response.status_code == 422  # pydantic request validation, le=20

    def test_run_reaches_completed_status_via_the_real_graph(self, api_client):
        dataset_id = _upload_telco(api_client)

        create = api_client.post(
            "/runs", json={"dataset_id": dataset_id, "target_column": "Churn"}
        )
        run_id = create.json()["run_id"]

        status = api_client.get(f"/runs/{run_id}")

        assert status.status_code == 200
        assert status.json()["status"] == "completed"
        assert status.json()["dataset_id"] == dataset_id
        assert status.json()["target_column"] == "Churn"


class TestGetRunStatus:
    def test_missing_run_returns_404(self, api_client):
        response = api_client.get("/runs/run_does_not_exist")
        assert response.status_code == 404


class TestGetRunResult:
    def test_missing_run_returns_404(self, api_client):
        response = api_client.get("/runs/run_does_not_exist/result")
        assert response.status_code == 404

    def test_completed_run_returns_real_validation_and_comparison(self, api_client):
        dataset_id = _upload_telco(api_client)
        create = api_client.post(
            "/runs", json={"dataset_id": dataset_id, "target_column": "Churn"}
        )
        run_id = create.json()["run_id"]

        result = api_client.get(f"/runs/{run_id}/result")

        assert result.status_code == 200
        body = result.json()
        assert body["status"] == "completed"
        assert body["validation"]["valid"] is True
        assert body["comparison"] is not None
        assert body["baseline"] is not None
        assert body["failure"] is None
        assert len(body["evaluation_results"]) >= 1

    def test_failed_run_returns_structured_failure(self, api_client, telco_df):
        """A genuinely leaky dataset drives the real graph to a structured DUPLICATE_PLAN failure."""
        leaky_df = telco_df.copy()
        leaky_df["leaky_dup"] = leaky_df["Churn"]
        import io
        csv_bytes = leaky_df.to_csv(index=False).encode("utf-8")

        upload = api_client.post("/datasets", files={"file": ("leaky.csv", io.BytesIO(csv_bytes), "text/csv")})
        dataset_id = upload.json()["dataset_id"]

        create = api_client.post("/runs", json={"dataset_id": dataset_id, "target_column": "Churn"})
        run_id = create.json()["run_id"]

        result = api_client.get(f"/runs/{run_id}/result")

        assert result.status_code == 200
        body = result.json()
        assert body["status"] == "failed"
        assert body["failure"] is not None
        assert body["failure"]["category"] == "DUPLICATE_PLAN"


class TestGetRunSummary:
    """Pre-6A Polish: GET /runs/{run_id}/summary."""

    def test_missing_run_returns_404(self, api_client):
        response = api_client.get("/runs/run_does_not_exist/summary")
        assert response.status_code == 404

    def test_completed_run_returns_a_real_summary(self, api_client):
        dataset_id = _upload_telco(api_client)
        create = api_client.post("/runs", json={"dataset_id": dataset_id, "target_column": "Churn"})
        run_id = create.json()["run_id"]

        response = api_client.get(f"/runs/{run_id}/summary")

        assert response.status_code == 200
        body = response.json()
        assert body["run_id"] == run_id
        assert body["status"] == "completed"
        assert body["retry_count"] == 0
        assert body["replanned"] is False
        assert len(body["candidate_models"]) == 2
        assert body["winning_model_id"] is not None
        assert body["selection_justification"]
        assert len(body["operations_executed"]) >= 1
        assert body["guardrail_valid"] is True

    def test_result_is_409_while_run_is_still_in_progress(self, api_client):
        class _RunningState:
            dataset_id = "d1"
            target_column = "t"
            status = "running"
            retry_count = 0
            plan_history: list = []

        run_store = InMemoryRunStore()
        run_store.create("run_inflight_summary", _RunningState())
        app.dependency_overrides[get_run_store] = lambda: run_store

        try:
            response = api_client.get("/runs/run_inflight_summary/summary")
            assert response.status_code == 409
        finally:
            app.dependency_overrides.pop(get_run_store, None)


class TestGetRunTimeline:
    """Pre-6A Polish: GET /runs/{run_id}/timeline."""

    def test_missing_run_returns_404(self, api_client):
        response = api_client.get("/runs/run_does_not_exist/timeline")
        assert response.status_code == 404

    def test_completed_run_returns_a_real_timeline(self, api_client):
        dataset_id = _upload_telco(api_client)
        create = api_client.post("/runs", json={"dataset_id": dataset_id, "target_column": "Churn"})
        run_id = create.json()["run_id"]

        response = api_client.get(f"/runs/{run_id}/timeline")

        assert response.status_code == 200
        body = response.json()
        assert body["run_id"] == run_id
        assert body["final_status"] == "completed"
        assert body["replan_count"] == 0
        phases = [p["phase"] for p in body["phases"]]
        assert "Profile" in phases
        assert phases[-1] == "Complete"

    def test_available_while_run_is_still_in_progress_unlike_result_and_summary(self, api_client):
        """Unlike /result and /summary, /timeline is not gated on
        terminal status — it reflects whatever phases have completed
        so far, exactly like the live SSE /events feed it reads from."""

        class _RunningState:
            dataset_id = "d1"
            target_column = "t"
            status = "running"
            retry_count = 0
            plan_history: list = []

        run_store = InMemoryRunStore()
        run_store.create("run_inflight_timeline", _RunningState())
        run_store.append_event(
            "run_inflight_timeline",
            TraceEvent(
                run_id="run_inflight_timeline", step_id="s1", attempt=0, node="profile",
                event_type="node_completed", timestamp="t1", status="success",
            ),
        )
        app.dependency_overrides[get_run_store] = lambda: run_store

        try:
            response = api_client.get("/runs/run_inflight_timeline/timeline")
            assert response.status_code == 200
            body = response.json()
            assert body["final_status"] is None
            assert [p["phase"] for p in body["phases"]] == ["Profile"]
        finally:
            app.dependency_overrides.pop(get_run_store, None)


class TestGetRunLearnExplanation:
    """Batch 6A (PIPER Learn: Learn-Explain): GET /runs/{run_id}/learn/explanation."""

    def test_missing_run_returns_404(self, api_client):
        response = api_client.get("/runs/run_does_not_exist/learn/explanation")
        assert response.status_code == 404

    def test_completed_run_returns_a_real_grounded_explanation(self, api_client):
        dataset_id = _upload_telco(api_client)
        create = api_client.post("/runs", json={"dataset_id": dataset_id, "target_column": "Churn"})
        run_id = create.json()["run_id"]

        response = api_client.get(f"/runs/{run_id}/learn/explanation")

        assert response.status_code == 200
        body = response.json()
        assert body["run_id"] == run_id
        assert body["status"] == "completed"
        assert len(body["preprocessing"]) >= 1
        assert body["model_selection"] is not None
        assert body["model_selection"]["justification"]
        assert len(body["evaluation"]) >= 1
        assert len(body["guardrail_checks"]) >= 1
        assert body["failure"] is None

    def test_result_is_409_while_run_is_still_in_progress(self, api_client):
        class _RunningState:
            dataset_id = "d1"
            target_column = "t"
            status = "running"
            retry_count = 0
            plan_history: list = []

        run_store = InMemoryRunStore()
        run_store.create("run_inflight_explanation", _RunningState())
        app.dependency_overrides[get_run_store] = lambda: run_store

        try:
            response = api_client.get("/runs/run_inflight_explanation/learn/explanation")
            assert response.status_code == 409
        finally:
            app.dependency_overrides.pop(get_run_store, None)


class TestLearnStaticContentEndpoints:
    """Batch 6A: GET /learn/formulas and GET /learn/comprehension-checks."""

    def test_formula_library_endpoint_returns_real_static_content(self, api_client):
        response = api_client.get("/learn/formulas")
        assert response.status_code == 200
        body = response.json()
        names = {entry["name"] for entry in body}
        assert "F1 Score" in names
        assert "Accuracy" in names

    def test_formula_library_endpoint_is_identical_across_calls(self, api_client):
        """Static content: two calls must return byte-identical JSON."""
        first = api_client.get("/learn/formulas").json()
        second = api_client.get("/learn/formulas").json()
        assert first == second

    def test_comprehension_checks_endpoint_returns_real_static_content(self, api_client):
        response = api_client.get("/learn/comprehension-checks")
        assert response.status_code == 200
        body = response.json()
        assert len(body) >= 5
        assert all(entry["question"].endswith("?") for entry in body)


class TestExploration:
    """Batch 6B (PIPER Learn: Learn-Explore): POST/GET /runs/{run_id}/explore."""

    def _completed_run_with_model_ids(self, api_client):
        dataset_id = _upload_telco(api_client)
        create = api_client.post("/runs", json={"dataset_id": dataset_id, "target_column": "Churn"})
        run_id = create.json()["run_id"]
        result = api_client.get(f"/runs/{run_id}/result").json()
        model_ids_by_algorithm = {m["algorithm"]: m["model_id"] for m in result["model_results"]}
        return run_id, model_ids_by_algorithm

    def test_missing_run_returns_404(self, api_client):
        response = api_client.post(
            "/runs/run_does_not_exist/explore",
            json={"base_model_id": "model_x", "new_algorithm": "logistic_regression"},
        )
        assert response.status_code == 404

    def test_model_swap_exploration_returns_a_real_isolated_result(self, api_client):
        run_id, model_ids_by_algorithm = self._completed_run_with_model_ids(api_client)
        base_model_id = model_ids_by_algorithm["random_forest"]

        response = api_client.post(
            f"/runs/{run_id}/explore",
            json={"base_model_id": base_model_id, "new_algorithm": "logistic_regression"},
        )

        assert response.status_code == 201
        body = response.json()
        assert body["run_id"] == run_id
        assert body["base_model_id"] == base_model_id
        assert body["training"]["algorithm"] == "logistic_regression"
        assert body["training"]["model_id"] != base_model_id
        assert body["variable_changed"]["kind"] == "model"
        assert body["evaluation_explanation"] is not None

    def test_hyperparameter_exploration_returns_a_real_result(self, api_client):
        run_id, model_ids_by_algorithm = self._completed_run_with_model_ids(api_client)
        base_model_id = model_ids_by_algorithm["random_forest"]

        response = api_client.post(
            f"/runs/{run_id}/explore",
            json={"base_model_id": base_model_id, "hyperparameter_name": "n_estimators", "hyperparameter_value": 300},
        )

        assert response.status_code == 201
        body = response.json()
        assert body["variable_changed"]["kind"] == "hyperparameter"
        assert body["training"]["parameters"]["n_estimators"] == 300

    def test_both_variables_provided_returns_400(self, api_client):
        run_id, model_ids_by_algorithm = self._completed_run_with_model_ids(api_client)
        base_model_id = model_ids_by_algorithm["random_forest"]

        response = api_client.post(
            f"/runs/{run_id}/explore",
            json={
                "base_model_id": base_model_id, "new_algorithm": "logistic_regression",
                "hyperparameter_name": "n_estimators", "hyperparameter_value": 100,
            },
        )
        assert response.status_code == 400

    def test_model_id_not_from_this_run_returns_400(self, api_client):
        run_id, _ = self._completed_run_with_model_ids(api_client)

        response = api_client.post(
            f"/runs/{run_id}/explore",
            json={"base_model_id": "model_unrelated", "new_algorithm": "logistic_regression"},
        )
        assert response.status_code == 400

    def test_get_single_exploration_round_trips(self, api_client):
        run_id, model_ids_by_algorithm = self._completed_run_with_model_ids(api_client)
        base_model_id = model_ids_by_algorithm["random_forest"]
        created = api_client.post(
            f"/runs/{run_id}/explore",
            json={"base_model_id": base_model_id, "new_algorithm": "logistic_regression"},
        ).json()

        response = api_client.get(f"/runs/{run_id}/explore/{created['experiment_id']}")

        assert response.status_code == 200
        assert response.json()["experiment_id"] == created["experiment_id"]

    def test_get_single_exploration_missing_returns_404(self, api_client):
        run_id, _ = self._completed_run_with_model_ids(api_client)
        response = api_client.get(f"/runs/{run_id}/explore/exp_does_not_exist")
        assert response.status_code == 404

    def test_list_explorations_for_a_run(self, api_client):
        run_id, model_ids_by_algorithm = self._completed_run_with_model_ids(api_client)
        base_model_id = model_ids_by_algorithm["random_forest"]

        assert api_client.get(f"/runs/{run_id}/explore").json() == []

        api_client.post(
            f"/runs/{run_id}/explore",
            json={"base_model_id": base_model_id, "new_algorithm": "logistic_regression"},
        )

        listed = api_client.get(f"/runs/{run_id}/explore").json()
        assert len(listed) == 1
        assert listed[0]["run_id"] == run_id

    def test_exploring_does_not_change_the_original_run_result(self, api_client):
        run_id, model_ids_by_algorithm = self._completed_run_with_model_ids(api_client)
        base_model_id = model_ids_by_algorithm["random_forest"]

        before = api_client.get(f"/runs/{run_id}/result").json()

        api_client.post(
            f"/runs/{run_id}/explore",
            json={"base_model_id": base_model_id, "new_algorithm": "logistic_regression"},
        )

        after = api_client.get(f"/runs/{run_id}/result").json()
        assert before == after


class TestRunResultBeforeTermination:
    def test_result_is_409_while_run_is_still_in_progress(self, api_client):
        """
        Directly pre-seeds run_store with a "running" record (rather
        than going through a real POST /runs, which TestClient would
        run to completion before returning — see module docstring) to
        exercise the 409 branch specifically.
        """

        class _RunningState:
            dataset_id = "d1"
            target_column = "t"
            status = "running"
            retry_count = 0
            plan_history: list = []

        run_store = InMemoryRunStore()
        run_store.create("run_inflight", _RunningState())
        app.dependency_overrides[get_run_store] = lambda: run_store

        try:
            response = api_client.get("/runs/run_inflight/result")
            assert response.status_code == 409
        finally:
            app.dependency_overrides.pop(get_run_store, None)


class TestLiveEventStream:
    def test_missing_run_returns_404(self, api_client):
        response = api_client.get("/runs/run_does_not_exist/events")
        assert response.status_code == 404

    def test_completed_run_streams_well_formed_sse_events(self, api_client):
        dataset_id = _upload_telco(api_client)
        create = api_client.post(
            "/runs", json={"dataset_id": dataset_id, "target_column": "Churn"}
        )
        run_id = create.json()["run_id"]

        with api_client.stream("GET", f"/runs/{run_id}/events") as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
            lines = [line for line in response.iter_lines() if line]

        assert len(lines) > 0
        parsed_events = []
        for line in lines:
            assert line.startswith("data: ")
            payload = line[len("data: "):]
            event = TraceEvent.model_validate_json(payload)
            assert event.run_id == run_id
            parsed_events.append(event)
        assert parsed_events[-1].event_type == "run_completed"

    def test_stream_waits_for_a_still_running_run_before_closing(self, api_client):
        """
        Proves the SSE endpoint genuinely polls/waits rather than
        immediately closing on a non-terminal run: seeds a "running"
        record with one pre-existing event, flips it to "completed"
        from a background thread after a short delay, and asserts the
        stream only closes AFTER that delay has elapsed.
        """

        class _RunningState:
            dataset_id = "d1"
            target_column = "t"
            status = "running"
            retry_count = 0
            plan_history: list = []

        class _CompletedState:
            dataset_id = "d1"
            target_column = "t"
            status = "completed"
            retry_count = 0
            plan_history: list = []

        run_store = InMemoryRunStore()
        run_store.create("run_live", _RunningState())
        run_store.append_event(
            "run_live",
            TraceEvent(
                run_id="run_live", step_id="s1", attempt=0, node="profile",
                event_type="node_started", timestamp="t", status="success",
            ),
        )
        app.dependency_overrides[get_run_store] = lambda: run_store

        def flip_to_completed_after_delay():
            time.sleep(0.5)
            run_store.update("run_live", _CompletedState())

        try:
            thread = threading.Thread(target=flip_to_completed_after_delay)
            start = time.perf_counter()
            thread.start()
            with api_client.stream("GET", "/runs/run_live/events") as response:
                lines = [line for line in response.iter_lines() if line]
            elapsed = time.perf_counter() - start
            thread.join()
        finally:
            app.dependency_overrides.pop(get_run_store, None)

        assert elapsed >= 0.5
        assert len(lines) == 1


class TestProductizationEndpoints:
    """V1.2 Batch 1: decision-trace / verdict / intervention / evidence."""

    def test_decision_trace_missing_run_returns_404(self, api_client):
        response = api_client.get("/runs/run_does_not_exist/decision-trace")
        assert response.status_code == 404

    def test_completed_run_returns_decision_trace_verdict_and_evidence(self, api_client):
        dataset_id = _upload_telco(api_client)
        create = api_client.post("/runs", json={"dataset_id": dataset_id, "target_column": "Churn"})
        run_id = create.json()["run_id"]

        trace = api_client.get(f"/runs/{run_id}/decision-trace")
        assert trace.status_code == 200
        body = trace.json()
        assert body["run_id"] == run_id
        ids = [s["id"] for s in body["stages"]]
        assert ids == [
            "LLM_PROPOSED", "VALIDATED", "ADEQUACY", "REPLAN",
            "EXECUTION", "TRAINING", "EVALUATION", "GUARDRAILS", "FINAL_VERDICT",
        ]
        blob = trace.text.lower()
        assert '"reasoning"' not in blob

        verdict = api_client.get(f"/runs/{run_id}/verdict")
        assert verdict.status_code == 200
        assert verdict.json()["run_id"] == run_id
        assert verdict.json()["outcome"] in {"ACCEPTED", "REJECTED", "HUMAN_INTERVENTION_REQUIRED"}

        intervention = api_client.get(f"/runs/{run_id}/intervention")
        assert intervention.status_code == 200

        evidence = api_client.get(f"/runs/{run_id}/evidence")
        assert evidence.status_code == 200
        assert evidence.json()["schema_version"] == "piper.evidence.v1"
        assert '"reasoning"' not in evidence.text.lower()

    def test_verdict_and_evidence_are_409_while_run_is_in_progress(self, api_client):
        class _RunningState:
            dataset_id = "d1"
            target_column = "t"
            status = "running"
            retry_count = 0
            plan_history: list = []
            planning_attempts: list = []

        run_store = InMemoryRunStore()
        run_store.create("run_inflight_product", _RunningState())
        app.dependency_overrides[get_run_store] = lambda: run_store

        try:
            assert api_client.get("/runs/run_inflight_product/decision-trace").status_code == 200
            assert api_client.get("/runs/run_inflight_product/verdict").status_code == 409
            assert api_client.get("/runs/run_inflight_product/intervention").status_code == 409
            assert api_client.get("/runs/run_inflight_product/evidence").status_code == 409
            assert api_client.get("/runs/run_inflight_product/replay").status_code == 409
        finally:
            app.dependency_overrides.pop(get_run_store, None)


class TestRunHistoryAndReplay:
    def test_list_runs_includes_completed_run(self, api_client):
        dataset_id = _upload_telco(api_client)
        create = api_client.post("/runs", json={"dataset_id": dataset_id, "target_column": "Churn"})
        run_id = create.json()["run_id"]

        listed = api_client.get("/runs")
        assert listed.status_code == 200
        runs = listed.json()["runs"]
        assert any(item["run_id"] == run_id for item in runs)

    def test_replay_rebuilds_evidence_without_llm(self, api_client):
        dataset_id = _upload_telco(api_client)
        create = api_client.post("/runs", json={"dataset_id": dataset_id, "target_column": "Churn"})
        run_id = create.json()["run_id"]

        replay = api_client.get(f"/runs/{run_id}/replay")
        assert replay.status_code == 200
        body = replay.json()
        assert body["llm_invoked"] is False
        assert body["source"] == "persisted_events_and_state"
        assert body["run_id"] == run_id
        assert body["evidence"]["schema_version"] == "piper.evidence.v1"
        assert '"reasoning"' not in replay.text.lower()


class TestOllamaSettings:
    def test_get_ollama_status_does_not_require_a_live_server(self, api_client):
        response = api_client.get("/settings/ollama")
        assert response.status_code == 200
        body = response.json()
        assert "host" in body
        assert "model" in body
        assert "reachable" in body
        assert "models" in body

    def test_put_model_does_not_accept_temperature(self, api_client):
        response = api_client.put(
            "/settings/ollama",
            json={"model": "qwen3:4b", "temperature": 0.0},
        )
        assert response.status_code == 422
