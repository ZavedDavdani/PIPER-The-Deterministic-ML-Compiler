"""Phase 5 — standalone inference, Test Flight, deployment package."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.agent.state import AgentState
from app.agent.tools.training import train_model
from app.artifacts.publisher import publish_run_artifacts
from app.deployment.csv_io import parse_unseen_csv, predictions_csv
from app.deployment.errors import InferenceError
from app.deployment.package import write_deployment_package
from app.deployment.predict import predict_unseen
from app.deployment.readiness import check_deployment_readiness
from app.deployment.schema import validate_inference_frame
from app.schemas.evaluation import ModelComparison, ModelComparisonEntry
from app.schemas.guardrails import PipelineValidationResult
from app.schemas.training import FeatureEngineeringIntent
from app.storage.model_store import InMemoryModelStore
from app.storage.run_store import InMemoryRunStore
from app.storage.split_store import InMemorySplitStore


def _split_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame(
        {
            "cat": rng.choice(["a", "b"], size=n),
            "num": rng.normal(size=n),
            "label": rng.choice(["yes", "no"], size=n),
        }
    )
    df.loc[:39, "label"] = "yes"
    df.loc[40:, "label"] = "no"
    return df.iloc[:64].reset_index(drop=True), df.iloc[64:].reset_index(drop=True)


def _train_winner(split_store: InMemorySplitStore, model_store: InMemoryModelStore) -> str:
    train_df, test_df = _split_frames()
    split_store.save("split_art", train_df, test_df)
    intent = FeatureEngineeringIntent(
        categorical_columns=["cat"],
        numeric_columns_to_scale=["num"],
    )
    result = train_model(
        "split_art",
        "label",
        "logistic_regression",
        {"C": 1.0, "max_iter": 1000},
        intent,
        split_store,
        model_store,
    )
    assert result.success is True
    return result.data.model_id


def _eligible_state(run_id: str, model_id: str) -> AgentState:
    return AgentState(
        run_id=run_id,
        dataset_id="ds_art",
        target_column="label",
        status="completed",
        split_id="split_art",
        validation=PipelineValidationResult(dataset_id="ds_art", target_column="label", valid=True),
        comparison=ModelComparison(
            models=[
                ModelComparisonEntry(
                    model_id=model_id,
                    algorithm="logistic_regression",
                    accuracy=0.8,
                    precision=0.8,
                    recall=0.8,
                    f1=0.8,
                    roc_auc=0.8,
                )
            ],
            recommended_model_id=model_id,
            justification="logistic_regression selected: F1 0.8.",
        ),
    )


def _seed_run(run_store: InMemoryRunStore, state: AgentState) -> None:
    run_store.create(state.run_id, state, display_dataset_id=state.dataset_id)
    run_store.update(state.run_id, state)


def _publish(tmp_path: Path) -> tuple[str, pd.DataFrame]:
    split_store = InMemorySplitStore()
    model_store = InMemoryModelStore()
    run_store = InMemoryRunStore()
    model_id = _train_winner(split_store, model_store)
    state = _eligible_state("run_fly", model_id)
    _seed_run(run_store, state)
    publish_run_artifacts(
        "run_fly",
        run_store=run_store,
        model_store=model_store,
        split_store=split_store,
        artifact_root=tmp_path,
    )
    unseen = _split_frames()[1][["cat", "num"]].copy()
    return "run_fly", unseen


class TestVerifiedLoad:
    def test_verified_artifact_loads_and_predicts(self, tmp_path: Path):
        run_id, unseen = _publish(tmp_path)
        result = predict_unseen(tmp_path, run_id, unseen)
        assert result["schema_status"] == "valid"
        assert result["row_count"] == len(unseen)
        assert result["parity"]["parity_status"] == "passed"
        assert result["data_kind"] == "NEW_UNSEEN_DATA"
        assert len(result["predictions"]) == len(unseen)

    def test_unverified_artifact_is_rejected(self, tmp_path: Path):
        run_id, unseen = _publish(tmp_path)
        status_path = tmp_path / run_id / "status.json"
        status_path.write_text(
            status_path.read_text(encoding="utf-8").replace("VERIFIED", "FAILED"),
            encoding="utf-8",
        )
        with pytest.raises(InferenceError) as exc:
            predict_unseen(tmp_path, run_id, unseen)
        assert exc.value.code == "artifact_not_verified"

    def test_missing_artifact_is_rejected(self, tmp_path: Path):
        with pytest.raises(InferenceError) as exc:
            predict_unseen(tmp_path, "run_missing", pd.DataFrame({"cat": ["a"], "num": [1.0]}))
        assert exc.value.code in {"artifact_missing", "invalid_run_id"}


class TestSchema:
    def test_missing_feature_rejected(self, tmp_path: Path):
        run_id, unseen = _publish(tmp_path)
        with pytest.raises(InferenceError) as extra:
            predict_unseen(tmp_path, run_id, unseen.drop(columns=["num"]))
        assert extra.value.code == "missing_features"

    def test_invalid_structure_rejected(self):
        with pytest.raises(InferenceError) as extra:
            validate_inference_frame(pd.DataFrame(), ["cat", "num"])
        assert extra.value.code == "invalid_input"

    def test_non_csv_rejected(self):
        with pytest.raises(InferenceError) as extra:
            parse_unseen_csv("notes.txt", b"hello")
        assert extra.value.code == "unsupported_file_type"


class TestCsvAndParity:
    def test_batch_csv_output_does_not_mutate_original(self, tmp_path: Path):
        run_id, unseen = _publish(tmp_path)
        raw = unseen.to_csv(index=False).encode("utf-8")
        parsed = parse_unseen_csv("new_data.csv", raw)
        result = predict_unseen(tmp_path, run_id, parsed)
        out = predictions_csv(parsed, result["predictions"])
        assert raw == unseen.to_csv(index=False).encode("utf-8")
        assert b"prediction" in out
        assert "prediction" not in parsed.columns

    def test_inference_parity_fail_closed(self, tmp_path: Path, monkeypatch):
        run_id, unseen = _publish(tmp_path)
        import joblib as joblib_mod

        real_load = joblib_mod.load
        calls = {"n": 0}

        def counted(path):
            calls["n"] += 1
            obj = real_load(path)
            if calls["n"] >= 2:

                class Flip:
                    def predict(self, X):
                        return np.arange(len(X))

                return Flip()
            return obj

        monkeypatch.setattr(joblib_mod, "load", counted)
        with pytest.raises(InferenceError) as extra:
            predict_unseen(tmp_path, run_id, unseen)
        assert extra.value.code == "inference_parity_failed"

    def test_corrupted_joblib_rejected(self, tmp_path: Path):
        run_id, unseen = _publish(tmp_path)
        path = tmp_path / run_id / "pipeline.joblib"
        path.write_bytes(b"not-a-joblib")
        with pytest.raises(InferenceError) as extra:
            predict_unseen(tmp_path, run_id, unseen)
        assert extra.value.code in {"hashes_mismatch", "pipeline_load_failed"}


class TestNoPlannerOrRetrain:
    def test_source_has_no_llm_or_graph(self):
        root = Path(__file__).resolve().parents[1] / "app" / "deployment"
        for path in root.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            assert "generate_plan" not in text
            assert "build_graph" not in text
            assert "import ollama" not in text
            assert "from ollama" not in text
            assert "import langgraph" not in text

    def test_predict_does_not_retrain(self, tmp_path: Path, monkeypatch):
        run_id, unseen = _publish(tmp_path)

        def _boom(*args, **kwargs):
            raise AssertionError("train_model must not run during inference")

        monkeypatch.setattr("app.agent.tools.training.train_model", _boom)
        predict_unseen(tmp_path, run_id, unseen)


class TestPackage:
    def test_package_and_dockerfile_are_optional_and_standalone(self, tmp_path: Path):
        run_id, unseen = _publish(tmp_path)
        payload = write_deployment_package(tmp_path, run_id)
        assert payload["docker_optional"] is True
        dest = tmp_path / run_id / "deployment_package"
        assert (dest / "inference.py").is_file()
        assert (dest / "Dockerfile").is_file()
        script = (dest / "inference.py").read_text(encoding="utf-8")
        assert "from app" not in script
        assert "langgraph" not in script
        assert "ollama" not in script
        docker = (dest / "Dockerfile").read_text(encoding="utf-8")
        assert "FROM python:3.11-slim" in docker
        assert "redis" not in docker.lower()
        ready = check_deployment_readiness(tmp_path, run_id)
        assert ready["status"] == "READY"
        check_names = {item["check"] for item in ready["checks"]}
        assert "prediction_succeeds" in check_names
        assert "inference_parity" in check_names
        import subprocess
        import sys

        csv_path = tmp_path / "unseen.csv"
        out_path = tmp_path / "preds.csv"
        unseen.to_csv(csv_path, index=False)
        proc = subprocess.run(
            [sys.executable, str(dest / "inference.py"), str(csv_path), "-o", str(out_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0
        scored = pd.read_csv(out_path)
        assert "prediction" in scored.columns
        assert len(scored) == len(unseen)


class TestDeploymentApi:
    def _verified(self, api_client, tmp_path: Path) -> str:
        api_client.app.state.artifact_dir = tmp_path
        csv_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "telco_customer_churn.csv"
        upload = api_client.post("/datasets", files={"file": ("telco.csv", csv_path.read_bytes(), "text/csv")})
        dataset_id = upload.json()["dataset_id"]
        created = api_client.post("/runs", json={"dataset_id": dataset_id, "target_column": "Churn"})
        run_id = created.json()["run_id"]
        assert api_client.get(f"/runs/{run_id}").json()["status"] == "completed"
        generated = api_client.post(f"/runs/{run_id}/artifacts")
        assert generated.status_code == 201, generated.text
        return run_id

    def test_predict_test_flight_package_and_rejections(self, api_client, tmp_path: Path):
        run_id = self._verified(api_client, tmp_path)
        ready = api_client.get(f"/runs/{run_id}/deployment")
        assert ready.status_code == 200
        assert ready.json()["status"] == "READY"
        csv_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "telco_customer_churn.csv"
        frame = pd.read_csv(csv_path).head(5)
        payload = api_client.post("/predict", json={"run_id": run_id, "rows": frame.to_dict(orient="records")})
        assert payload.status_code == 200, payload.text
        body = payload.json()
        assert body["row_count"] == 5
        assert body["schema_status"] == "valid"
        assert body["parity"]["parity_status"] == "passed"
        missing = api_client.post("/predict", json={"run_id": run_id, "rows": [{"not_a_feature": 1}]})
        assert missing.status_code == 422
        csv_bytes = frame.to_csv(index=False).encode("utf-8")
        flight = api_client.post(
            f"/runs/{run_id}/test-flight",
            files={"file": ("new_data.csv", csv_bytes, "text/csv")},
        )
        assert flight.status_code == 200, flight.text
        csv_out = api_client.post(
            f"/runs/{run_id}/test-flight.csv",
            files={"file": ("new_data.csv", csv_bytes, "text/csv")},
        )
        assert csv_out.status_code == 200
        assert b"prediction" in csv_out.content
        pkg = api_client.post(f"/runs/{run_id}/deployment/package")
        assert pkg.status_code == 201
        docker = api_client.get(f"/runs/{run_id}/deployment/package/files/Dockerfile")
        assert docker.status_code == 200
        assert b"python:3.11-slim" in docker.content
        traversal = api_client.get(f"/runs/{run_id}/deployment/package/files/../manifest.json")
        assert traversal.status_code == 404
        unknown = api_client.post("/predict", json={"run_id": "run_does_not_exist", "rows": [{"a": 1}]})
        assert unknown.status_code == 404
        missing_run = api_client.get("/runs/run_does_not_exist/deployment")
        assert missing_run.status_code == 404
        bad_file = api_client.post(
            f"/runs/{run_id}/test-flight",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )
        assert bad_file.status_code == 400
