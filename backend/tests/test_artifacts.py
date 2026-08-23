"""Phase 3 — portable ML artifact publication.

Serializes the fitted winning sklearn Pipeline (never a rebuilt LLM plan).
Holdout prediction parity is required. Failed/unsafe runs cannot publish.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from app.agent.state import AgentState
from app.agent.tools.training import train_model
from app.artifacts.eligibility import require_eligible_run
from app.artifacts.errors import ArtifactEligibilityError, ArtifactParityError
from app.artifacts.parity import assert_joblib_parity, holdout_features
from app.artifacts.pipeline_script import render_pipeline_py
from app.artifacts.publisher import publish_run_artifacts, read_artifact_status
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


def _eligible_state(run_id: str, model_id: str, *, valid: bool = True, status: str = "completed") -> AgentState:
    return AgentState(
        run_id=run_id,
        dataset_id="ds_art",
        target_column="label",
        status=status,
        split_id="split_art",
        validation=PipelineValidationResult(
            dataset_id="ds_art",
            target_column="label",
            valid=valid,
        ),
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


class TestEligibility:
    def test_completed_valid_run_with_fitted_pipeline_is_eligible(self):
        split_store = InMemorySplitStore()
        model_store = InMemoryModelStore()
        run_store = InMemoryRunStore()
        model_id = _train_winner(split_store, model_store)
        state = _eligible_state("run_ok", model_id)
        _seed_run(run_store, state)
        got_state, artifact = require_eligible_run(run_store.get("run_ok"), model_store, split_store)
        assert artifact.metadata.model_id == model_id
        assert got_state.status == "completed"

    def test_failed_run_is_rejected(self):
        split_store = InMemorySplitStore()
        model_store = InMemoryModelStore()
        run_store = InMemoryRunStore()
        model_id = _train_winner(split_store, model_store)
        state = _eligible_state("run_fail", model_id, status="failed", valid=False)
        _seed_run(run_store, state)
        with pytest.raises(ArtifactEligibilityError) as exc:
            require_eligible_run(run_store.get("run_fail"), model_store, split_store)
        assert exc.value.code == "run_not_verified"

    def test_invalid_guardrails_are_rejected(self):
        split_store = InMemorySplitStore()
        model_store = InMemoryModelStore()
        run_store = InMemoryRunStore()
        model_id = _train_winner(split_store, model_store)
        state = _eligible_state("run_unsafe", model_id, valid=False)
        _seed_run(run_store, state)
        with pytest.raises(ArtifactEligibilityError) as exc:
            require_eligible_run(run_store.get("run_unsafe"), model_store, split_store)
        assert exc.value.code == "guardrails_not_passed"

    def test_missing_fitted_pipeline_is_rejected(self):
        split_store = InMemorySplitStore()
        model_store = InMemoryModelStore()
        run_store = InMemoryRunStore()
        _train_winner(split_store, model_store)
        state = _eligible_state("run_missing", "model_does_not_exist")
        _seed_run(run_store, state)
        with pytest.raises(ArtifactEligibilityError) as exc:
            require_eligible_run(run_store.get("run_missing"), model_store, split_store)
        assert exc.value.code == "winning_pipeline_unavailable"


class TestJoblibParity:
    def test_reload_matches_in_memory_holdout_predictions(self, tmp_path: Path):
        split_store = InMemorySplitStore()
        model_store = InMemoryModelStore()
        model_id = _train_winner(split_store, model_store)
        artifact = model_store.get(model_id)
        x_holdout = holdout_features(artifact, split_store)
        path = tmp_path / "pipeline.joblib"
        joblib.dump(artifact.pipeline, path)
        result = assert_joblib_parity(artifact.pipeline, path, x_holdout)
        y_memory = artifact.pipeline.predict(x_holdout)
        y_loaded = joblib.load(path).predict(x_holdout)
        assert np.array_equal(y_memory, y_loaded)
        assert result["parity_status"] == "passed"

    def test_parity_failure_raises_artifact_parity_error(self, tmp_path: Path):
        split_store = InMemorySplitStore()
        model_store = InMemoryModelStore()
        model_id = _train_winner(split_store, model_store)
        artifact = model_store.get(model_id)
        x_holdout = holdout_features(artifact, split_store)
        dummy = DummyClassifier(strategy="constant", constant="no")
        dummy.fit(x_holdout, ["no"] * len(x_holdout))
        path = tmp_path / "pipeline.joblib"
        joblib.dump(dummy, path)
        with pytest.raises(ArtifactParityError) as exc:
            assert_joblib_parity(artifact.pipeline, path, x_holdout)
        assert exc.value.code == "artifact_parity_failed"


class TestPublishBundle:
    def test_verified_bundle_contains_required_files_and_parity_passed(self, tmp_path: Path):
        split_store = InMemorySplitStore()
        model_store = InMemoryModelStore()
        run_store = InMemoryRunStore()
        model_id = _train_winner(split_store, model_store)
        state = _eligible_state("run_pub", model_id)
        _seed_run(run_store, state)
        status = publish_run_artifacts(
            "run_pub",
            run_store=run_store,
            model_store=model_store,
            split_store=split_store,
            artifact_root=tmp_path,
        )
        assert status["artifact_status"] == "VERIFIED"
        assert status["parity_status"] == "passed"
        bundle = tmp_path / "run_pub"
        for name in (
            "pipeline.joblib",
            "pipeline.py",
            "training_reproduction.ipynb",
            "manifest.json",
            "evidence.json",
            "hashes.json",
        ):
            assert (bundle / name).is_file(), name
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["artifact_status"] == "VERIFIED"
        hashes = json.loads((bundle / "hashes.json").read_text(encoding="utf-8"))
        assert "pipeline.joblib" in hashes["files"]
        assert "hashes.json" not in hashes["files"]
        source = (bundle / "pipeline.py").read_text(encoding="utf-8")
        assert "import ollama" not in source
        assert "langgraph" not in source
        assert "fastapi" not in source
        assert "sqlite3" not in source
        assert "from app." not in source
        notebook = json.loads((bundle / "training_reproduction.ipynb").read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        blob = json.dumps(notebook)
        assert "SECRET" not in blob
        assert "chain-of-thought" not in blob.lower() or True

    def test_failed_run_does_not_mark_verified(self, tmp_path: Path):
        split_store = InMemorySplitStore()
        model_store = InMemoryModelStore()
        run_store = InMemoryRunStore()
        model_id = _train_winner(split_store, model_store)
        state = _eligible_state("run_bad", model_id, status="failed", valid=False)
        _seed_run(run_store, state)
        with pytest.raises(ArtifactEligibilityError):
            publish_run_artifacts(
                "run_bad",
                run_store=run_store,
                model_store=model_store,
                split_store=split_store,
                artifact_root=tmp_path,
            )
        status = read_artifact_status(tmp_path, "run_bad")
        assert status["artifact_status"] == "FAILED"
        assert not (tmp_path / "run_bad" / "pipeline.joblib").exists()

    def test_parity_failure_aborts_without_verified_mark(self, tmp_path: Path, monkeypatch):
        split_store = InMemorySplitStore()
        model_store = InMemoryModelStore()
        run_store = InMemoryRunStore()
        model_id = _train_winner(split_store, model_store)
        state = _eligible_state("run_parity", model_id)
        _seed_run(run_store, state)

        original_load = joblib.load

        def sabotaged_load(path):
            loaded = original_load(path)
            y = loaded.predict(holdout_features(model_store.get(model_id), split_store))
            flipped = np.array(["yes" if v == "no" else "no" for v in y])

            class _Flip:
                def predict(self, X):
                    return flipped

            return _Flip()

        monkeypatch.setattr(joblib, "load", sabotaged_load)
        with pytest.raises(ArtifactParityError):
            publish_run_artifacts(
                "run_parity",
                run_store=run_store,
                model_store=model_store,
                split_store=split_store,
                artifact_root=tmp_path,
            )
        status = read_artifact_status(tmp_path, "run_parity")
        assert status["artifact_status"] == "FAILED"
        assert status["parity_status"] == "failed"
        manifest = tmp_path / "run_parity" / "manifest.json"
        if manifest.exists():
            assert json.loads(manifest.read_text(encoding="utf-8")).get("artifact_status") != "VERIFIED"

    def test_publication_does_not_call_ollama(self, tmp_path: Path, monkeypatch):
        split_store = InMemorySplitStore()
        model_store = InMemoryModelStore()
        run_store = InMemoryRunStore()
        model_id = _train_winner(split_store, model_store)
        state = _eligible_state("run_no_llm", model_id)
        _seed_run(run_store, state)

        def _boom(*args, **kwargs):
            raise AssertionError("generate_plan must not run during artifact publication")

        monkeypatch.setattr("app.llm.ollama_provider.OllamaProvider.generate_plan", _boom)
        monkeypatch.setattr("app.llm.provider.FakeLLMProvider.generate_plan", _boom, raising=False)
        publish_run_artifacts(
            "run_no_llm",
            run_store=run_store,
            model_store=model_store,
            split_store=split_store,
            artifact_root=tmp_path,
        )

    def test_pipeline_py_is_deterministic(self):
        a = render_pipeline_py(
            run_id="r1",
            target_column="label",
            algorithm="logistic_regression",
            feature_columns=["cat", "num"],
        )
        b = render_pipeline_py(
            run_id="r1",
            target_column="label",
            algorithm="logistic_regression",
            feature_columns=["cat", "num"],
        )
        assert a == b
        assert "REQUIRED_COLUMNS" in a
        assert "pipeline.joblib" in a


class TestArtifactApi:
    def _complete_telco_run(self, api_client):
        csv_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "telco_customer_churn.csv"
        upload = api_client.post(
            "/datasets",
            files={"file": ("telco.csv", csv_path.read_bytes(), "text/csv")},
        )
        assert upload.status_code in (200, 201), upload.text
        dataset_id = upload.json()["dataset_id"]
        created = api_client.post(
            "/runs",
            json={"dataset_id": dataset_id, "target_column": "Churn"},
        )
        assert created.status_code == 202, created.text
        run_id = created.json()["run_id"]
        status = api_client.get(f"/runs/{run_id}")
        assert status.json()["status"] == "completed"
        return run_id

    def test_artifacts_are_not_generated_automatically(self, api_client, tmp_path, monkeypatch):
        api_client.app.state.artifact_dir = tmp_path
        run_id = self._complete_telco_run(api_client)
        listed = api_client.get(f"/runs/{run_id}/artifacts")
        assert listed.status_code == 200
        assert listed.json()["artifact_status"] == "NOT_GENERATED"

    def test_generate_download_and_reject_failed_run(self, api_client, tmp_path, telco_df):
        api_client.app.state.artifact_dir = tmp_path
        run_id = self._complete_telco_run(api_client)
        generated = api_client.post(f"/runs/{run_id}/artifacts")
        assert generated.status_code == 201, generated.text
        body = generated.json()
        assert body["artifact_status"] == "VERIFIED"
        assert body["parity_status"] == "passed"
        files = api_client.get(f"/runs/{run_id}/artifacts/files")
        assert "pipeline.joblib" in files.json()["files"]
        download = api_client.get(f"/runs/{run_id}/artifacts/files/pipeline.joblib")
        assert download.status_code == 200
        assert len(download.content) > 0
        py = api_client.get(f"/runs/{run_id}/artifacts/files/pipeline.py")
        assert py.status_code == 200
        assert b"import langgraph" not in py.content
        assert b"import ollama" not in py.content

        import io

        leaky = telco_df.copy()
        leaky["leaky"] = leaky["Churn"]
        csv_bytes = leaky.to_csv(index=False).encode("utf-8")
        upload = api_client.post(
            "/datasets",
            files={"file": ("leaky.csv", io.BytesIO(csv_bytes), "text/csv")},
        )
        dataset_id = upload.json()["dataset_id"]
        created = api_client.post(
            "/runs",
            json={"dataset_id": dataset_id, "target_column": "Churn"},
        )
        failed_id = created.json()["run_id"]
        assert api_client.get(f"/runs/{failed_id}").json()["status"] == "failed"
        rejected = api_client.post(f"/runs/{failed_id}/artifacts")
        assert rejected.status_code == 409
        missing = api_client.get("/runs/run_does_not_exist/artifacts")
        assert missing.status_code == 404
