"""Phase 4 — governance, explainability, fingerprints, fairness."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

from app.agent.state import AgentState, OperationRecord
from app.agent.tools.training import train_model
from app.governance.assemble import assemble_governance_bundle
from app.governance.documents import GOVERNANCE_DOCUMENT_NAMES, render_governance_document
from app.governance.explainability import extract_feature_importance
from app.governance.fairness import analyze_subgroups
from app.governance.hashing import HASH_ALGORITHM, sha256_bytes
from app.governance.model_card import build_model_card
from app.schemas.evaluation import ConfusionMatrix, EvaluationResult, ModelComparison, ModelComparisonEntry
from app.schemas.guardrails import PipelineValidationResult
from app.schemas.training import FeatureEngineeringIntent, TrainingResult
from app.storage.dataset_store import InMemoryDatasetStore
from app.storage.model_store import InMemoryModelStore, ModelArtifact
from app.storage.run_store import InMemoryRunStore
from app.storage.split_store import InMemorySplitStore


def _frames(*, n: int = 80, extra_group: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    data = {
        "cat": rng.choice(["a", "b"], size=n),
        "num": rng.normal(size=n),
        "label": rng.choice(["yes", "no"], size=n),
    }
    if extra_group:
        data["group"] = rng.choice(["g1", "g2"], size=n)
    df = pd.DataFrame(data)
    cut = int(n * 0.8)
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


def _train(split_store, model_store, *, algorithm="logistic_regression", extra_group=False, n=80):
    train_df, test_df = _frames(n=n, extra_group=extra_group)
    split_store.save("split_gov", train_df, test_df)
    intent = FeatureEngineeringIntent(
        categorical_columns=["cat"],
        numeric_columns_to_scale=["num"],
    )
    params = {"C": 1.0, "max_iter": 1000} if algorithm == "logistic_regression" else {"n_estimators": 50}
    result = train_model(
        "split_gov",
        "label",
        algorithm,
        params,
        intent,
        split_store,
        model_store,
    )
    assert result.success is True
    return result.data


def _state(run_id: str, trained: TrainingResult, *, valid: bool = True, status: str = "completed") -> AgentState:
    return AgentState(
        run_id=run_id,
        dataset_id="ds_gov",
        target_column="label",
        task_type="binary_classification",
        status=status,
        split_id=trained.split_id,
        profile={
            "dataset_id": "ds_gov",
            "rows": 80,
            "columns": 3,
            "column_profiles": [
                {
                    "name": "cat",
                    "dtype": "object",
                    "missing_count": 2,
                    "missing_percentage": 2.5,
                    "unique_count": 2,
                    "unique_percentage": 2.5,
                    "sample_values": [],
                },
                {
                    "name": "num",
                    "dtype": "float64",
                    "missing_count": 0,
                    "missing_percentage": 0.0,
                    "unique_count": 80,
                    "unique_percentage": 100.0,
                    "sample_values": [],
                },
                {
                    "name": "label",
                    "dtype": "object",
                    "missing_count": 0,
                    "missing_percentage": 0.0,
                    "unique_count": 2,
                    "unique_percentage": 2.5,
                    "sample_values": [],
                },
            ],
            "duplicate_rows": 3,
            "memory_usage_bytes": 128,
        },
        cleaning_log=[
            OperationRecord(
                operation_id="op1",
                tool_name="impute_missing_values",
                arguments={"column": "cat", "strategy": "mode"},
                result_summary="Imputed cat",
                reason="missing values",
                timestamp="2026-08-24T00:00:00Z",
            )
        ],
        validation=PipelineValidationResult(
            dataset_id="ds_gov",
            target_column="label",
            valid=valid,
        ),
        model_results=[trained],
        evaluation_results=[
            EvaluationResult(
                model_id=trained.model_id,
                split_id=trained.split_id,
                accuracy=0.81,
                precision=0.8,
                recall=0.79,
                f1=0.8,
                roc_auc=0.82,
                confusion_matrix=ConfusionMatrix(tn=8, fp=2, fn=2, tp=4),
                test_rows=16,
            )
        ],
        comparison=ModelComparison(
            models=[
                ModelComparisonEntry(
                    model_id=trained.model_id,
                    algorithm=trained.algorithm,
                    accuracy=0.81,
                    precision=0.8,
                    recall=0.79,
                    f1=0.8,
                    roc_auc=0.82,
                )
            ],
            recommended_model_id=trained.model_id,
            justification=f"{trained.algorithm} selected: F1 0.8.",
        ),
    )


class TestHashing:
    def test_sha256_known_payload(self):
        assert HASH_ALGORITHM == "sha256"
        assert sha256_bytes(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


class TestModelAndDataCards:
    def test_model_card_from_recorded_metrics_not_invented(self):
        split_store = InMemorySplitStore()
        model_store = InMemoryModelStore()
        trained = _train(split_store, model_store)
        state = _state("run_card", trained)
        card = build_model_card(
            "run_card",
            state,
            dataset_id="ds_gov",
            model_store=model_store,
            artifact_status={"artifact_status": "NOT_GENERATED"},
        )
        assert card.status == "AVAILABLE"
        names = {item.name: item.value for item in card.evaluation_metrics}
        assert names["f1"] == 0.8
        assert card.winning_algorithm == "logistic_regression"

    def test_missing_evaluation_does_not_fabricate_zeros(self):
        split_store = InMemorySplitStore()
        model_store = InMemoryModelStore()
        trained = _train(split_store, model_store)
        state = _state("run_empty", trained)
        state.evaluation_results = []
        card = build_model_card(
            "run_empty",
            state,
            dataset_id="ds_gov",
            model_store=model_store,
            artifact_status=None,
        )
        assert card.evaluation_metrics == []

    def test_failed_run_model_card_is_not_available(self):
        split_store = InMemorySplitStore()
        model_store = InMemoryModelStore()
        trained = _train(split_store, model_store)
        state = _state("run_fail", trained, valid=False, status="failed")
        card = build_model_card(
            "run_fail",
            state,
            dataset_id="ds_gov",
            model_store=model_store,
            artifact_status=None,
        )
        assert card.status == "NOT_AVAILABLE"

    def test_data_card_omits_sample_values_and_uses_executed_ops(self):
        split_store = InMemorySplitStore()
        model_store = InMemoryModelStore()
        trained = _train(split_store, model_store)
        state = _state("run_data", trained)
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("ds_gov", pd.DataFrame({"cat": ["a"], "num": [1.0], "label": ["yes"]}))
        bundle = assemble_governance_bundle(
            "run_data",
            run_status="completed",
            state=state,
            dataset_id="ds_gov",
            model_store=model_store,
            split_store=split_store,
            dataset_store=dataset_store,
        )
        dumped = bundle.data_card.model_dump()
        assert "sample_values" not in str(dumped.get("column_summaries"))
        assert bundle.data_card.preprocessing_operations[0]["tool_name"] == "impute_missing_values"
        assert any("duplicate_rows" in item for item in bundle.data_card.data_quality_findings)


class TestFingerprints:
    def test_content_hash_distinct_from_metadata(self, tmp_path: Path):
        split_store = InMemorySplitStore()
        model_store = InMemoryModelStore()
        trained = _train(split_store, model_store)
        state = _state("run_fp", trained)
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("ds_gov", pd.DataFrame({"cat": ["a", "b"], "num": [1.0, 2.0], "label": ["yes", "no"]}))
        bundle = assemble_governance_bundle(
            "run_fp",
            run_status="completed",
            state=state,
            dataset_id="ds_gov",
            model_store=model_store,
            split_store=split_store,
            dataset_store=dataset_store,
            artifact_root=tmp_path,
        )
        kinds = {entry.name: entry.kind for entry in bundle.fingerprints.content_hashes}
        assert kinds["dataset"] == "CONTENT_HASH"
        assert kinds["executed_plan"] == "CONTENT_HASH"
        assert bundle.fingerprints.hash_algorithm == "sha256"
        assert "tamper" in bundle.fingerprints.caveat.lower()
        dataset_hash = next(e for e in bundle.fingerprints.content_hashes if e.name == "dataset")
        assert dataset_hash.available is True
        assert len(dataset_hash.digest) == 64


class TestExplainability:
    def test_logistic_maps_one_hot_names_and_direction(self):
        split_store = InMemorySplitStore()
        model_store = InMemoryModelStore()
        trained = _train(split_store, model_store, algorithm="logistic_regression")
        report = extract_feature_importance(_state("run_lr", trained), model_store)
        assert report.status == "AVAILABLE"
        assert report.method == "logistic_regression_coefficients"
        assert report.rows
        assert any(row.direction in {"positive", "negative", "neutral"} for row in report.rows)

    def test_random_forest_impurity_importance(self):
        split_store = InMemorySplitStore()
        model_store = InMemoryModelStore()
        trained = _train(split_store, model_store, algorithm="random_forest")
        report = extract_feature_importance(_state("run_rf", trained), model_store)
        assert report.status == "AVAILABLE"
        assert report.method == "random_forest_impurity"
        clf = model_store.get(trained.model_id).pipeline.named_steps["classifier"]
        assert isinstance(clf, RandomForestClassifier)
        assert all(row.direction is None for row in report.rows)

    def test_unsupported_estimator_is_not_available(self):
        split_store = InMemorySplitStore()
        model_store = InMemoryModelStore()
        trained = _train(split_store, model_store)
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit([[0], [1]], ["yes", "no"])
        artifact = model_store.get(trained.model_id)
        model_store.save(trained.model_id, ModelArtifact(metadata=artifact.metadata, pipeline=dummy))
        report = extract_feature_importance(_state("run_dummy", trained), model_store)
        assert report.status == "NOT_AVAILABLE"
        assert report.rows == []


class TestFairness:
    def test_operator_specified_column_and_insufficient_n(self):
        split_store = InMemorySplitStore()
        model_store = InMemoryModelStore()
        trained = _train(split_store, model_store, extra_group=True, n=80)
        report = analyze_subgroups(
            _state("run_fair_small", trained),
            columns=["group"],
            model_store=model_store,
            split_store=split_store,
        )
        assert report.status in {"INSUFFICIENT_DATA", "AVAILABLE"}
        if report.status == "INSUFFICIENT_DATA":
            assert any(row.warning for row in report.groups)
            assert all(row.f1 is None for row in report.groups if not row.sufficient)

    def test_sufficient_groups_compute_rates_without_legal_claim(self):
        split_store = InMemorySplitStore()
        model_store = InMemoryModelStore()
        trained = _train(split_store, model_store, extra_group=True, n=400)
        state = _state("run_fair", trained)
        report = analyze_subgroups(
            state,
            columns=["group"],
            model_store=model_store,
            split_store=split_store,
        )
        assert report.status == "AVAILABLE"
        assert "compliance" in report.disclaimer.lower() or "legal" in report.disclaimer.lower()
        sufficient = [row for row in report.groups if row.sufficient]
        assert sufficient
        assert all(row.selection_rate is not None for row in sufficient)

    def test_does_not_infer_protected_columns(self):
        split_store = InMemorySplitStore()
        model_store = InMemoryModelStore()
        trained = _train(split_store, model_store, extra_group=True, n=400)
        report = analyze_subgroups(
            _state("run_none", trained),
            columns=[],
            model_store=model_store,
            split_store=split_store,
        )
        assert report.status == "NOT_REQUESTED"
        assert report.groups == []


class TestNoLlm:
    def test_governance_package_has_no_llm_imports(self):
        root = Path(__file__).resolve().parents[1] / "app" / "governance"
        for path in root.glob("*.py"):
            text = path.read_text(encoding="utf-8").lower()
            assert "generate_plan" not in text
            assert "ollama" not in text
            assert "openai" not in text


class TestDocumentsAndAssemble:
    def test_documents_are_deterministic(self):
        split_store = InMemorySplitStore()
        model_store = InMemoryModelStore()
        trained = _train(split_store, model_store)
        state = _state("run_doc", trained)
        kwargs = dict(
            run_status="completed",
            state=state,
            dataset_id="ds_gov",
            model_store=model_store,
            split_store=split_store,
        )
        a = assemble_governance_bundle("run_doc", **kwargs)
        b = assemble_governance_bundle("run_doc", **kwargs)
        assert a.model_dump(mode="json") == b.model_dump(mode="json")
        media, body = render_governance_document(a, "model_card.md")
        assert media == "text/markdown"
        assert "Model Card" in body
        assert GOVERNANCE_DOCUMENT_NAMES


class TestGovernanceApi:
    def _complete(self, api_client):
        csv_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "telco_customer_churn.csv"
        upload = api_client.post(
            "/datasets",
            files={"file": ("telco.csv", csv_path.read_bytes(), "text/csv")},
        )
        dataset_id = upload.json()["dataset_id"]
        created = api_client.post("/runs", json={"dataset_id": dataset_id, "target_column": "Churn"})
        run_id = created.json()["run_id"]
        assert api_client.get(f"/runs/{run_id}").json()["status"] == "completed"
        return run_id

    def test_governance_and_documents_and_404(self, api_client, tmp_path):
        api_client.app.state.artifact_dir = tmp_path
        run_id = self._complete(api_client)
        response = api_client.get(f"/runs/{run_id}/governance")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["schema_version"] == "piper.governance.v1"
        assert body["model_card"]["status"] == "AVAILABLE"
        assert body["data_card"]["status"] == "AVAILABLE"
        assert body["fingerprints"]["hash_algorithm"] == "sha256"
        assert "generate_plan" not in response.text.lower()
        doc = api_client.get(f"/runs/{run_id}/governance/documents/model_card.md")
        assert doc.status_code == 200
        assert b"Model Card" in doc.content
        missing = api_client.get("/runs/run_does_not_exist/governance")
        assert missing.status_code == 404

    def test_in_progress_is_409(self, api_client):
        from app.api.dependencies import get_run_store
        from app.main import app

        store = InMemoryRunStore()
        state = AgentState(run_id="run_live", dataset_id="ds", target_column="label", status="running")
        store.create("run_live", state)
        app.dependency_overrides[get_run_store] = lambda: store
        try:
            response = api_client.get("/runs/run_live/governance")
            assert response.status_code == 409
        finally:
            app.dependency_overrides.pop(get_run_store, None)

    def test_fairness_query_does_not_block_and_warns(self, api_client, tmp_path):
        api_client.app.state.artifact_dir = tmp_path
        run_id = self._complete(api_client)
        response = api_client.get(f"/runs/{run_id}/governance/fairness", params={"column": "gender"})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] in {"AVAILABLE", "INSUFFICIENT_DATA", "NOT_AVAILABLE"}
        assert "legal" in body["disclaimer"].lower() or "compliance" in body["disclaimer"].lower()
        assert body["requested_columns"] == ["gender"]
