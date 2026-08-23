"""
Shared fixtures for the M1 test suite.

Every test gets its own fresh InMemoryDatasetStore (function-scoped) so
tests are isolated from each other — no shared mutable state across
tests, no ordering dependencies.
"""

from __future__ import annotations

import os

# Isolate the default API TestClient from the local SQLite product store.
os.environ["PIPER_RUN_STORE"] = "memory"

from pathlib import Path

import pandas as pd
import pytest

from app.storage import InMemoryDatasetStore

TELCO_CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "telco_customer_churn.csv"


@pytest.fixture()
def store() -> InMemoryDatasetStore:
    """A fresh, empty in-memory dataset store for each test."""
    return InMemoryDatasetStore()


@pytest.fixture()
def telco_df() -> pd.DataFrame:
    """
    The real Telco Customer Churn CSV, loaded fresh for each test.

    Loaded from disk each time (not module-cached) so no test can
    accidentally mutate a shared DataFrame object and affect another
    test — mirrors the same isolation principle enforced by
    InMemoryDatasetStore.get()/save() themselves.
    """
    if not TELCO_CSV_PATH.exists():
        pytest.skip(f"Telco CSV not found at {TELCO_CSV_PATH}")
    return pd.read_csv(TELCO_CSV_PATH)


@pytest.fixture()
def loaded_store(store: InMemoryDatasetStore, telco_df: pd.DataFrame) -> InMemoryDatasetStore:
    """A store pre-populated with the real Telco CSV under 'dataset_001'."""
    store.save("dataset_001", telco_df)
    return store


def heuristic_llm_provider():
    """
    M3 Phase 5: build_graph() now requires an llm_provider — this
    factory returns a FakeLLMProvider whose generate_plan() replicates
    the EXACT logic the retired M2/Phase-4 dummy heuristic used to
    apply directly inside plan_node_v2 (drop >=99%-unique columns,
    convert/impute name-matched "charges"/"total" columns, encode
    remaining non-numeric columns, scale remaining numeric columns).

    This exists so every pre-Phase-5 test that calls build_graph()
    keeps testing its ORIGINAL invariant (leakage detection, duplicate-
    plan prevention, multi-model flow, reproducibility metadata,
    failure taxonomy, etc.) unchanged — driven by a fake LLM instead of
    the retired heuristic, not by weakening what's actually asserted.

    On REPLAN (a second call with the same profile), this fake
    provider has no ability to propose something different — matching
    the retired heuristic's own documented "honest limitation" — so a
    second call from the SAME dataset produces the SAME plan, which
    correctly triggers DUPLICATE_PLAN exactly as it always did. Tests
    that specifically need a genuinely different second plan (REPLAN
    behavioral tests) build their own dedicated FakeLLMProvider/stub
    instead of using this shared factory — see test_llm_replan.py.
    """
    from app.agent.tools._profiling_helpers import is_numeric_dtype
    from app.llm.provider import FakeLLMProvider, LLMProviderResult, ProposedPlan, ProposedPlanStep

    class _HeuristicReplicatingProvider:
        def generate_plan(self, context):
            profile = context.dataset_context
            target_column = profile.get("target_column")
            column_contexts = profile.get("column_contexts", [])

            steps: list[ProposedPlanStep] = []
            columns_being_dropped: set = set()
            columns_being_converted_to_numeric: set = set()

            for col in column_contexts:
                name = col["name"]
                if name == target_column:
                    continue

                if col["unique_percentage"] >= 99.0:
                    steps.append(ProposedPlanStep(
                        action=f"Drop identifier-like column '{name}'",
                        tool_name="drop_column",
                        arguments={"column": name, "reason": "unique_percentage >= 99%"},
                        reasoning=f"'{name}' is {col['unique_percentage']}% unique.",
                    ))
                    columns_being_dropped.add(name)
                    continue

                looks_numeric_ish = any(kw in name.lower() for kw in ("charges", "total"))
                if looks_numeric_ish and col["dtype"] not in ("int64", "float64"):
                    steps.append(ProposedPlanStep(
                        action=f"Convert '{name}' to numeric",
                        tool_name="convert_column_type",
                        arguments={"column": name, "target_type": "numeric"},
                        reasoning=f"'{name}' looks numeric but is {col['dtype']}.",
                    ))
                    steps.append(ProposedPlanStep(
                        action=f"Impute missing values in '{name}'",
                        tool_name="impute_missing_values",
                        arguments={"column": name, "strategy": "median"},
                        reasoning="Failed conversions become NaN.",
                    ))
                    columns_being_converted_to_numeric.add(name)

            categorical_columns: list = []
            numeric_columns: list = []
            for col in column_contexts:
                name = col["name"]
                if name == target_column or name in columns_being_dropped:
                    continue
                will_be_numeric = col["dtype"] in ("int64", "float64") or name in columns_being_converted_to_numeric
                (numeric_columns if will_be_numeric else categorical_columns).append(name)

            if categorical_columns:
                steps.append(ProposedPlanStep(
                    action=f"Encode categorical columns: {categorical_columns}",
                    tool_name="encode_categorical_features",
                    arguments={"columns": categorical_columns},
                    reasoning="Remaining non-numeric columns need one-hot encoding.",
                ))
            if numeric_columns:
                steps.append(ProposedPlanStep(
                    action=f"Scale numeric columns: {numeric_columns}",
                    tool_name="scale_features",
                    arguments={"columns": numeric_columns},
                    reasoning="Remaining numeric columns need standard scaling.",
                ))

            return LLMProviderResult(success=True, plan=ProposedPlan(steps=steps))

    return _HeuristicReplicatingProvider()


@pytest.fixture()
def api_client():
    """
    M5: a TestClient against the real FastAPI app (app.main.app), with
    the LLM provider dependency overridden to heuristic_llm_provider()
    — the same deterministic, Ollama-independent double used
    throughout the rest of this suite — so API tests never touch a
    real Ollama server, consistent with the project-wide rule that the
    normal `pytest -q` suite must stay fully deterministic.

    `with TestClient(app) as client:` re-runs app.main's lifespan
    startup on entry, which reassigns FRESH InMemory*Store instances to
    app.state each time — so every test using this fixture gets
    isolated, empty stores, no cross-test state leakage, without any
    extra fixture plumbing.
    """
    from fastapi.testclient import TestClient

    from app.api.dependencies import get_llm_provider
    from app.main import app

    app.dependency_overrides[get_llm_provider] = heuristic_llm_provider
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_llm_provider, None)
