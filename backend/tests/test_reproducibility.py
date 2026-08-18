"""
Behavioral tests for reproducibility metadata (Phase 2).

Two layers, matching the two things under test:

1. TestDatasetFingerprint / TestEnvironmentMetadata: the isolated
   fingerprinting and environment-capture functions, tested directly
   (fast, precise about exactly what changes the fingerprint).
2. TestReproducibilityGraphIntegration: the REAL graph
   (graph.invoke), proving reproducibility_node is actually wired in
   and state.reproducibility survives to the final result — not just
   that the helper functions work in isolation.
"""

from __future__ import annotations

import sys

import numpy
import pandas as pd
import pytest
import sklearn

from app.agent import AgentState, build_graph
from tests.conftest import heuristic_llm_provider
from app.agent.tools.preparation import RANDOM_STATE as SPLIT_RANDOM_STATE
from app.agent.tools.training import RANDOM_STATE as MODEL_RANDOM_STATE
from app.schemas.reproducibility import capture_environment_metadata, dataset_fingerprint
from app.storage import InMemoryDatasetStore, InMemoryModelStore, InMemorySplitStore


# --- Dataset fingerprint ------------------------------------------------


class TestDatasetFingerprint:
    def test_identical_dataframe_same_fingerprint(self):
        """(1) Same DataFrame -> same fingerprint."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        assert dataset_fingerprint(df) == dataset_fingerprint(df.copy())

    def test_one_changed_cell_changes_fingerprint(self):
        """(2) One changed cell -> different fingerprint."""
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        df2 = df1.copy()
        df2.loc[1, "a"] = 999

        assert dataset_fingerprint(df1) != dataset_fingerprint(df2)

    def test_changed_column_value_changes_fingerprint(self):
        """(3) Changed column value -> different fingerprint."""
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        df2 = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "CHANGED"]})

        assert dataset_fingerprint(df1) != dataset_fingerprint(df2)

    def test_column_order_is_significant(self):
        """
        (4) Column order behavior matches the documented dataset
        contract: nothing in this project's existing dataset contract
        (DatasetStore, profiling, cleaning tools) treats column order
        as insignificant, so the fingerprint treats reordered columns
        as a genuinely different dataset (order-SENSITIVE), matching
        the explicit canonicalization decision documented in
        app/schemas/reproducibility.py.
        """
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        df2 = df1[["b", "a"]]

        assert dataset_fingerprint(df1) != dataset_fingerprint(df2)

    def test_fingerprinting_does_not_mutate_original_dataframe(self):
        """(5) Fingerprinting does not mutate the original DataFrame."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        before = df.copy()

        dataset_fingerprint(df)

        pd.testing.assert_frame_equal(df, before)

    def test_row_order_is_significant(self):
        """
        Row order is likewise treated as significant (see module
        docstring's canonicalization decision) — reordered rows
        produce a different fingerprint, since nothing in the existing
        dataset contract treats row order as unordered.
        """
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = df1.iloc[::-1].reset_index(drop=True)

        assert dataset_fingerprint(df1) != dataset_fingerprint(df2)

    def test_dtype_difference_changes_fingerprint(self):
        """
        Two columns with identical-looking values but different
        dtypes (string '1' vs int 1) must NOT collide — dtype is part
        of the canonical representation per the documented decision.
        """
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": ["1", "2", "3"]})

        assert dataset_fingerprint(df1) != dataset_fingerprint(df2)

    def test_missing_values_are_captured_deterministically(self):
        """
        NaN handling: two DataFrames with NaN in the same position
        fingerprint identically to each other, and differently from a
        DataFrame with a real value there — proving NaN isn't silently
        ignored or randomly unstable across calls.
        """
        import numpy as np

        df1 = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        df2 = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        df3 = pd.DataFrame({"a": [1.0, 2.0, 3.0]})

        assert dataset_fingerprint(df1) == dataset_fingerprint(df2)
        assert dataset_fingerprint(df1) != dataset_fingerprint(df3)


# --- Environment metadata -----------------------------------------------


class TestEnvironmentMetadata:
    def test_python_version_matches_actual_runtime(self):
        """(6) Python version equals sys.version information."""
        metadata = capture_environment_metadata()
        assert metadata.python_version == sys.version

    def test_pandas_version_matches_installed_pandas(self):
        """(7) pandas version equals the installed pandas version."""
        metadata = capture_environment_metadata()
        assert metadata.pandas_version == pd.__version__

    def test_numpy_version_matches_installed_numpy(self):
        """(8) NumPy version equals the installed NumPy version."""
        metadata = capture_environment_metadata()
        assert metadata.numpy_version == numpy.__version__

    def test_sklearn_version_matches_installed_sklearn(self):
        """(9) scikit-learn version equals the installed scikit-learn version."""
        metadata = capture_environment_metadata()
        assert metadata.sklearn_version == sklearn.__version__


# --- Real graph integration ----------------------------------------------


@pytest.fixture()
def telco_store(telco_df: pd.DataFrame) -> InMemoryDatasetStore:
    store = InMemoryDatasetStore()
    store.save("dataset_001", telco_df)
    return store


@pytest.fixture()
def fresh_stores(telco_store):
    return telco_store, InMemorySplitStore(), InMemoryModelStore()


class TestReproducibilityGraphIntegration:
    def test_split_random_state_recorded_correctly(self, fresh_stores):
        """(10) split_random_state is recorded correctly."""
        dataset_store, split_store, model_store = fresh_stores
        graph = build_graph(dataset_store, split_store, model_store, heuristic_llm_provider())
        initial = AgentState(run_id="repro_001", dataset_id="dataset_001", target_column="Churn")

        result = graph.invoke(initial, config={"recursion_limit": 50})

        assert result["reproducibility"].split_random_state == SPLIT_RANDOM_STATE

    def test_model_random_state_recorded_correctly(self, fresh_stores):
        """(11) model_random_state is recorded correctly."""
        dataset_store, split_store, model_store = fresh_stores
        graph = build_graph(dataset_store, split_store, model_store, heuristic_llm_provider())
        initial = AgentState(run_id="repro_002", dataset_id="dataset_001", target_column="Churn")

        result = graph.invoke(initial, config={"recursion_limit": 50})

        assert result["reproducibility"].model_random_state == MODEL_RANDOM_STATE

    def test_metadata_populated_through_real_graph(self, fresh_stores):
        """(12) Metadata is populated through the REAL graph."""
        dataset_store, split_store, model_store = fresh_stores
        graph = build_graph(dataset_store, split_store, model_store, heuristic_llm_provider())
        initial = AgentState(run_id="repro_003", dataset_id="dataset_001", target_column="Churn")

        result = graph.invoke(initial, config={"recursion_limit": 50})

        assert result["reproducibility"] is not None
        assert result["reproducibility"].dataset_fingerprint != ""
        assert result["reproducibility"].environment.python_version == sys.version

    def test_final_state_contains_reproducibility_metadata(self, fresh_stores):
        """(13) Final AgentState contains reproducibility metadata."""
        dataset_store, split_store, model_store = fresh_stores
        graph = build_graph(dataset_store, split_store, model_store, heuristic_llm_provider())
        initial = AgentState(run_id="repro_004", dataset_id="dataset_001", target_column="Churn")

        result = graph.invoke(initial, config={"recursion_limit": 50})

        assert result["status"] == "completed"
        assert "reproducibility" in result
        assert result["reproducibility"] is not None

    def test_repeated_deterministic_runs_produce_identical_metadata(self, telco_df):
        """
        (14) Repeated deterministic graph runs produce identical
        reproducibility metadata values, except for intentionally
        run-specific identifiers (run_id, generated model_ids/split_ids
        are NOT part of ReproducibilityMetadata at all — everything in
        it is expected to be byte-identical across runs on the same
        dataset/environment).

        Each iteration gets its own fresh dataset_store/dataset_id, not
        a shared one — clean_node writes cleaned data back to the SAME
        dataset_id it read from (existing, pre-Phase-2 M1/M2 behavior:
        the working dataset is deliberately mutated in place as part of
        a single run's pipeline). Reusing one store/id across repeated
        invoke() calls would mean run 2 starts from run 1's
        already-cleaned dataset, not the same raw input — that would be
        a test-fixture bug, not evidence about reproducibility_node
        itself.
        """
        results = []
        for i in range(3):
            dataset_store = InMemoryDatasetStore()
            dataset_store.save(f"dataset_{i}", telco_df)
            split_store = InMemorySplitStore()
            model_store = InMemoryModelStore()
            graph = build_graph(dataset_store, split_store, model_store, heuristic_llm_provider())
            initial = AgentState(run_id=f"repro_005_{i}", dataset_id=f"dataset_{i}", target_column="Churn")
            result = graph.invoke(initial, config={"recursion_limit": 50})
            results.append(result["reproducibility"])

        fingerprints = {r.dataset_fingerprint for r in results}
        pipeline_fps = {r.pipeline_fingerprint for r in results}
        split_states = {r.split_random_state for r in results}
        model_states = {r.model_random_state for r in results}
        environments = {
            (r.environment.python_version, r.environment.pandas_version,
             r.environment.numpy_version, r.environment.sklearn_version)
            for r in results
        }

        assert len(fingerprints) == 1
        assert len(pipeline_fps) == 1
        assert len(split_states) == 1
        assert len(model_states) == 1
        assert len(environments) == 1

    def test_changing_dataset_changes_recorded_fingerprint(self, telco_df):
        """(15) Changing the dataset changes the recorded dataset fingerprint."""
        clean_store = InMemoryDatasetStore()
        clean_store.save("dataset_clean", telco_df)

        altered_df = telco_df.copy()
        altered_df.loc[0, "MonthlyCharges"] = 99999.0
        altered_store = InMemoryDatasetStore()
        altered_store.save("dataset_altered", altered_df)

        graph_clean = build_graph(clean_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        result_clean = graph_clean.invoke(
            AgentState(run_id="repro_006a", dataset_id="dataset_clean", target_column="Churn"),
            config={"recursion_limit": 50},
        )

        graph_altered = build_graph(altered_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        result_altered = graph_altered.invoke(
            AgentState(run_id="repro_006b", dataset_id="dataset_altered", target_column="Churn"),
            config={"recursion_limit": 50},
        )

        assert (
            result_clean["reproducibility"].dataset_fingerprint
            != result_altered["reproducibility"].dataset_fingerprint
        )

    def test_original_dataset_unchanged_by_reproducibility_node_itself(self, telco_df):
        """
        Confirms reproducibility_node itself never calls
        dataset_store.save() and never mutates any DataFrame it reads —
        isolated by calling the node function directly against a
        split_store, rather than through the full graph. The full
        graph's overall dataset mutation behavior (clean_node writing
        cleaned data back to the same dataset_id) is existing,
        pre-Phase-2 M1/M2 behavior and out of scope for this test — see
        test_repeated_deterministic_runs_produce_identical_metadata's
        docstring for why fixtures must not conflate the two.
        """
        from app.agent.nodes.real_nodes import reproducibility_node
        from app.agent.tools import split_dataset

        split_store = InMemorySplitStore()
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_001", telco_df)

        split_result = split_dataset("dataset_001", "Churn", 0.2, dataset_store, split_store)
        assert split_result.success

        before = dataset_store.get("dataset_001").copy()

        state = AgentState(
            run_id="repro_007", dataset_id="dataset_001", target_column="Churn",
            split_id=split_result.data.split_id,
        )
        update = reproducibility_node(state, split_store)
        assert update["status"] == "running"

        after = dataset_store.get("dataset_001")
        pd.testing.assert_frame_equal(before, after)


# --- Pipeline fingerprint (optional, implemented) -------------------------


class TestPipelineFingerprint:
    def test_same_executable_configuration_same_fingerprint(self, telco_df):
        """
        (16) Same executable configuration -> same fingerprint.

        Each run gets its own fresh dataset_store/dataset_id (same
        reasoning as test_repeated_deterministic_runs_produce_identical_metadata:
        clean_node mutates the working dataset in place under its
        dataset_id as existing, pre-Phase-2 behavior, so two invoke()
        calls sharing one store/id would not actually be "the same
        input" for the second call).
        """
        store1 = InMemoryDatasetStore()
        store1.save("dataset_pf1", telco_df)
        graph1 = build_graph(store1, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        result1 = graph1.invoke(
            AgentState(run_id="repro_pf_001", dataset_id="dataset_pf1", target_column="Churn"),
            config={"recursion_limit": 50},
        )

        store2 = InMemoryDatasetStore()
        store2.save("dataset_pf2", telco_df)
        graph2 = build_graph(store2, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        result2 = graph2.invoke(
            AgentState(run_id="repro_pf_002", dataset_id="dataset_pf2", target_column="Churn"),
            config={"recursion_limit": 50},
        )

        assert (
            result1["reproducibility"].pipeline_fingerprint
            == result2["reproducibility"].pipeline_fingerprint
        )

    def test_changed_executable_configuration_different_fingerprint(self, telco_df):
        """
        (17) Changed executable configuration -> different fingerprint.

        Uses a dataset with a different schema shape (drops a column
        up front) so the dummy planner produces a genuinely different
        executable plan (different cleaning/feature-engineering steps),
        which must change the pipeline fingerprint.
        """
        store_a = InMemoryDatasetStore()
        store_a.save("dataset_a", telco_df)

        reduced_df = telco_df.drop(columns=["PaymentMethod"])
        store_b = InMemoryDatasetStore()
        store_b.save("dataset_b", reduced_df)

        graph_a = build_graph(store_a, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        result_a = graph_a.invoke(
            AgentState(run_id="repro_pf_003", dataset_id="dataset_a", target_column="Churn"),
            config={"recursion_limit": 50},
        )

        graph_b = build_graph(store_b, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        result_b = graph_b.invoke(
            AgentState(run_id="repro_pf_004", dataset_id="dataset_b", target_column="Churn"),
            config={"recursion_limit": 50},
        )

        assert (
            result_a["reproducibility"].pipeline_fingerprint
            != result_b["reproducibility"].pipeline_fingerprint
        )

    def test_rationale_does_not_affect_fingerprint(self):
        """
        (18) Changing only rationale/commentary -> same fingerprint.

        Directly exercises canonicalize_plan() (the underlying
        machinery pipeline_fingerprint reuses) with two PlanSteps that
        differ ONLY in `reasoning`/`action` (human-readable commentary)
        — plan_canonical.py already excludes these fields by design
        (see CanonicalPlanStep), so this proves that exclusion holds
        for the reproducibility use case too, without duplicating
        plan_canonical.py's own test coverage.
        """
        from app.agent.plan_canonical import canonicalize_plan
        from app.agent.state import PlanStep

        step_a = PlanStep(
            step_id="step_01",
            action="Drop identifier-like column 'customerID'",
            tool_name="drop_column",
            arguments={"column": "customerID", "reason": "unique_percentage >= 99%"},
            reasoning="This is reasoning A.",
        )
        step_b = step_a.model_copy(update={
            "action": "Completely different human-readable label",
            "reasoning": "This is totally different reasoning B, much longer and more detailed.",
        })

        hash_a = canonicalize_plan([step_a], "Churn").plan_hash()
        hash_b = canonicalize_plan([step_b], "Churn").plan_hash()

        assert hash_a == hash_b
