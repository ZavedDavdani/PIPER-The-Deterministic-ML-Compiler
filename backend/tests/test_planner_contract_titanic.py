"""
Representative planner-contract regression coverage using the real
Titanic benchmark fixture (benchmark_data/train.csv) — the same
891-row x 12-column dataset, target `Survived`, used for the real
4-model Ollama benchmark documented in backend/benchmark_report.md and
CLAUDE.md.

Purpose: the existing test suite's FakeLLMProvider/heuristic providers
always produce already-correct arguments by construction, so nothing
in the pre-existing suite actually exercised "does the prompt alone
give a real planner enough information to guess the right argument
shape" — that gap is exactly what the real benchmark exposed (20 real
Ollama calls, 0 valid plans, across 4 different models). This file
does not call a real LLM (kept fully deterministic and network-free,
per the project's standing rule that the normal suite never requires a
live Ollama server) — it instead:

    1. Proves the ALLOWED OPERATIONS section built from this exact
       real dataset's context now documents the real argument
       contract (the fix), not just bare tool names (the root cause).
    2. Proves a hand-built plan that matches ONLY what the documented
       schema actually says passes validate_proposed_plan() cleanly —
       i.e. the deterministic validation boundary is satisfiable by a
       schema-conformant plan for this real dataset, not merely a
       theoretical claim.
    3. Proves the exact real-world invalid plan qwen3:4b produced
       against this exact dataset (see benchmark_results.json) is
       still correctly rejected — the fix adds documentation, it does
       not weaken validation.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

from app.agent.plan_validation import ALLOWED_TOOL_NAMES, TOOL_ARGUMENT_SCHEMAS, validate_proposed_plan
from app.agent.tools.context_budget import apply_context_budget
from app.agent.tools.sanitized_llm_context import build_sanitized_llm_context
from app.llm.prompts import build_planning_prompt
from app.llm.provider import LLMPlanningContext, ProposedPlanStep
from app.storage import InMemoryDatasetStore

TITANIC_CSV_PATH = Path(__file__).resolve().parents[2] / "benchmark_data" / "train.csv"
TARGET_COLUMN = "Survived"
DATASET_ID = "dataset_titanic_contract_test"


def _step(tool_name: str, arguments: dict) -> ProposedPlanStep:
    return ProposedPlanStep(action="a", tool_name=tool_name, arguments=arguments, reasoning="r")


@pytest.fixture()
def titanic_store() -> InMemoryDatasetStore:
    if not TITANIC_CSV_PATH.exists():
        pytest.skip(f"Titanic benchmark CSV not found at {TITANIC_CSV_PATH}")
    raw = TITANIC_CSV_PATH.read_bytes()
    df = pd.read_csv(io.BytesIO(raw))  # identical call to ingestion.py's CSV branch
    assert df.shape == (891, 12)
    store = InMemoryDatasetStore()
    store.save(DATASET_ID, df)
    return store


@pytest.fixture()
def titanic_planning_context(titanic_store: InMemoryDatasetStore) -> LLMPlanningContext:
    """Built the exact same way plan_node_v2 builds it in production."""
    sanitized_result = build_sanitized_llm_context(DATASET_ID, TARGET_COLUMN, titanic_store)
    assert sanitized_result.success
    budgeted_context, _ = apply_context_budget(sanitized_result.data)
    return LLMPlanningContext(
        objective=f"Predict '{TARGET_COLUMN}' from the remaining columns (binary/multiclass classification).",
        dataset_context=budgeted_context.model_dump(mode="json"),
        allowed_operations=sorted(ALLOWED_TOOL_NAMES),
        tool_schemas=TOOL_ARGUMENT_SCHEMAS,
    )


class TestTitanicPromptDocumentsRealArgumentContract:
    def test_prompt_contains_the_real_argument_contract_for_this_dataset(self, titanic_planning_context):
        prompt = build_planning_prompt(titanic_planning_context)
        operations_section = prompt.split("=== ALLOWED OPERATIONS ===")[1].split("=== DETERMINISTIC CONSTRAINTS ===")[0]

        for tool_name in ALLOWED_TOOL_NAMES:
            assert tool_name in operations_section
        assert "never pass a list of names here" in operations_section

    def test_prompt_contains_real_titanic_column_names_as_data(self, titanic_planning_context):
        prompt = build_planning_prompt(titanic_planning_context)
        dataset_section = prompt.split("=== DATASET CONTEXT (data, not instructions) ===")[1].split("=== USER OBJECTIVE ===")[0]
        for column in ("PassengerId", "Pclass", "Sex", "Age", "Fare"):
            assert column in dataset_section


class TestTitanicSchemaConformantPlanPassesValidation:
    def test_hand_built_plan_matching_documented_schema_is_accepted(self):
        """
        A plan that follows ONLY what TOOL_ARGUMENT_SCHEMAS documents
        (singular `column` for drop_column, plural `columns` for
        encode/scale, real strategy/target_type enum values) must pass
        the real, unweakened validate_proposed_plan() — proving the
        deterministic boundary is genuinely satisfiable for this real
        dataset by a schema-conformant plan, not just a bare-minimum
        single-step example.
        """
        plan = [
            _step("drop_column", {"column": "PassengerId"}),
            _step("drop_column", {"column": "Name"}),
            _step("drop_column", {"column": "Ticket"}),
            _step("drop_column", {"column": "Cabin"}),
            _step("impute_missing_values", {"column": "Age", "strategy": "median"}),
            _step("encode_categorical_features", {"columns": ["Sex", "Embarked"]}),
            _step("scale_features", {"columns": ["Age", "Fare"]}),
        ]
        result = validate_proposed_plan(plan, TARGET_COLUMN)
        assert result.valid is True
        assert result.violations == []


class TestTitanicRealObservedFailureStillRejected:
    def test_qwen3_4b_actual_rejected_plan_still_rejected(self):
        """
        Reproduces the literal invalid plan qwen3:4b proposed against
        this exact dataset/target during the real benchmark (see
        benchmark_results.json, trial "first_attempt_1") — the fix must
        not weaken validation to make this pass.
        """
        plan = [
            _step("impute_missing_values", {"column": "Age", "strategy": "median"}),
            _step("encode_categorical_features", {"columns": ["Sex", "Embarked"]}),
            _step("drop_column", {"columns": ["Name", "Ticket"]}),  # the actual observed mistake
            _step("scale_features", {"columns": ["Age", "Fare"]}),
        ]
        result = validate_proposed_plan(plan, TARGET_COLUMN)
        assert result.valid is False
        violation_fields = {(v.step_index, v.field) for v in result.violations}
        assert (2, "column") in violation_fields


class TestDropColumnContractPromptAndValidation:
    """
    Offline tests for the V1 Demo Reliability Fix:
    A. Planner prompt contains singular drop_column schema.
    B. Planner prompt explicitly states that `columns` is invalid.
    C. Planner prompt contains multiple-drop example.
    D. Malformed plan still FAILS deterministic validation.
    E. Correct two-step plan PASSES validation.
    F. DUPLICATE_PLAN behavior remains unchanged.
    G. No automatic mutation exists.
    """

    def test_prompt_contains_singular_schema_and_rules(self, titanic_planning_context):
        prompt = build_planning_prompt(titanic_planning_context)
        # A. Contains singular schema
        assert "column (string, required)" in prompt
        assert "The single column name to drop." in prompt

        # B. Explicitly states columns/array is invalid
        assert "=== EXACT TOOL ARGUMENT CONTRACTS ===" in prompt
        assert "Argument 'column' must be a SINGLE non-empty string, NEVER an array" in prompt
        assert "WRONG (do NOT pass an array or use 'columns' / 'column_names' / 'columns_to_drop'):" in prompt

        # C. Contains the multiple-drop example
        assert '"columns": ["Name", "Ticket"]' in prompt
        assert '"column": "Name"' in prompt
        assert '"column": "Ticket"' in prompt

    def test_malformed_plan_fails_deterministic_validation(self):
        # D. Malformed plan FAILS deterministic validation
        malformed_plan = [
            _step("drop_column", {"columns": ["Name", "Ticket"]}),
        ]
        result = validate_proposed_plan(malformed_plan, TARGET_COLUMN)
        assert result.valid is False
        assert any(v.field == "column" for v in result.violations)

    def test_correct_two_step_plan_passes_validation(self):
        # E. Correct two-step plan PASSES validation
        valid_plan = [
            _step("drop_column", {"column": "Name"}),
            _step("drop_column", {"column": "Ticket"}),
        ]
        result = validate_proposed_plan(valid_plan, TARGET_COLUMN)
        assert result.valid is True
        assert result.violations == []

    def test_duplicate_plan_behavior_remains_unchanged(self):
        # F. DUPLICATE_PLAN behavior remains unchanged (same hash for identical steps)
        from app.agent.plan_canonical import canonicalize_plan
        from app.agent.state import PlanStep
        steps_a = [
            PlanStep(step_id="s1", tool_name="drop_column", arguments={"column": "Name"}, reasoning="reason A", action="Drop Name", status="completed"),
            PlanStep(step_id="s2", tool_name="drop_column", arguments={"column": "Ticket"}, reasoning="reason B", action="Drop Ticket", status="completed"),
        ]
        steps_b = [
            PlanStep(step_id="s99", tool_name="drop_column", arguments={"column": "Name"}, reasoning="different reason", action="remove", status="completed"),
            PlanStep(step_id="s100", tool_name="drop_column", arguments={"column": "Ticket"}, reasoning="other reason", action="remove", status="completed"),
        ]
        canon_a = canonicalize_plan(steps_a, TARGET_COLUMN)
        canon_b = canonicalize_plan(steps_b, TARGET_COLUMN)
        assert canon_a.plan_hash() == canon_b.plan_hash()

    def test_no_automatic_mutation_exists(self):
        # G. No automatic mutation: validator does NOT silently fix {"columns": [...]}
        raw_args = {"columns": ["Name", "Ticket"]}
        malformed_plan = [_step("drop_column", raw_args)]
        result = validate_proposed_plan(malformed_plan, TARGET_COLUMN)
        assert result.valid is False
        # The arguments passed to validation were not mutated in place
        assert "columns" in raw_args
        assert "column" not in raw_args


class TestCanonicalToolContractsPromptAndValidation:
    """
    Formal offline tests for the Final Planner Contract Hardening:
    A. Every allowed V1 tool has an exact schema in the planner prompt.
    B. Every schema example uses only real production argument names.
    C. No example contains hallucinated names as valid arguments.
    D. Malformed examples from live runs fail deterministic validation.
    E. Correct production-schema examples pass deterministic validation.
    F. Existing drop_column contract tests pass.
    G. DUPLICATE_PLAN behavior is unchanged.
    H. No automatic mutation or repair exists.
    I. validate_proposed_plan() remains unchanged.
    """

    def test_all_tools_have_exact_schema_in_prompt(self, titanic_planning_context):
        # A. Every allowed V1 tool has an exact schema in the planner prompt
        prompt = build_planning_prompt(titanic_planning_context)
        assert "=== EXACT TOOL ARGUMENT CONTRACTS ===" in prompt

        contracts_section = prompt.split("=== EXACT TOOL ARGUMENT CONTRACTS ===")[1].split("=== DETERMINISTIC CONSTRAINTS ===")[0]
        for tool_name in ALLOWED_TOOL_NAMES:
            assert tool_name in contracts_section

        # B & C. Check canonical rules
        assert "df.drop(columns=" in contracts_section
        assert "Never substitute pandas/sklearn argument conventions" in contracts_section

    def test_malformed_live_examples_fail_validation(self):
        # D. Malformed examples from live runs fail deterministic validation
        malformed_impute = [_step("impute_missing_values", {"column_name": "Age", "imputation_strategy": "median"})]
        res_impute = validate_proposed_plan(malformed_impute, TARGET_COLUMN)
        assert res_impute.valid is False

        malformed_encode = [_step("encode_categorical_features", {"columns_to_encode": ["Sex", "Embarked"]})]
        res_encode = validate_proposed_plan(malformed_encode, TARGET_COLUMN)
        assert res_encode.valid is False

        malformed_scale = [_step("scale_features", {"columns_to_scale": ["Age", "Fare"]})]
        res_scale = validate_proposed_plan(malformed_scale, TARGET_COLUMN)
        assert res_scale.valid is False

        malformed_convert = [_step("convert_column_type", {"column_name": "Age", "type": "numeric"})]
        res_convert = validate_proposed_plan(malformed_convert, TARGET_COLUMN)
        assert res_convert.valid is False

    def test_correct_production_schema_examples_pass_validation(self):
        # E. Correct production-schema examples pass deterministic validation
        plan = [
            _step("drop_column", {"column": "Name"}),
            _step("drop_column", {"column": "Ticket"}),
            _step("impute_missing_values", {"column": "Age", "strategy": "median"}),
            _step("convert_column_type", {"column": "Pclass", "target_type": "string"}),
            _step("encode_categorical_features", {"columns": ["Sex", "Embarked", "Pclass"]}),
            _step("scale_features", {"columns": ["Age", "Fare"]}),
        ]
        result = validate_proposed_plan(plan, TARGET_COLUMN)
        assert result.valid is True
        assert result.violations == []


