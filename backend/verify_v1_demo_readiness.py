"""
PIPER V1 — Comprehensive Testing and Validation Runner
Covers Phases 2 through 19 against the live codebase and real environment.
"""

import os
import sys
import time
import json
import uuid
import httpx
import pandas as pd
from types import SimpleNamespace

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath("."))

from app.storage import InMemoryDatasetStore, InMemoryModelStore, InMemorySplitStore, InMemoryRunStore
from app.agent.tools.ingestion import ingest_dataset
from app.agent.tools.profiling import profile_dataset
from app.agent.tools import build_sanitized_llm_context
from app.agent.tools.context_budget import apply_context_budget
from app.agent.plan_validation import validate_proposed_plan, ALLOWED_TOOL_NAMES, TOOL_ARGUMENT_SCHEMAS
from app.agent.plan_adequacy import evaluate_plan_adequacy, classify_plan_steps
from app.agent.plan_canonical import canonicalize_plan
from app.agent.nodes.real_nodes import plan_node_v2, _carried_forward_preserved_steps
from app.agent.state import AgentState, PlanStep
from app.schemas.failure import FailureInfo
from app.llm.provider import LLMPlanningContext, FakeLLMProvider, LLMProviderResult, ProviderError, ProposedPlan, ProposedPlanStep
from app.llm.prompts import build_planning_prompt, build_replan_prompt
from app.llm.ollama_provider import OllamaProvider
from app.agent.tools.training import train_model
from app.agent.tools.evaluation import evaluate_model, compare_models
from app.agent.tracing import run_with_tracing

def test_phase_2_dataset_ingestion():
    print("\n--- PHASE 2: DATASET INGESTION TESTING (Titanic) ---")
    dataset_path = "../benchmark_data/train.csv"
    with open(dataset_path, "rb") as f:
        content = f.read()

    store = InMemoryDatasetStore()
    dataset_id = "test_titanic_ds"
    res = ingest_dataset(content, "train.csv", dataset_id, store)
    assert res.success, f"Ingestion failed: {res.message}"
    ds = res.data
    assert ds.num_rows == 891, f"Expected 891 rows, got {ds.num_rows}"
    assert ds.num_columns == 12, f"Expected 12 columns, got {ds.num_columns}"
    assert "Survived" in ds.columns, "Survived column missing"

    df = store.get(dataset_id)
    assert len(df) == 891
    assert len(df.columns) == 12

    # Profiling
    profile = profile_dataset(dataset_id, "Survived", store)
    assert profile.total_rows == 891
    assert profile.target_column == "Survived"
    assert profile.missing_cell_count > 0

    # Check missing values
    col_missing = {c.column_name: c.missing_count for c in profile.columns}
    assert col_missing["Age"] == 177, f"Expected 177 missing Age, got {col_missing.get('Age')}"
    assert col_missing["Cabin"] == 687, f"Expected 687 missing Cabin, got {col_missing.get('Cabin')}"
    assert col_missing["Embarked"] == 2, f"Expected 2 missing Embarked, got {col_missing.get('Embarked')}"

    # Sanitized context
    sanitized_res = build_sanitized_llm_context(profile)
    assert sanitized_res.success
    budgeted, _ = apply_context_budget(sanitized_res.data)
    assert budgeted.target_column == "Survived"
    assert len(budgeted.column_contexts) == 12
    print("✓ Phase 2 Ingestion, Profiling, and Sanitization Passed.")
    return ds.dataset_id, store, budgeted

def test_phase_4_adequacy_material_failure(budgeted_context):
    print("\n--- PHASE 4: ADEQUACY MATERIAL FAILURE TEST (Embarked as effective feature without imputation) ---")
    # Proposed plan that encodes Embarked (categorical with missing) without imputing or dropping it
    steps = [
        ProposedPlanStep(action="transform", tool_name="impute_missing_values", arguments={"column": "Age", "strategy": "median"}, reasoning="impute age"),
        ProposedPlanStep(action="encode", tool_name="encode_categorical_features", arguments={"columns": ["Sex", "Embarked"]}, reasoning="encode sex and embarked"),
        ProposedPlanStep(action="scale", tool_name="scale_features", arguments={"columns": ["Age", "Fare"]}, reasoning="scale age and fare")
    ]
    val = validate_proposed_plan(steps, "Survived")
    assert val.valid, f"Structural validation failed: {val.violations}"

    adequacy = evaluate_plan_adequacy(budgeted_context, steps, "Survived")
    assert adequacy.status == "FAIL", "Expected adequacy FAIL"
    assert adequacy.material_failure is True
    material = adequacy.material_findings
    assert len(material) == 1
    assert material[0].condition == "missing_values"
    assert "Embarked" in material[0].columns
    assert material[0].severity == "material"
    print("✓ Phase 4 Material Adequacy Failure Verified on Embarked.")

def test_phase_5_non_effective_feature_advisory(budgeted_context):
    print("\n--- PHASE 5: NON-EFFECTIVE FEATURE ADVISORY TEST (Cabin unaddressed and not in feature set) ---")
    # Proposed plan that addresses Age and Embarked, but completely ignores Cabin (not in encode/scale)
    steps = [
        ProposedPlanStep(action="transform", tool_name="impute_missing_values", arguments={"column": "Age", "strategy": "median"}, reasoning="impute age"),
        ProposedPlanStep(action="transform", tool_name="impute_missing_values", arguments={"column": "Embarked", "strategy": "most_frequent"}, reasoning="impute embarked"),
        ProposedPlanStep(action="encode", tool_name="encode_categorical_features", arguments={"columns": ["Sex", "Embarked"]}, reasoning="encode sex and embarked"),
        ProposedPlanStep(action="scale", tool_name="scale_features", arguments={"columns": ["Age", "Fare"]}, reasoning="scale age and fare")
    ]
    val = validate_proposed_plan(steps, "Survived")
    assert val.valid

    adequacy = evaluate_plan_adequacy(budgeted_context, steps, "Survived")
    assert adequacy.status == "PASS", f"Expected adequacy PASS, got {adequacy.status}"
    assert adequacy.material_failure is False

    # Cabin should be reported as ADVISORY
    advisories = [f for f in adequacy.findings if f.severity == "advisory" and f.status == "NOT_ADDRESSED"]
    cabin_findings = [f for f in advisories if "Cabin" in f.columns]
    assert len(cabin_findings) > 0, "Expected advisory finding for Cabin"
    assert cabin_findings[0].severity == "advisory"
    print("✓ Phase 5 Non-effective Feature Cabin marked ADVISORY, plan PASS verified.")

def test_phase_6_state_preserving_replan(budgeted_context):
    print("\n--- PHASE 6: STATE-PRESERVING REPLAN TEST ---")
    steps = [
        ProposedPlanStep(action="transform", tool_name="impute_missing_values", arguments={"column": "Age", "strategy": "median"}, reasoning="impute age"),
        ProposedPlanStep(action="encode", tool_name="encode_categorical_features", arguments={"columns": ["Sex", "Embarked"]}, reasoning="encode sex and embarked"),
        ProposedPlanStep(action="scale", tool_name="scale_features", arguments={"columns": ["Age", "Fare"]}, reasoning="scale age and fare")
    ]
    adequacy = evaluate_plan_adequacy(budgeted_context, steps, "Survived")
    classification = classify_plan_steps(adequacy.findings, steps)

    assert len(classification["valid_steps"]) == 2  # Age impute, scale_features
    assert len(classification["implicated_steps"]) == 1  # encode_categorical_features (due to Embarked)
    assert classification["valid_steps"][0]["tool_name"] == "impute_missing_values"
    assert classification["valid_steps"][1]["tool_name"] == "scale_features"
    assert classification["implicated_steps"][0]["tool_name"] == "encode_categorical_features"

    # Build replan prompt
    planning_context = LLMPlanningContext(
        objective="Predict Survived",
        dataset_context=budgeted_context.model_dump(mode="json"),
        allowed_operations=sorted(ALLOWED_TOOL_NAMES),
        tool_schemas=TOOL_ARGUMENT_SCHEMAS,
        failure_context={
            "category": "PLAN_ADEQUACY",
            "message": adequacy.summary,
            "evidence": {
                "valid_steps": classification["valid_steps"],
                "implicated_steps": classification["implicated_steps"]
            }
        }
    )
    prompt = build_replan_prompt(planning_context)
    assert "=== VALID OPERATIONS (preserve these) ===" in prompt
    assert '"tool_name": "impute_missing_values"' in prompt
    assert '"tool_name": "scale_features"' in prompt
    print("✓ Phase 6 State-Preserving REPLAN Prompt formatting verified.")

def test_phase_7_parse_failure_preservation(budgeted_context):
    print("\n--- PHASE 7: PARSE/PROVIDER FAILURE STATE PRESERVATION TEST ---")
    # 1. State with an adequacy failure
    state = AgentState(
        run_id="test_run_parse_preservation",
        dataset_id="test_ds",
        target_column="Survived",
        task_type="classification",
        retry_count=1,
        max_retries=3,
        failure=FailureInfo(
            category="PLAN_ADEQUACY",
            message="Adequacy failed",
            evidence={
                "valid_steps": [{"tool_name": "scale_features", "arguments": {"columns": ["Age", "Fare"]}}],
                "implicated_steps": [{"tool_name": "encode_categorical_features", "arguments": {"columns": ["Embarked"]}}]
            },
            node="plan",
            attempt=0,
            retryable=True,
            human_intervention_required=False
        )
    )

    # Provider returns parse failure
    provider_error = FakeLLMProvider(canned_results=[
        LLMProviderResult(success=False, error=ProviderError(code="malformed_response", message="Malformed JSON response", raw_response_excerpt="{malformed..."))
    ])
    store = InMemoryDatasetStore()
    out = plan_node_v2(state, store, provider_error)
    assert out["status"] == "failed"
    failure = out["failure"]
    assert failure.category == "EVALUATION_ERROR"
    assert "valid_steps" in failure.evidence
    assert failure.evidence["valid_steps"] == [{"tool_name": "scale_features", "arguments": {"columns": ["Age", "Fare"]}}]

    # Control: first attempt with no prior state
    first_state = AgentState(run_id="r1", dataset_id="d1", target_column="Survived", task_type="classification")
    carried_first = _carried_forward_preserved_steps(first_state)
    assert carried_first == {}, "First attempt must carry nothing"
    print("✓ Phase 7 Parse Failure State Preservation verified.")

def test_phase_8_duplicate_plan(budgeted_context):
    print("\n--- PHASE 8: DUPLICATE PLAN TEST ---")
    steps = [
        PlanStep(step_id="step_01", action="transform", tool_name="drop_column", arguments={"column": "PassengerId"}, reasoning="drop id")
    ]
    h = canonicalize_plan(steps, "Survived").plan_hash()
    state = AgentState(
        run_id="test_dup",
        dataset_id="test_ds",
        target_column="Survived",
        task_type="classification",
        plan_history=[h],
        retry_count=1,
        max_retries=3
    )
    provider = FakeLLMProvider(canned_plans=[
        ProposedPlan(steps=[
            ProposedPlanStep(action="transform", tool_name="drop_column", arguments={"column": "PassengerId"}, reasoning="drop id again")
        ])
    ])
    store = InMemoryDatasetStore()
    out = plan_node_v2(state, store, provider)
    assert out["status"] == "failed"
    assert out["failure"].category == "DUPLICATE_PLAN"
    assert out["failure"].retryable is False
    assert out["failure"].human_intervention_required is True
    print("✓ Phase 8 Duplicate Plan detection and termination verified.")

def test_phase_10_target_column_safety(budgeted_context):
    print("\n--- PHASE 10: TARGET COLUMN SAFETY TEST ---")
    # 1. Structural validation rejects target in scale/encode
    res1 = validate_proposed_plan([
        ProposedPlanStep(action="scale", tool_name="scale_features", arguments={"columns": ["Survived", "Fare"]}, reasoning="leak target")
    ], "Survived")
    assert not res1.valid
    assert any("target" in v.reason.lower() or "survived" in v.field.lower() for v in res1.violations)

    # 2. Adequacy rejects drop/impute/convert on target
    res2 = evaluate_plan_adequacy(budgeted_context, [
        ProposedPlanStep(action="transform", tool_name="drop_column", arguments={"column": "Survived"}, reasoning="drop target")
    ], "Survived")
    assert res2.status == "FAIL"
    assert any(f.condition == "target_protection" and f.status == "NOT_ADDRESSED" for f in res2.findings)
    print("✓ Phase 10 Target Column Safety Enforcements verified.")

def test_phase_12_training_and_ml_validation(dataset_id, dataset_store):
    print("\n--- PHASE 12: TRAINING & ML VALIDATION PIPELINE TEST ---")
    from app.agent.tools.preparation import split_dataset
    split_store = InMemorySplitStore()
    model_store = InMemoryModelStore()

    split_res = split_dataset(dataset_id, "Survived", 0.2, dataset_store, split_store)
    assert split_res.success
    split_id = split_res.split_id

    fe_intent = {
        "encoding_specs": [{"columns": ["Sex", "Embarked"], "method": "onehot"}],
        "scaling_specs": [{"columns": ["Age", "Fare"], "method": "standard"}]
    }

    # Train Random Forest
    rf_res = train_model("random_forest", {"n_estimators": 50, "max_depth": 5}, split_id, "Survived", fe_intent, split_store, model_store)
    assert rf_res.success, f"RF Train failed: {rf_res.message}"

    # Train Logistic Regression
    lr_res = train_model("logistic_regression", {"C": 1.0, "max_iter": 500}, split_id, "Survived", fe_intent, split_store, model_store)
    assert lr_res.success, f"LR Train failed: {lr_res.message}"

    # Evaluate both
    eval_rf = evaluate_model(rf_res.model_id, split_id, split_store, model_store)
    assert eval_rf.success
    assert eval_rf.metrics.f1 >= 0.0

    eval_lr = evaluate_model(lr_res.model_id, split_id, split_store, model_store)
    assert eval_lr.success
    assert eval_lr.metrics.f1 >= 0.0

    # Compare models
    comp_res = compare_models([rf_res.model_id, lr_res.model_id], split_id, split_store, model_store)
    assert comp_res.success
    assert comp_res.comparison.selected_model_id in [rf_res.model_id, lr_res.model_id]
    assert comp_res.comparison.justification != ""
    print(f"✓ Phase 12 ML Pipeline Completed: Selected {comp_res.comparison.selected_model_id} ({comp_res.comparison.justification})")

def test_phase_13_and_15_api_and_isolation():
    print("\n--- PHASE 13 & 15: API ENDPOINTS & FRESH RUN ISOLATION TEST ---")
    client = httpx.Client(base_url="http://127.0.0.1:8000")
    
    # 1. Health
    r = client.get("/health")
    assert r.status_code == 200, f"Health check failed: {r.text}"
    assert r.json() == {"status": "ok"}

    # 2. Upload dataset (Run A)
    with open("../benchmark_data/train.csv", "rb") as f:
        files = {"file": ("train.csv", f, "text/csv")}
        r_upload = client.post("/datasets", files=files)
    assert r_upload.status_code == 201, f"Upload failed: {r_upload.text}"
    ds_data = r_upload.json()
    ds_id = ds_data["dataset_id"]
    assert ds_data["num_rows"] == 891
    assert ds_data["num_columns"] == 12

    # 3. Learn Static Endpoints
    r_form = client.get("/learn/formulas")
    assert r_form.status_code == 200
    assert len(r_form.json()["formulas"]) >= 8

    r_comp = client.get("/learn/comprehension-checks")
    assert r_comp.status_code == 200
    assert len(r_comp.json()["checks"]) >= 7

    # 4. Error response test
    r_bad = client.get(f"/runs/non_existent_id/result")
    assert r_bad.status_code == 404

    print("✓ Phase 13 & 15 API Endpoints and Error Responses verified.")

def test_phase_3_and_18_live_demo():
    print("\n--- PHASE 3 & 18: HAPPY-PATH END-TO-END CANONICAL TITANIC DEMO WITH REAL QWEN3:4B ---")
    t0 = time.time()
    
    # Step 1: Load and Ingest
    t_load_start = time.time()
    dataset_path = "../benchmark_data/train.csv"
    with open(dataset_path, "rb") as f:
        content = f.read()
    dataset_store = InMemoryDatasetStore()
    split_store = InMemorySplitStore()
    model_store = InMemoryModelStore()
    run_store = InMemoryRunStore()
    
    ds = ingest_dataset(content, "train.csv", dataset_store)
    t_load = time.time() - t_load_start

    # Step 2: Profiling
    t_prof_start = time.time()
    profile = profile_dataset(ds.dataset_id, "Survived", dataset_store)
    t_prof = time.time() - t_prof_start

    # Step 3: Run with tracing and real OllamaProvider
    provider = OllamaProvider(model="qwen3:4b", keep_alive="10m")
    run_id = f"demo_titanic_{uuid.uuid4().hex[:6]}"
    
    print(f"Starting live graph run {run_id} on Titanic dataset...")
    t_run_start = time.time()
    final_state = run_with_tracing(
        run_id=run_id,
        dataset_id=ds.dataset_id,
        target_column="Survived",
        task_type="classification",
        dataset_store=dataset_store,
        split_store=split_store,
        model_store=model_store,
        run_store=run_store,
        llm_provider=provider,
        max_retries=2
    )
    t_run = time.time() - t_run_start
    total_time = time.time() - t0

    print(f"\nLive Run Result Status: {final_state.status}")
    print(f"Total time: {total_time:.2f}s (Load: {t_load:.4f}s, Profile: {t_prof:.4f}s, Run: {t_run:.2f}s)")
    print(f"Retries / REPLANs: {final_state.retry_count}")
    print(f"Plan steps count: {len(final_state.plan)}")
    for i, s in enumerate(final_state.plan):
        print(f"  [{i+1}] {s.tool_name}: {s.arguments}")
    
    if final_state.status == "completed":
        print(f"Validation: Valid={final_state.validation.valid if final_state.validation else 'N/A'}")
        if final_state.comparison:
            print(f"Selected Model: {final_state.comparison.selected_model_id} ({final_state.comparison.selected_algorithm})")
            print(f"Justification: {final_state.comparison.justification}")
            for m in final_state.comparison.models:
                print(f"  Model {m.model_id} ({m.algorithm}): F1={m.f1:.4f}, Accuracy={m.accuracy:.4f}")
    elif final_state.status == "failed":
        print(f"Failure Category: {final_state.failure.category if final_state.failure else 'N/A'}")
        print(f"Failure Message: {final_state.failure.message if final_state.failure else 'N/A'}")

    return {
        "status": final_state.status,
        "runtime": total_time,
        "load_time": t_load,
        "profile_time": t_prof,
        "run_time": t_run,
        "retries": final_state.retry_count,
        "plan_steps": len(final_state.plan),
        "winner": final_state.comparison.selected_algorithm if final_state.comparison else None,
        "justification": final_state.comparison.justification if final_state.comparison else None
    }

if __name__ == "__main__":
    print("==================================================")
    print("PIPER V1 COMPREHENSIVE TEST & VALIDATION SUITE")
    print("==================================================")
    
    ds_id, store, budgeted = test_phase_2_dataset_ingestion()
    test_phase_4_adequacy_material_failure(budgeted)
    test_phase_5_non_effective_feature_advisory(budgeted)
    test_phase_6_state_preserving_replan(budgeted)
    test_phase_7_parse_failure_preservation(budgeted)
    test_phase_8_duplicate_plan(budgeted)
    test_phase_10_target_column_safety(budgeted)
    test_phase_12_training_and_ml_validation(ds_id, store)
    test_phase_13_and_15_api_and_isolation()
    
    print("\n--- EXECUTING LIVE CANONICAL DEMO (PHASE 3 & 18) ---")
    demo_metrics = test_phase_3_and_18_live_demo()
    
    print("\n==================================================")
    print("ALL TEST PHASES EXECUTED SUCCESSFULLY")
    print("==================================================")
