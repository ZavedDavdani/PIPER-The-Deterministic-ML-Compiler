"""
Formal tests for AgentState — the ad-hoc checks run during state.py
development, made permanent.
"""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from app.agent.state import AgentState, PlanStep


class TestInstantiationAndDefaults:
    def test_instantiates_with_only_required_fields(self):
        state = AgentState(run_id="run_001", dataset_id="dataset_001", target_column="Churn")
        assert state.run_id == "run_001"

    def test_task_type_defaults_to_none(self):
        state = AgentState(run_id="r", dataset_id="d", target_column="Churn")
        assert state.task_type is None

    def test_retry_count_defaults_to_zero(self):
        state = AgentState(run_id="r", dataset_id="d", target_column="Churn")
        assert state.retry_count == 0

    def test_max_retries_defaults_to_two(self):
        state = AgentState(run_id="r", dataset_id="d", target_column="Churn")
        assert state.max_retries == 2

    def test_status_defaults_to_initialized(self):
        state = AgentState(run_id="r", dataset_id="d", target_column="Churn")
        assert state.status == "initialized"

    def test_all_logs_default_to_empty_lists(self):
        state = AgentState(run_id="r", dataset_id="d", target_column="Churn")
        assert state.plan == []
        assert state.cleaning_log == []
        assert state.feature_log == []
        assert state.tool_trace == []
        assert state.model_results == []
        assert state.evaluation_results == []

    def test_error_defaults_to_none(self):
        state = AgentState(run_id="r", dataset_id="d", target_column="Churn")
        assert state.error is None


class TestListIndependenceBetweenInstances:
    """Rules out the classic mutable-default-argument bug."""

    def test_appending_to_one_instance_plan_does_not_affect_another(self):
        state_a = AgentState(run_id="run_a", dataset_id="d1", target_column="Churn")
        state_b = AgentState(run_id="run_b", dataset_id="d2", target_column="Churn")

        state_a.plan.append(
            PlanStep(step_id="s1", action="test", tool_name="drop_column", reasoning="test")
        )

        assert len(state_a.plan) == 1
        assert len(state_b.plan) == 0

    def test_appending_to_one_instance_tool_trace_does_not_affect_another(self):
        state_a = AgentState(run_id="run_a", dataset_id="d1", target_column="Churn")
        state_b = AgentState(run_id="run_b", dataset_id="d2", target_column="Churn")

        assert state_a.cleaning_log is not state_b.cleaning_log
        assert state_a.tool_trace is not state_b.tool_trace


class TestPydanticValidation:
    def test_invalid_task_type_rejected(self):
        with pytest.raises(ValidationError):
            AgentState(
                run_id="r", dataset_id="d", target_column="Churn",
                task_type="multiclass_classification",
            )

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            AgentState(run_id="r", dataset_id="d", target_column="Churn", made_up_field="x")

    def test_valid_binary_classification_task_type_accepted(self):
        state = AgentState(
            run_id="r", dataset_id="d", target_column="Churn",
            task_type="binary_classification",
        )
        assert state.task_type == "binary_classification"

    def test_invalid_status_literal_rejected(self):
        with pytest.raises(ValidationError):
            AgentState(run_id="r", dataset_id="d", target_column="Churn", status="not_a_real_status")


class TestLangGraphCompatibility:
    """
    AgentState must work as a live langgraph StateGraph state_schema,
    not just as a standalone Pydantic model — this was flagged
    explicitly during design and is worth locking in as a real test,
    not just an ad-hoc check that could silently stop being true after
    a langgraph upgrade.
    """

    def test_agentstate_works_as_stategraph_schema(self):
        from langgraph.graph import StateGraph, END

        def dummy_node(state: AgentState) -> dict:
            return {"status": "running", "retry_count": state.retry_count + 1}

        graph = StateGraph(AgentState)
        graph.add_node("dummy", dummy_node)
        graph.set_entry_point("dummy")
        graph.add_edge("dummy", END)
        compiled = graph.compile()

        initial = AgentState(run_id="run_001", dataset_id="dataset_001", target_column="Churn")
        result = compiled.invoke(initial)

        assert result["status"] == "running"
        assert result["retry_count"] == 1

    def test_nested_pydantic_field_is_revalidated_mid_graph(self):
        """
        A node returning a dict for a Pydantic-typed field must be
        re-validated into the real model before the NEXT node sees it
        — confirmed during design, locked here as a regression test.
        """
        from langgraph.graph import StateGraph, END
        from app.schemas.guardrails import PipelineValidationResult

        def node_a(state: AgentState) -> dict:
            val = PipelineValidationResult(
                dataset_id=state.dataset_id, target_column=state.target_column,
                valid=True, checks=[],
            )
            return {"validation": val.model_dump()}

        seen_type = {}

        def node_b(state: AgentState) -> dict:
            seen_type["type"] = type(state.validation)
            return {"status": "completed"}

        graph = StateGraph(AgentState)
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.set_entry_point("a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)
        compiled = graph.compile()

        compiled.invoke(AgentState(run_id="r", dataset_id="d", target_column="Churn"))

        assert seen_type["type"] is PipelineValidationResult
