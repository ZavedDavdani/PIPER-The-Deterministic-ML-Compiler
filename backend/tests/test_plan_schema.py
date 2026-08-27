"""
Regression tests for strict planner JSON schema and replan violation surfacing.
"""

from __future__ import annotations

import json

import pytest

from app.agent.plan_validation import TOOL_ARGUMENT_SCHEMAS, validate_proposed_plan
from app.llm.plan_schema import build_plan_json_schema
from app.llm.prompts import build_replan_prompt
from app.llm.provider import LLMPlanningContext, ProposedPlan, ProposedPlanStep
from app.llm.gemini_provider import GeminiProvider
from unittest.mock import MagicMock, patch


@pytest.fixture()
def titanic_context() -> LLMPlanningContext:
    return LLMPlanningContext(
        objective="Predict Survived (binary classification).",
        dataset_context={"columns": ["PassengerId", "Survived", "Age", "Sex", "Cabin"], "rows": 891},
        allowed_operations=list(TOOL_ARGUMENT_SCHEMAS.keys()),
        tool_schemas=TOOL_ARGUMENT_SCHEMAS,
    )


class TestPlanJsonSchema:
    def test_drop_column_requires_non_empty_column_in_schema(self):
        schema = build_plan_json_schema(
            ["drop_column"],
            {"drop_column": TOOL_ARGUMENT_SCHEMAS["drop_column"]},
        )
        arguments_schema = schema["properties"]["steps"]["items"]["properties"]["arguments"]
        assert "column" in arguments_schema["properties"]
        assert arguments_schema["properties"]["column"]["minLength"] == 1
        assert "column" in arguments_schema["required"]

    def test_valid_drop_column_plan_parses_and_validates(self):
        plan = ProposedPlan(
            steps=[
                ProposedPlanStep(
                    action="Drop Cabin",
                    tool_name="drop_column",
                    arguments={"column": "Cabin"},
                    reasoning="High missingness.",
                )
            ]
        )
        result = validate_proposed_plan(plan.steps, target_column="Survived")
        assert result.valid is True

    def test_missing_drop_column_column_is_rejected(self):
        plan = ProposedPlan(
            steps=[
                ProposedPlanStep(
                    action="Drop Cabin",
                    tool_name="drop_column",
                    arguments={},
                    reasoning="Bad.",
                )
            ]
        )
        result = validate_proposed_plan(plan.steps, target_column="Survived")
        assert result.valid is False
        assert result.violations[0].field == "column"
        assert "non-empty string" in result.violations[0].reason


class TestGeminiPlannerContract:
    def test_gemini_receives_strict_schema_for_tool_arguments(self, titanic_context):
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "steps": [
                {
                    "action": "Drop Cabin",
                    "tool_name": "drop_column",
                    "arguments": {"column": "Cabin"},
                    "reasoning": "High missingness.",
                }
            ]
        })
        provider = GeminiProvider(api_key="test-key", model="gemini-test")

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = provider.generate_plan(titanic_context)

            assert result.success is True
            config = mock_client.models.generate_content.call_args[1]["config"]
            schema = config.response_json_schema
            step_items = schema["properties"]["steps"]["items"]
            variants = step_items["oneOf"] if "oneOf" in step_items else [step_items]
            drop_schema = next(
                item for item in variants if item["properties"]["tool_name"]["const"] == "drop_column"
            )
            drop_args = drop_schema["properties"]["arguments"]
            assert "column" in drop_args["required"]

    def test_replan_prompt_contains_validation_violation(self, titanic_context):
        replan_context = titanic_context.model_copy(
            update={
                "failure_context": {
                    "category": "EVALUATION_ERROR",
                    "message": "validation failed",
                    "evidence": {
                        "violations": [
                            {
                                "step_index": 0,
                                "tool_name": "drop_column",
                                "field": "column",
                                "reason": "'column' is required and must be a non-empty string.",
                            }
                        ],
                        "rejected_steps": [
                            {"step_index": 0, "tool_name": "drop_column", "arguments": {}}
                        ],
                    },
                }
            }
        )
        prompt = build_replan_prompt(replan_context)
        assert "VALIDATION VIOLATIONS" in prompt
        assert "drop_column.column" in prompt
        assert "'column' is required and must be a non-empty string." in prompt
        assert '"arguments": {}' in prompt
