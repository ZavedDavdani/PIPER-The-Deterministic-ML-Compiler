"""
JSON Schema for structured LLM plan output.

Builds a strict per-tool schema from TOOL_ARGUMENT_SCHEMAS so structured-
output providers (Gemini, OpenAI, Ollama) cannot emit empty argument objects
like drop_column(arguments={}).
"""

from __future__ import annotations

from app.agent.plan_validation import TOOL_ARGUMENT_SCHEMAS

# Legacy fallback when no tool schemas are available (tests / bare callers).
PLAN_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "tool_name": {"type": "string"},
                    "arguments": {"type": "object"},
                    "reasoning": {"type": "string"},
                },
                "required": ["action", "tool_name", "arguments", "reasoning"],
            },
        },
    },
    "required": ["steps"],
}


def _argument_property_schema(arg_spec: dict) -> dict:
    arg_type = arg_spec.get("type", "string")
    if "array" in str(arg_type):
        return {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "minItems": 1,
        }
    schema: dict = {"type": "string", "minLength": 1}
    if enum_vals := arg_spec.get("enum"):
        schema["enum"] = list(enum_vals)
    return schema


def _tool_step_schema(tool_name: str, tool_def: dict) -> dict:
    arg_properties: dict = {}
    required_arguments: list[str] = []
    for arg_name, arg_spec in tool_def.get("arguments", {}).items():
        arg_properties[arg_name] = _argument_property_schema(arg_spec)
        if arg_spec.get("required"):
            required_arguments.append(arg_name)

    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "minLength": 1},
            "tool_name": {"const": tool_name},
            "arguments": {
                "type": "object",
                "properties": arg_properties,
                "required": required_arguments,
                "additionalProperties": False,
            },
            "reasoning": {"type": "string", "minLength": 1},
        },
        "required": ["action", "tool_name", "arguments", "reasoning"],
        "additionalProperties": False,
    }


def build_plan_json_schema(
    allowed_operations: list[str] | None = None,
    tool_schemas: dict | None = None,
) -> dict:
    """
    Build a strict JSON schema for ProposedPlan structured output.

    When tool schemas are available, each step must match exactly one
    allowed tool variant with all required arguments present and non-empty.
    """
    schemas = tool_schemas if tool_schemas else TOOL_ARGUMENT_SCHEMAS
    operations = list(allowed_operations) if allowed_operations else list(schemas.keys())
    step_variants = [
        _tool_step_schema(tool_name, schemas[tool_name])
        for tool_name in operations
        if tool_name in schemas
    ]
    if not step_variants:
        return PLAN_JSON_SCHEMA

    step_items: dict = step_variants[0] if len(step_variants) == 1 else {"oneOf": step_variants}

    return {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": step_items,
                "minItems": 1,
            }
        },
        "required": ["steps"],
        "additionalProperties": False,
    }
