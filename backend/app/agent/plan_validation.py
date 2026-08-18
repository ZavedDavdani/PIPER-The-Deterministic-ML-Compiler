"""
Deterministic tool/argument validation for LLM-proposed plan steps
(M3 Phase 4).

The LLM is untrusted input. Before any LLM-generated PlanStep reaches
execution, this module validates it deterministically — no execution,
no side effects, purely a structural/allowlist check against the
EXISTING deterministic tool implementations. This is the mandatory
central boundary the M3 instructions require: an unknown tool_name (or
malformed/type-wrong arguments) must produce a structured failure, and
must NEVER silently become status="skipped" the way an unrecognized
tool_name currently does in clean_node/feature_engineer_node's
existing if/elif dispatch (that existing dispatch behavior is
UNCHANGED by this module — this is a new, separate gate that runs
BEFORE a plan reaches those nodes at all, once wired in a later phase).

This is deliberately NOT a broad tool-registry rewrite (explicitly out
of scope per the M3 instructions) — it's a single allowlist table plus
a validation function, sized to exactly what's needed to reject an
LLM-hallucinated tool_name/arguments before execution, nothing more.

Only the five tool_names plan_node_v2 has ever produced are in the
allowlist: drop_column, convert_column_type, impute_missing_values,
encode_categorical_features, scale_features. No new tools are
introduced — the LLM can propose only operations this system already
knows how to execute deterministically. It can never propose executing
arbitrary Python, shell commands, or any tool_name outside this fixed
set.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# --- Argument schema definitions --------------------------------------
#
# Each entry describes exactly the arguments the corresponding real
# tool function actually requires (cross-checked against
# app/agent/tools/cleaning_ops.py, type_conversion.py,
# feature_engineering.py's real signatures) — not a reinvented or
# looser contract.

_VALID_IMPUTE_STRATEGIES = frozenset({"mean", "median", "mode"})
_VALID_CONVERT_TARGET_TYPES = frozenset({"numeric", "string", "datetime"})

ALLOWED_TOOL_NAMES = frozenset({
    "drop_column",
    "convert_column_type",
    "impute_missing_values",
    "encode_categorical_features",
    "scale_features",
})
"""
The complete, fixed set of tool_names an LLM-proposed PlanStep may
ever reference. Anything outside this set is rejected — never
executed, never silently skipped.
"""

TOOL_ARGUMENT_SCHEMAS: dict = {
    "drop_column": {
        "description": "Drop exactly ONE column from the dataset.",
        "arguments": {
            "column": {
                "type": "string",
                "required": True,
                "description": "The single column name to drop.",
            },
        },
        "note": (
            "column is a single string, not a list. To drop multiple "
            "columns, propose one drop_column step PER column — never "
            "pass a list of names here."
        ),
        "example": {"column": "PassengerId"},
    },
    "convert_column_type": {
        "description": "Convert exactly ONE column to a different type.",
        "arguments": {
            "column": {
                "type": "string",
                "required": True,
                "description": "The single column name to convert.",
            },
            "target_type": {
                "type": "string",
                "required": True,
                "enum": sorted(_VALID_CONVERT_TARGET_TYPES),
                "description": "The type to convert the column to.",
            },
        },
        "example": {"column": "Age", "target_type": "numeric"},
    },
    "impute_missing_values": {
        "description": "Fill missing values in exactly ONE column.",
        "arguments": {
            "column": {
                "type": "string",
                "required": True,
                "description": "The single column name to impute.",
            },
            "strategy": {
                "type": "string",
                "required": True,
                "enum": sorted(_VALID_IMPUTE_STRATEGIES),
                "description": "How to compute the fill value.",
            },
        },
        "example": {"column": "Age", "strategy": "median"},
    },
    "encode_categorical_features": {
        "description": "One-hot encode one or more categorical columns in a single step.",
        "arguments": {
            "columns": {
                "type": "array of strings",
                "required": True,
                "description": "One or more column names to encode.",
            },
        },
        "note": (
            "columns is a list, even for a single column (e.g. "
            '["Sex"]). Use ONE encode_categorical_features step for '
            "every column you want encoded — do not call this tool "
            "once per column."
        ),
        "example": {"columns": ["Sex", "Embarked"]},
    },
    "scale_features": {
        "description": "Standard-scale one or more numeric columns in a single step.",
        "arguments": {
            "columns": {
                "type": "array of strings",
                "required": True,
                "description": "One or more column names to scale.",
            },
        },
        "note": (
            "columns is a list, even for a single column (e.g. "
            '["Fare"]). Use ONE scale_features step for every column '
            "you want scaled — do not call this tool once per column."
        ),
        "example": {"columns": ["Age", "Fare"]},
    },
}
"""
Declarative, LLM-facing description of exactly what
_validate_step_arguments() below actually enforces per tool — the
single source of truth this module's own validation logic is built
from, rendered into the planner prompt (app/llm/prompts.py) so the LLM
sees the SAME contract the deterministic validator checks against,
instead of guessing from bare tool names alone.

Genuine finding (post-4-model-benchmark investigation, 20 real Ollama
calls across qwen3:4b/qwen3.5:4b/llama3.2:3b/gemma3:4b): the prompt
previously rendered `context.allowed_operations` as nothing but a bare
JSON array of tool_name strings — no argument names, types,
required/optional status, or examples were ever communicated. Every
one of the 20 real calls converged on structurally similar guessing
failures (drop_column given a `columns` list instead of singular
`column`; invented field names like `method`/`column_names`/
`columns_to_encode`) — independent evidence across four different
model architectures that this was a genuine prompt-contract gap, not a
model-capability problem. TestToolArgumentSchemasMatchValidator (see
tests/test_plan_validation.py) asserts this dict can never silently
drift from the real validation logic below — every declared "example"
is round-tripped through the real validate_proposed_plan() and
asserted valid, and every declared enum is asserted equal to this
module's own _VALID_*_STRATEGIES/_VALID_*_TARGET_TYPES constants.

Deliberately NOT used to relax, replace, or bypass
_validate_step_arguments() in any way — it is documentation data
consumed only by the prompt layer. Validation remains the sole,
unweakened authority; nothing here is ever trusted at execution time.
"""


class ToolValidationViolation(BaseModel):
    """
    One specific reason a proposed step was rejected. `field` is the
    argument key involved (or "tool_name" for a tool-level violation),
    so the evidence is precise enough for an LLM to correct on REPLAN,
    not just a bare "invalid step" message.
    """

    model_config = ConfigDict(extra="forbid")

    step_index: int = Field(..., ge=0, description="Index of the offending step within the proposed plan.")
    tool_name: str
    field: str
    reason: str


class PlanValidationResult(BaseModel):
    """
    Output of validate_proposed_plan(). valid=True means every step in
    the proposed plan passed every check in this module — it does NOT
    mean the plan will succeed at execution time (a valid drop_column
    call can still fail for dataset-specific reasons handled by the
    real tool itself); it only means the plan is structurally safe to
    attempt.
    """

    model_config = ConfigDict(extra="forbid")

    valid: bool
    violations: list[ToolValidationViolation] = Field(default_factory=list)


def _validate_step_arguments(
    step_index: int, tool_name: str, arguments: dict, target_column: str
) -> list[ToolValidationViolation]:
    """
    Validates one step's arguments against the real tool it names.
    Only called for tool_names already confirmed to be in
    ALLOWED_TOOL_NAMES — an unrecognized tool_name is rejected before
    this function is ever reached (see validate_proposed_plan()).
    """
    violations: list[ToolValidationViolation] = []

    def _require_str_field(field: str) -> str | None:
        value = arguments.get(field)
        if not isinstance(value, str) or not value:
            violations.append(
                ToolValidationViolation(
                    step_index=step_index, tool_name=tool_name, field=field,
                    reason=f"'{field}' is required and must be a non-empty string.",
                )
            )
            return None
        return value

    def _require_str_list_field(field: str) -> list[str] | None:
        value = arguments.get(field)
        if not isinstance(value, list) or not value or not all(isinstance(v, str) and v for v in value):
            violations.append(
                ToolValidationViolation(
                    step_index=step_index, tool_name=tool_name, field=field,
                    reason=f"'{field}' is required and must be a non-empty list of non-empty strings.",
                )
            )
            return None
        return value

    def _reject_target_as_feature(column_or_columns) -> None:
        """
        Structural leakage guard at the plan-validation level — mirrors
        train_model()'s own target_leakage_in_features check
        (app/agent/tools/training.py), applied earlier, before
        execution, so an LLM proposing to encode/scale/drop-as-feature
        the target column is caught here too, not only deep inside
        training. drop_column is intentionally exempt from this check:
        dropping the target column is rejected by drop_column() itself
        via a DIFFERENT, existing rule (its own explicit
        "cannot drop the target column" contract) — this function does
        not duplicate that; it only guards the FEATURE-list arguments
        used by encode/scale.
        """
        columns = column_or_columns if isinstance(column_or_columns, list) else [column_or_columns]
        if target_column in columns:
            violations.append(
                ToolValidationViolation(
                    step_index=step_index, tool_name=tool_name, field="columns",
                    reason=(
                        f"Target column '{target_column}' cannot be listed as a feature "
                        f"for '{tool_name}' — this would be structural leakage."
                    ),
                )
            )

    if tool_name == "drop_column":
        _require_str_field("column")
        # "reason" is optional in the real tool (falls back to
        # step.reasoning if absent — see clean_node), so it is not
        # required here.

    elif tool_name == "convert_column_type":
        _require_str_field("column")
        target_type = arguments.get("target_type")
        if target_type not in _VALID_CONVERT_TARGET_TYPES:
            violations.append(
                ToolValidationViolation(
                    step_index=step_index, tool_name=tool_name, field="target_type",
                    reason=f"'target_type' must be one of {sorted(_VALID_CONVERT_TARGET_TYPES)}, got {target_type!r}.",
                )
            )

    elif tool_name == "impute_missing_values":
        _require_str_field("column")
        strategy = arguments.get("strategy")
        if strategy not in _VALID_IMPUTE_STRATEGIES:
            violations.append(
                ToolValidationViolation(
                    step_index=step_index, tool_name=tool_name, field="strategy",
                    reason=f"'strategy' must be one of {sorted(_VALID_IMPUTE_STRATEGIES)}, got {strategy!r}.",
                )
            )

    elif tool_name in ("encode_categorical_features", "scale_features"):
        columns = _require_str_list_field("columns")
        if columns is not None:
            _reject_target_as_feature(columns)

    return violations


def validate_proposed_plan(
    proposed_steps: list,
    target_column: str,
) -> PlanValidationResult:
    """
    Validates a list of provider-layer ProposedPlanStep objects (or
    anything exposing the same .tool_name/.arguments attributes — kept
    duck-typed deliberately, so this module has no import dependency
    on app.llm.provider, preserving the layering: provider layer
    doesn't know about validation, validation doesn't need to know
    about the provider layer's exact types).

    Rejects (all deterministic, no LLM judgment involved):
    - any tool_name outside ALLOWED_TOOL_NAMES (an unknown/hallucinated
      tool_name — including anything resembling arbitrary code
      execution, e.g. "exec_python"/"run_shell_command"/"eval" — is
      rejected here structurally, since it can never match a name in
      the fixed allowlist)
    - missing/wrong-typed required arguments for the named tool
    - an invalid enum-like value (unsupported strategy/target_type)
    - the target column listed as a feature for encode/scale

    Returns a PlanValidationResult with EVERY violation found across
    EVERY step, not just the first — so a REPLAN attempt has complete
    evidence to correct, not just one problem at a time.
    """
    all_violations: list[ToolValidationViolation] = []

    for index, step in enumerate(proposed_steps):
        tool_name = getattr(step, "tool_name", None)
        arguments = getattr(step, "arguments", None)

        if not isinstance(tool_name, str) or tool_name not in ALLOWED_TOOL_NAMES:
            all_violations.append(
                ToolValidationViolation(
                    step_index=index,
                    tool_name=str(tool_name),
                    field="tool_name",
                    reason=(
                        f"'{tool_name}' is not a recognized operation. Allowed operations: "
                        f"{sorted(ALLOWED_TOOL_NAMES)}."
                    ),
                )
            )
            continue  # cannot validate arguments against an unknown tool

        if not isinstance(arguments, dict):
            all_violations.append(
                ToolValidationViolation(
                    step_index=index, tool_name=tool_name, field="arguments",
                    reason="'arguments' must be an object/dict.",
                )
            )
            continue

        all_violations.extend(_validate_step_arguments(index, tool_name, arguments, target_column))

    return PlanValidationResult(valid=len(all_violations) == 0, violations=all_violations)
