"""
Real, production node implementations used by graph.py: profile_node
and clean_node.

Historical note: this module was originally named for M2 and also
contained dummy/stub versions of plan_node, train_node, evaluate_node,
and validate_node — those have been REMOVED (not merely deprecated)
after confirming, by direct inspection, that graph.py and every test
file import ONLY profile_node and clean_node from this module. The
real planning/training/evaluation/validation logic lives in
real_nodes.py (plan_node_v2, train_node_v2, evaluate_node_v2,
baseline_node, validate_node_v2, sanitize_node, validate_input_node,
split_node, feature_engineer_node) — that is the sole, authoritative
production path. There is no longer a second, parallel "stub" graph
implementation anywhere in this codebase.

profile_node   -> profile_dataset(), resolves task_type, enforces
                  constraint #9 (non-binary target -> hard failure)
clean_node     -> drop_column(), convert_column_type(),
                  impute_missing_values() against the real dataset,
                  executing whatever cleaning-group steps plan_node_v2
                  produced

Every node function takes AgentState and returns a partial-update
dict, per LangGraph's convention (confirmed against langgraph==1.2.10
in the state.py verification step).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.agent.state import AgentState, OperationRecord, PlanStep, ToolTraceEntry
from app.agent.tools import (
    convert_column_type,
    detect_missing_values,
    drop_column,
    impute_missing_values,
    profile_dataset,
)
from app.storage import DatasetStore

BINARY_TARGET_VALUES_MAX = 2  # a target column with more than 2 distinct values is not binary


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _trace(node: str, tool_name: str, arguments: dict, success: bool, result_summary: dict, duration_ms: float) -> ToolTraceEntry:
    return ToolTraceEntry(
        trace_id=_new_id("trace"),
        timestamp=_now(),
        node=node,
        tool_name=tool_name,
        arguments=arguments,
        result=result_summary,
        success=success,
        duration_ms=duration_ms,
    )


# --- PROFILE (real) ------------------------------------------------------


def profile_node(state: AgentState, store: DatasetStore) -> dict:
    """
    Calls the real profile_dataset() tool, resolves task_type, and
    stores the profile in state.

    Constraint #9 enforced here: if the target column doesn't resolve
    to a genuine binary classification problem, task_type stays None
    and status becomes "failed" immediately — no replan, since this is
    a scope violation, not a recoverable mistake.
    """
    result = profile_dataset(state.dataset_id, store)
    trace = _trace(
        node="profiler",
        tool_name="profile_dataset",
        arguments={"dataset_id": state.dataset_id},
        success=result.success,
        result_summary={"message": result.message},
        duration_ms=0.0,
    )

    if not result.success:
        return {
            "status": "failed",
            "task_type": None,
            "error": f"Profiling failed: {result.error.message}",
            "tool_trace": state.tool_trace + [trace],
        }

    profile_dict = result.data.model_dump()

    # Resolve task_type by inspecting the target column directly
    # (profile_dataset gives us column-level info but not target
    # cardinality in a form we can use without re-reading the data;
    # this uses the store directly, deliberately, since task-type
    # resolution is a control-flow decision, not a tool call).
    df = store.get(state.dataset_id)

    if state.target_column not in df.columns:
        return {
            "status": "failed",
            "task_type": None,
            "error": f"Target column '{state.target_column}' does not exist in the dataset.",
            "profile": profile_dict,
            "tool_trace": state.tool_trace + [trace],
        }

    distinct_target_values = df[state.target_column].dropna().nunique()

    if distinct_target_values != BINARY_TARGET_VALUES_MAX:
        return {
            "status": "failed",
            "task_type": None,
            "error": (
                f"Target column '{state.target_column}' has "
                f"{distinct_target_values} distinct value(s); V1 only "
                f"supports binary classification (exactly 2). This is "
                f"a scope violation, not a recoverable error — no retry."
            ),
            "profile": profile_dict,
            "tool_trace": state.tool_trace + [trace],
        }

    return {
        "status": "running",
        "task_type": "binary_classification",
        "profile": profile_dict,
        "tool_trace": state.tool_trace + [trace],
    }



# --- CLEAN/ENGINEER (real, executes the plan) ----------------------------


def clean_node(state: AgentState, store: DatasetStore) -> dict:
    """
    Executes each PlanStep whose tool_name is a cleaning tool, in
    order, against the real dataset. Feature-engineering tools
    (encode_categorical_features etc.) don't exist yet, so any plan
    step referencing them is marked "skipped", not silently dropped.
    """
    cleaning_log: list[OperationRecord] = []
    new_trace: list[ToolTraceEntry] = []
    updated_plan: list[PlanStep] = []

    for step in state.plan:
        if step.tool_name == "drop_column":
            result = drop_column(
                state.dataset_id,
                step.arguments["column"],
                step.arguments.get("reason", step.reasoning),
                store,
                target_column=state.target_column,
            )
        elif step.tool_name == "convert_column_type":
            result = convert_column_type(
                state.dataset_id,
                step.arguments["column"],
                step.arguments["target_type"],
                store,
            )
        elif step.tool_name == "impute_missing_values":
            # Only meaningful after the preceding conversion step has
            # actually run and produced NaNs — check first, since
            # calling this with nothing missing is a defined error
            # (no_missing_values), not a crash, but we don't want a
            # noisy "failed" cleaning_log entry for a no-op.
            missing_check = detect_missing_values(state.dataset_id, store)
            column = step.arguments["column"]
            has_missing = any(
                e.column == column for e in missing_check.data.columns_with_missing
            ) if missing_check.success else False

            if not has_missing:
                updated_plan.append(step.model_copy(update={"status": "skipped"}))
                continue

            result = impute_missing_values(
                state.dataset_id,
                column,
                step.arguments["strategy"],
                store,
            )
        else:
            # Feature-engineering / other tools not yet implemented.
            updated_plan.append(step.model_copy(update={"status": "skipped"}))
            continue

        new_trace.append(
            _trace(
                node="cleaner",
                tool_name=step.tool_name,
                arguments=step.arguments,
                success=result.success,
                result_summary={"message": result.message},
                duration_ms=0.0,
            )
        )

        if result.success:
            cleaning_log.append(
                OperationRecord(
                    operation_id=_new_id("op"),
                    tool_name=step.tool_name,
                    arguments=step.arguments,
                    result_summary=result.message,
                    reason=step.reasoning,
                    timestamp=_now(),
                )
            )
            updated_plan.append(step.model_copy(update={"status": "completed"}))
        else:
            updated_plan.append(step.model_copy(update={"status": "failed"}))

    return {
        "status": "running",
        "plan": updated_plan,
        "cleaning_log": state.cleaning_log + cleaning_log,
        "tool_trace": state.tool_trace + new_trace,
    }

