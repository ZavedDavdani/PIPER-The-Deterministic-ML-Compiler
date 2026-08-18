"""
PlanDiff (Phase 3 of the deterministic-core completion batch).

Compares two CanonicalPlan objects and reports what changed, using
ONLY executable semantics — tool_name + arguments per step. Natural-
language rationale is never part of the comparison because it was
already excluded at canonicalization time (Phase 2); PlanDiff doesn't
need to re-exclude it, it simply never sees it.

A step is identified for diff purposes by its (tool_name, arguments)
pair as a whole — two steps are "the same operation" only if both
match exactly. Changing a single argument value (e.g.
strategy="median" -> strategy="mean") is reported as a CHANGED entry
when the same tool_name appears in both plans with different
arguments and isn't otherwise accounted for as unchanged; unmatched
tool_name occurrences fall back to ADDED/REMOVED.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.agent.plan_canonical import CanonicalPlan, CanonicalPlanStep


class PlanDiffEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    arguments: dict = Field(default_factory=dict)


class ChangedStepEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    old_arguments: dict
    new_arguments: dict


class PlanDiff(BaseModel):
    """
    Output of diff_plans(). is_duplicate=True iff old_plan.plan_hash()
    == new_plan.plan_hash() — a purely deterministic, hash-based
    comparison of executable semantics, never a judgment call and
    never something an LLM decides.
    """

    model_config = ConfigDict(extra="forbid")

    old_plan_hash: str
    new_plan_hash: str
    is_duplicate: bool
    added: list[PlanDiffEntry] = Field(default_factory=list)
    removed: list[PlanDiffEntry] = Field(default_factory=list)
    changed: list[ChangedStepEntry] = Field(default_factory=list)
    unchanged: list[PlanDiffEntry] = Field(default_factory=list)


def diff_plans(old_plan: CanonicalPlan, new_plan: CanonicalPlan) -> PlanDiff:
    """
    Deterministic, executable-semantics-only diff between two
    canonical plans.
    """
    old_plan_hash = old_plan.plan_hash()
    new_plan_hash = new_plan.plan_hash()
    is_duplicate = old_plan_hash == new_plan_hash

    if is_duplicate:
        unchanged = [PlanDiffEntry(tool_name=s.tool_name, arguments=s.arguments) for s in new_plan.steps]
        return PlanDiff(
            old_plan_hash=old_plan_hash, new_plan_hash=new_plan_hash, is_duplicate=True,
            added=[], removed=[], changed=[], unchanged=unchanged,
        )

    def _hashable(value):
        """
        Recursively converts a value into a hashable form for use as
        a dict key component. Real tool arguments legitimately include
        list values (e.g. scale_features'/encode_categorical_features'
        columns=[...]), which are not hashable as-is — this was a real
        bug caught by testing against those exact tools' argument
        shapes, not assumed away.
        """
        if isinstance(value, dict):
            return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
        if isinstance(value, list):
            return tuple(_hashable(v) for v in value)
        return value

    def step_key(step: CanonicalPlanStep) -> tuple:
        return (step.tool_name, tuple(sorted((k, _hashable(v)) for k, v in step.arguments.items())))

    old_by_key: dict[tuple, list[CanonicalPlanStep]] = {}
    for s in old_plan.steps:
        old_by_key.setdefault(step_key(s), []).append(s)

    new_by_key: dict[tuple, list[CanonicalPlanStep]] = {}
    for s in new_plan.steps:
        new_by_key.setdefault(step_key(s), []).append(s)

    added: list[PlanDiffEntry] = []
    removed: list[PlanDiffEntry] = []
    unchanged: list[PlanDiffEntry] = []

    seen_keys_in_order: list[tuple] = []
    for s in new_plan.steps:
        k = step_key(s)
        if k not in seen_keys_in_order:
            seen_keys_in_order.append(k)
    for s in old_plan.steps:
        k = step_key(s)
        if k not in seen_keys_in_order:
            seen_keys_in_order.append(k)

    for key in seen_keys_in_order:
        old_steps_for_key = old_by_key.get(key, [])
        new_steps_for_key = new_by_key.get(key, [])
        old_count = len(old_steps_for_key)
        new_count = len(new_steps_for_key)
        tool_name = key[0]
        # Use the ORIGINAL arguments dict from a real step instance
        # (whichever side has one), not a reconstruction from the
        # hashable key form — the key form converts lists to tuples
        # for hashing purposes, which must never leak into the
        # displayed/reported arguments.
        source_step = (new_steps_for_key or old_steps_for_key)[0]
        arguments = source_step.arguments

        if old_count > 0 and new_count > 0:
            unchanged.append(PlanDiffEntry(tool_name=tool_name, arguments=arguments))
            extra_new = new_count - old_count
            extra_old = old_count - new_count
            for _ in range(max(extra_new, 0)):
                added.append(PlanDiffEntry(tool_name=tool_name, arguments=arguments))
            for _ in range(max(extra_old, 0)):
                removed.append(PlanDiffEntry(tool_name=tool_name, arguments=arguments))
        elif new_count > 0:
            for _ in range(new_count):
                added.append(PlanDiffEntry(tool_name=tool_name, arguments=arguments))
        elif old_count > 0:
            for _ in range(old_count):
                removed.append(PlanDiffEntry(tool_name=tool_name, arguments=arguments))

    # "changed" — pair up same-tool_name added/removed entries that
    # aren't already accounted for as unchanged. Best-effort,
    # first-available matching by tool_name.
    changed: list[ChangedStepEntry] = []
    removed_by_tool: dict[str, list[PlanDiffEntry]] = {}
    for r in removed:
        removed_by_tool.setdefault(r.tool_name, []).append(r)

    remaining_added: list[PlanDiffEntry] = []
    for a in added:
        candidates = removed_by_tool.get(a.tool_name)
        if candidates:
            old_entry = candidates.pop(0)
            changed.append(ChangedStepEntry(
                tool_name=a.tool_name, old_arguments=old_entry.arguments, new_arguments=a.arguments,
            ))
        else:
            remaining_added.append(a)

    remaining_removed: list[PlanDiffEntry] = []
    for entries in removed_by_tool.values():
        remaining_removed.extend(entries)

    return PlanDiff(
        old_plan_hash=old_plan_hash,
        new_plan_hash=new_plan_hash,
        is_duplicate=False,
        added=remaining_added,
        removed=remaining_removed,
        changed=changed,
        unchanged=unchanged,
    )
