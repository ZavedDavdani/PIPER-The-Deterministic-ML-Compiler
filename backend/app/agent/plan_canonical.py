"""
Canonical executable plan representation (Phase 2 of the deterministic-
core completion batch).

Purpose: derive a deterministic identity for a plan based ONLY on what
it will actually DO — never on natural-language rationale, generated
IDs, timestamps, or commentary. Two plans with identical executable
semantics (even if a human or LLM wrote completely different reasoning
for them) must produce the SAME canonical identity. This is what makes
duplicate-plan detection (Phase 4) possible and honest: it compares
what would actually execute, not how it was explained.

ORDER SENSITIVITY (explicit, not a blanket sort-everything policy):
    cleaning/feature-engineering step SEQUENCE  -> ORDER-SENSITIVE
        (dropping a column before converting another's type can behave
        differently than the reverse order in principle, and even
        where it wouldn't, treating order as significant is the safe
        default for a sequence of mutating operations)
    the SET of steps at the tool_name+arguments level -> compared as
        an ordered sequence, not sorted, for exactly this reason

    model candidate list -> ORDER-INSENSITIVE
        (training model A then B produces the same two trained models
        as training B then A; candidate set membership is what matters,
        not the order they were listed in)
"""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field

from app.agent.state import PlanStep


class CanonicalPlanStep(BaseModel):
    """
    The execution-relevant subset of a PlanStep. Deliberately excludes
    step_id (generated), action (human-readable label), reasoning
    (rationale), and status (runtime state) — none of these affect
    what will actually execute.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_name: str
    arguments: dict = Field(default_factory=dict)


class CanonicalPlan(BaseModel):
    """
    The full canonical, execution-relevant representation of a plan.

    steps is an ORDERED sequence (cleaning/feature-engineering
    operations are order-sensitive — see module docstring). It is
    never sorted.

    model_candidates is stored as a SORTED tuple of (algorithm,
    sorted-params) pairs specifically because model training order
    doesn't change what gets trained — this is the one deliberately
    order-INSENSITIVE field, and the sorting is what makes that
    insensitivity actually hold under hashing (two plans listing the
    same two models in different orders must hash identically).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_column: str
    steps: tuple[CanonicalPlanStep, ...] = Field(default_factory=tuple)
    model_candidates: tuple[tuple, ...] = Field(
        default_factory=tuple,
        description="Sorted tuple of (algorithm, sorted-params-as-tuple) pairs. Order-insensitive by construction.",
    )

    def canonical_json(self) -> str:
        """
        Deterministic JSON serialization — sorted keys, no whitespace
        variance, so identical CanonicalPlan content always produces
        byte-identical JSON regardless of construction order or
        Python dict iteration quirks.
        """
        payload = {
            "target_column": self.target_column,
            "steps": [
                {"tool_name": s.tool_name, "arguments": _sorted_json_safe(s.arguments)}
                for s in self.steps
            ],
            "model_candidates": [list(mc) for mc in self.model_candidates],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def plan_hash(self) -> str:
        """
        Deterministic SHA-256 hash of the canonical JSON — this is the
        plan identity used for duplicate detection (Phase 4). Same
        executable content -> same hash, regardless of rationale,
        step_id, timestamps, or construction order of independent
        fields.
        """
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _sorted_json_safe(value):
    """
    Recursively normalizes a value for deterministic JSON
    serialization: dict keys sorted, lists left in their given order
    (arguments' internal lists, e.g. a column list for
    encode_categorical_features, ARE order-sensitive in the sense that
    they came from the plan step as-authored — we don't second-guess
    that here, only ensure dict key order doesn't introduce spurious
    hash differences).
    """
    if isinstance(value, dict):
        return {k: _sorted_json_safe(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_sorted_json_safe(v) for v in value]
    return value


def canonicalize_plan(
    plan: list,
    target_column: str,
    model_candidates=None,
) -> CanonicalPlan:
    """
    Builds a CanonicalPlan from a real AgentState.plan (list of
    PlanStep) plus the target column and any model candidates under
    consideration for this attempt.

    Steps are kept in their given order (order-sensitive). Only
    steps with status in {"pending", "completed"} are included —
    "skipped"/"failed" steps did not (or will not) actually execute,
    so they carry no executable identity; a plan that skipped a step
    due to an environment quirk is not thereby a "different plan" in
    the sense that matters for duplicate detection.
    """
    included_statuses = {"pending", "completed"}
    canonical_steps = tuple(
        CanonicalPlanStep(tool_name=step.tool_name, arguments=step.arguments)
        for step in plan
        if step.status in included_statuses
    )

    normalized_candidates = tuple(
        sorted(
            (
                (algorithm, tuple(sorted(params.items())))
                for algorithm, params in (model_candidates or [])
            )
        )
    )

    return CanonicalPlan(
        target_column=target_column,
        steps=canonical_steps,
        model_candidates=normalized_candidates,
    )
