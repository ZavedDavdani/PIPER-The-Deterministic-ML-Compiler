"""
Plan Adequacy schemas — the deterministic "is this plan SUFFICIENT?"
layer, distinct from "is this plan STRUCTURALLY VALID?".

The distinction this module exists to make explicit:

    validate_proposed_plan()  answers: is every step well-formed?
                              (known tool_name, correct argument shape,
                              target not listed as a feature)

    evaluate_plan_adequacy()  answers: given what we deterministically
                              KNOW about this dataset, does the plan
                              actually address the conditions that
                              matter before TRAIN?

A plan can be fully VALID and still be inadequate — this was observed
in real benchmark data, not hypothesised: qwen3.5:4b produced a
single-step plan (`encode_categorical_features(["Sex","Embarked"])`)
that passed every existing check while leaving `Age` (19.87% missing)
entirely unaddressed. See CLAUDE.md's "Plan Adequacy" section.

This layer is READ-ONLY. It never adds, removes, reorders, or rewrites
a plan step, and never drops a column. It produces evidence; the
existing REPLAN loop is what acts on it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AdequacyCondition = Literal[
    "missing_values",
    "target_protection",
    "identifier_like_column",
    "imputation_strategy_compatibility",
    "empty_feature_set",
]
"""
Deterministically detectable dataset conditions this layer evaluates.

Deliberately small. Each entry requires (a) evidence PIPER already
computes, (b) a plan that can be objectively checked against it, and
(c) a material effect on the V1 pipeline. Conditions that would need a
new profiling subsystem, or that amount to generic ML "best practice",
are intentionally excluded — see CLAUDE.md for what was considered and
left out.

empty_feature_set: the plan contains no encode_categorical_features or
scale_features step, so feature_engineer_node would produce an empty
feature matrix and train_model() would fail with "At least one feature
column is required."
"""

AdequacyStatus = Literal["ADDRESSED", "NOT_ADDRESSED", "NOT_APPLICABLE"]
"""
ADDRESSED       — the plan contains an operation that deterministically
                  resolves the condition for the affected column(s).
NOT_ADDRESSED   — the condition holds for this dataset and nothing in
                  the plan resolves it.
NOT_APPLICABLE  — the condition does not arise for this dataset at all
                  (e.g. no column has any missing values).

There is deliberately NO "UNKNOWN" status. Every condition here is
decidable from evidence PIPER already has; a status that means "we
couldn't tell" would hide exactly the kind of gap this layer exists to
surface. In particular, missing-value adequacy is NEVER made
conditional on which estimator might run — see the V1 invariant in
plan_adequacy.py.
"""

AdequacySeverity = Literal["material", "advisory"]
"""
material  — blocks execution and routes into the EXISTING REPLAN loop.
advisory  — recorded as evidence, surfaced to the planner on a REPLAN
            that happens for some other reason, but never itself a
            reason to reject a plan.

Severity is a property of the CONDITION, decided by whether it can
compromise the V1 execution contract — not a per-run judgement call and
never LLM-influenced.
"""


class AdequacyFinding(BaseModel):
    """
    One condition evaluated against one plan, with the deterministic
    evidence that produced it. `evidence` states the measured fact;
    `reason` states the V1 invariant that makes it matter.
    """

    model_config = ConfigDict(extra="forbid")

    condition: AdequacyCondition
    columns: list[str] = Field(
        default_factory=list,
        description="Affected column(s). Empty for a NOT_APPLICABLE finding about the dataset as a whole.",
    )
    status: AdequacyStatus
    severity: AdequacySeverity
    evidence: str = Field(..., description="The measured, deterministic fact — e.g. \"Age has 19.87% missing values\".")
    reason: str = Field(..., description="The V1 invariant this condition relates to, and why the observed plan does or does not satisfy it.")


class PlanAdequacyResult(BaseModel):
    """
    Output of evaluate_plan_adequacy(). `material_failure` is the ONE
    field routing consults — it is True iff at least one finding is
    both `material` and `NOT_ADDRESSED`. `status` is a redundant,
    human-facing mirror of that, kept because it reads clearly in API
    responses and REPLAN evidence.

    Advisory findings never influence `status`/`material_failure`; they
    travel alongside as evidence.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["PASS", "FAIL"]
    material_failure: bool
    findings: list[AdequacyFinding] = Field(default_factory=list)
    summary: str

    @property
    def material_findings(self) -> list[AdequacyFinding]:
        return [f for f in self.findings if f.severity == "material" and f.status == "NOT_ADDRESSED"]
