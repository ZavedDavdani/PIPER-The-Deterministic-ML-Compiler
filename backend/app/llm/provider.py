"""
LLMProvider abstraction (M3 Phase A / Step 2).

The LLM is a PLANNER/PROPOSER ONLY. Every provider implementation in
this module returns a raw proposal — a list of dicts intended to
become PlanStep instances — never anything that executes code, never
anything that mutates AgentState, never anything that chooses graph
routing. Pydantic validation of the proposal into real PlanStep
objects, deterministic tool/argument validation, canonicalization,
and duplicate-plan detection all happen OUTSIDE this module (in
plan_node_v2, wired in a later step) — this module's job stops at
"did the provider return something plan-shaped, or did it fail."

This module is deliberately NOT wired into graph.py or plan_node_v2
yet (that's a later step) — it exists standalone, tested purely
through FakeLLMProvider and (separately, later) a live-Ollama-gated
integration test.
"""

from __future__ import annotations

from typing import Literal, Optional, Protocol

from pydantic import BaseModel, ConfigDict, Field

# --- Request context (what the provider receives) -------------------------


class LLMPlanningContext(BaseModel):
    """
    What a provider needs to propose a plan. Deliberately does NOT
    include a raw pandas DataFrame or unsanitized dataset content —
    building the actual sanitized context view is a later step (per
    the M3 instructions, "sanitized LLM context" is explicitly out of
    scope for this step). For now, dataset_context is an opaque,
    already-prepared string/dict the CALLER is responsible for having
    sanitized before construction — this module does not sanitize
    anything itself, and does not claim to.

    failure_context / previous_plan_summary are optional and only
    populated on a REPLAN attempt (also wired in a later step) — kept
    here now so the schema doesn't need to change when that wiring
    happens.
    """

    model_config = ConfigDict(extra="forbid")

    objective: str = Field(..., description="Natural-language task objective, e.g. target column + task type.")
    dataset_context: dict = Field(
        default_factory=dict,
        description="Already-sanitized dataset context (profile-like metadata). Caller's responsibility to sanitize; this module does not.",
    )
    allowed_operations: list[str] = Field(
        default_factory=list,
        description="The tool_names the LLM is permitted to propose. Advisory at the provider/prompt level only — the authoritative allowlist enforcement happens in deterministic validation (a later step), never here.",
    )
    tool_schemas: dict = Field(
        default_factory=dict,
        description=(
            "Optional per-tool argument contract (see "
            "app.agent.plan_validation.TOOL_ARGUMENT_SCHEMAS for the real "
            "producer) — argument names/types/required-ness/examples for "
            "each entry in allowed_operations. Purely advisory prompt "
            "content, same as allowed_operations; the authoritative check "
            "is still validate_proposed_plan(), never this field. Empty "
            "by default so every existing caller that only sets "
            "allowed_operations is unaffected — build_planning_prompt()/ "
            "build_replan_prompt() fall back to rendering the bare "
            "allowed_operations list when this is empty."
        ),
    )
    failure_context: Optional[dict] = Field(
        default=None,
        description="Structured FailureInfo.model_dump()-shaped data from the previous attempt, present only on REPLAN.",
    )
    previous_plan_summary: Optional[dict] = Field(
        default=None,
        description="A structured summary (e.g. a PlanDiff-shaped dict) of the previous attempt's executable plan, present only on REPLAN.",
    )


# --- Response (what the provider returns) ----------------------------------


class ProposedPlanStep(BaseModel):
    """
    The provider-layer proposal shape — deliberately mirrors
    app.agent.state.PlanStep's fields but is a SEPARATE model, not the
    same class. Keeping it separate means a change to AgentState.PlanStep
    (a graph-internal type) can never silently change what this
    standalone, not-yet-wired provider layer accepts, and vice versa —
    the later wiring step is what explicitly maps one to the other,
    and that mapping is where any semantic differences would need to
    be resolved deliberately, not accidentally via shared inheritance.
    """

    model_config = ConfigDict(extra="forbid")

    action: str
    tool_name: str
    arguments: dict = Field(default_factory=dict)
    reasoning: str


class ProposedPlan(BaseModel):
    """A provider's successful proposal — validated JSON, nothing more."""

    model_config = ConfigDict(extra="forbid")

    steps: list[ProposedPlanStep] = Field(default_factory=list)


ProviderErrorCode = Literal[
    "malformed_response",
    "invalid_plan_schema",
    "provider_unavailable",
    "timeout",
    "http_error",
]


class ProviderError(BaseModel):
    """
    Structured failure from a provider call. Never an uncontrolled
    exception escaping to the caller — every failure mode a provider
    can hit is represented here, so the (future) caller can build a
    proper FailureInfo from it without needing to catch arbitrary
    exception types.
    """

    model_config = ConfigDict(extra="forbid")

    code: ProviderErrorCode
    message: str
    raw_response_excerpt: Optional[str] = Field(
        default=None,
        description="A short excerpt of whatever raw content was received (if any), for debugging — never the full content of a pathological response.",
    )


class LLMProviderResult(BaseModel):
    """
    The envelope every LLMProvider.generate_plan() call returns.
    Exactly one of plan/error is set — mirrors the project's existing
    ToolResult[T] envelope convention (success/data/error), kept as a
    distinct type here rather than reusing ToolResult directly since
    this is provider-layer, not tool-layer, and the two should not be
    conflated even though the shape rhymes.
    """

    model_config = ConfigDict(extra="forbid")

    success: bool
    plan: Optional[ProposedPlan] = None
    error: Optional[ProviderError] = None


# --- Provider protocol -----------------------------------------------------


class LLMProvider(Protocol):
    """
    The abstraction plan_node_v2 will eventually depend on (wired in a
    later step) — never a concrete provider directly. Any conforming
    implementation (FakeLLMProvider, OllamaProvider, or a future
    provider for a different backend) can be substituted without the
    caller changing.
    """

    def generate_plan(self, context: LLMPlanningContext) -> LLMProviderResult:
        """
        Propose a plan for the given context. MUST NOT raise for
        ordinary failure modes (malformed output, provider
        unavailable, timeout) — those are represented in the returned
        LLMProviderResult.error. Raising is reserved for genuine
        programming errors, not expected failure modes.
        """
        ...


# --- FakeLLMProvider (deterministic, for tests) -----------------------------


class FakeLLMProvider:
    """
    Deterministic test double. Configured with a fixed scenario at
    construction time, so a single test can assert exactly what
    happens for that scenario without any real network call or any
    non-determinism. Also records every LLMPlanningContext it
    receives, in order — this is what lets a later test prove the
    sanitization boundary (what the provider actually received) once
    that wiring exists; already built now so that capability doesn't
    need to be retrofitted later.
    """

    def __init__(
        self,
        scenario: Literal[
            "valid_plan", "malformed_json", "invalid_plan", "provider_failure"
        ] = "valid_plan",
        fixed_plan: Optional[ProposedPlan] = None,
    ) -> None:
        self.scenario = scenario
        self.fixed_plan = fixed_plan
        self.received_contexts: list[LLMPlanningContext] = []

    def generate_plan(self, context: LLMPlanningContext) -> LLMProviderResult:
        self.received_contexts.append(context)

        if self.scenario == "valid_plan":
            plan = self.fixed_plan or ProposedPlan(
                steps=[
                    ProposedPlanStep(
                        action="Drop identifier-like column 'customerID'",
                        tool_name="drop_column",
                        arguments={"column": "customerID", "reason": "unique_percentage >= 99%"},
                        reasoning="Fake provider deterministic default plan.",
                    )
                ]
            )
            return LLMProviderResult(success=True, plan=plan)

        if self.scenario == "malformed_json":
            return LLMProviderResult(
                success=False,
                error=ProviderError(
                    code="malformed_response",
                    message="Fake provider: response was not valid JSON.",
                    raw_response_excerpt="{not valid json...",
                ),
            )

        if self.scenario == "invalid_plan":
            return LLMProviderResult(
                success=False,
                error=ProviderError(
                    code="invalid_plan_schema",
                    message="Fake provider: response was valid JSON but did not match the Plan schema.",
                    raw_response_excerpt='{"steps": [{"tool_name": 123}]}',
                ),
            )

        # provider_failure
        return LLMProviderResult(
            success=False,
            error=ProviderError(
                code="provider_unavailable",
                message="Fake provider: simulated provider unavailability.",
            ),
        )
