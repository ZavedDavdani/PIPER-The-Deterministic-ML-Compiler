"""
Static PIPER Learn educational content endpoints (Phase 6: Student Mode).
Global, static, and generic knowledge base (never per-run/per-dataset).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.learning.comprehension import COMPREHENSION_CHECKS
from app.learning.formulas import FORMULA_LIBRARY
from app.learning.registry import ACTION_REGISTRY, CONCEPTS, METRIC_GUIDANCE, MODEL_FAMILIES
from app.schemas.learning import ComprehensionCheck, ConceptDefinition, FormulaEntry

router = APIRouter(prefix="/learn", tags=["learning"])


@router.get("/formulas", response_model=list[FormulaEntry])
def get_formula_library() -> list[FormulaEntry]:
    return FORMULA_LIBRARY


@router.get("/comprehension-checks", response_model=list[ComprehensionCheck])
def get_comprehension_checks() -> list[ComprehensionCheck]:
    return COMPREHENSION_CHECKS


@router.get("/concepts", response_model=list[ConceptDefinition])
def get_concepts() -> list[ConceptDefinition]:
    return CONCEPTS


@router.get("/actions")
def get_action_registry() -> dict[str, Any]:
    return ACTION_REGISTRY


@router.get("/models")
def get_model_families() -> dict[str, Any]:
    return MODEL_FAMILIES


@router.get("/metrics")
def get_metric_guidance() -> dict[str, Any]:
    return METRIC_GUIDANCE
