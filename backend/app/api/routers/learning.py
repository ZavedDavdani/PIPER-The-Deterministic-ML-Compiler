"""
Static PIPER Learn (Batch 6A: Learn-Explain) content endpoints — the
formula library and comprehension checks. Both are global, static, and
generic (never per-run/per-dataset), so unlike the per-run
GET /runs/{run_id}/learn/explanation endpoint (app/api/routers/runs.py),
these need no run_id and no RunStore dependency at all.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.learning.comprehension import COMPREHENSION_CHECKS
from app.learning.formulas import FORMULA_LIBRARY
from app.schemas.learning import ComprehensionCheck, FormulaEntry

router = APIRouter(prefix="/learn", tags=["learning"])


@router.get("/formulas", response_model=list[FormulaEntry])
def get_formula_library() -> list[FormulaEntry]:
    return FORMULA_LIBRARY


@router.get("/comprehension-checks", response_model=list[ComprehensionCheck])
def get_comprehension_checks() -> list[ComprehensionCheck]:
    return COMPREHENSION_CHECKS
