"""
Phase 4 — deterministic model/data cards, fingerprints, explainability,
and optional subgroup analysis.

Everything here is derived from recorded PIPER evidence and fitted
objects already stored for the run. Nothing calls an LLM, nothing
rebuilds results from the original proposed plan, and missing evidence
is reported as unavailable rather than invented.
"""

from app.governance.assemble import assemble_governance_bundle
from app.governance.documents import GOVERNANCE_DOCUMENT_NAMES, render_governance_document

__all__ = [
    "assemble_governance_bundle",
    "GOVERNANCE_DOCUMENT_NAMES",
    "render_governance_document",
]
