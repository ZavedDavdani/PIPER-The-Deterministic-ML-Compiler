"""
PIPER Learn (Batch 6A: Learn-Explain).

A read-only explanation layer over existing PIPER execution state.
Nothing in this package can construct or return an AgentState update,
call a graph node, or otherwise influence a run — every function here
takes already-computed state/records as input and returns a schema
from app/schemas/learning.py. See app/learning/explain.py's module
docstring for the full design rationale.
"""
