"""
Node implementations.

m2_nodes.py: profile_node and clean_node are real and still used
directly by the real graph. plan_node/train_node/evaluate_node/
validate_node in m2_nodes.py are the ORIGINAL M2 stub versions —
retained for historical/reference purposes and for M2's own dedicated
tests (test_graph.py still exercises them against a graph built the
old way in some tests), but the REAL graph (graph.py) no longer uses
train_node/evaluate_node/validate_node/plan_node from this module.

real_nodes.py: the real pipeline nodes actually wired into graph.py —
plan_node_v2 (extended to also plan feature engineering),
feature_engineer_node, split_node, train_node_v2, evaluate_node_v2,
validate_node_v2. Every one of these calls a real tool; no ML logic
lives in the node functions themselves.
"""

from app.agent.nodes.m2_nodes import clean_node, profile_node
from app.agent.nodes.real_nodes import (
    evaluate_node_v2,
    feature_engineer_node,
    plan_node_v2,
    split_node,
    train_node_v2,
    validate_node_v2,
)

__all__ = [
    "profile_node",
    "clean_node",
    "plan_node_v2",
    "feature_engineer_node",
    "split_node",
    "train_node_v2",
    "evaluate_node_v2",
    "validate_node_v2",
]
