import pandas as pd
import numpy as np

from app.storage import InMemoryDatasetStore, InMemorySplitStore, InMemoryModelStore
from app.agent.tools.guardrails import check_target_imbalance, validate_pipeline
from app.agent import AgentState, build_graph

# Exact fixture from tests/test_graph.py
np.random.seed(0)
n = 2000

df = pd.DataFrame({
    "feature_a": np.random.rand(n),
    "feature_b": np.random.choice(["X", "Y", "Z"], n),
    "target": ["No"] * 1900 + ["Yes"] * 100,
})

dataset_store = InMemoryDatasetStore()
dataset_store.save("dataset_imbalanced", df)

print("=" * 60)
print("A. DATASET")
print("=" * 60)

print(df["target"].value_counts())
print("Minority percentage:",
      df["target"].value_counts(normalize=True).min() * 100)

print()
print("=" * 60)
print("B. DIRECT check_target_imbalance()")
print("=" * 60)

r1 = check_target_imbalance(
    "dataset_imbalanced",
    "target",
    dataset_store,
)

print("success:", r1.success)

if r1.data:
    print("severity:", r1.data.severity)
    print("severely_imbalanced:", r1.data.severely_imbalanced)
    print("minority_percentage:", r1.data.minority_percentage)
    print("warning_threshold:", r1.data.warning_threshold_percent)
    print("failure_threshold:", r1.data.failure_threshold_percent)

print("message:", r1.message)

print()
print("=" * 60)
print("C. DIRECT validate_pipeline()")
print("=" * 60)

r2 = validate_pipeline(
    "dataset_imbalanced",
    "target",
    dataset_store,
)

print("success:", r2.success)

if r2.data:
    print("valid:", r2.data.valid)

    print(
        "checks:",
        [
            (
                c.check,
                c.passed,
                c.severity,
                c.message,
            )
            for c in r2.data.checks
        ],
    )

    print(
        "warnings:",
        [
            (
                w.check,
                w.passed,
                w.severity,
                w.message,
            )
            for w in r2.data.warnings
        ],
    )

print()
print("=" * 60)
print("D. FULL GRAPH")
print("=" * 60)

split_store = InMemorySplitStore()
model_store = InMemoryModelStore()

graph = build_graph(
    dataset_store,
    split_store,
    model_store,
)

initial = AgentState(
    run_id="run_imb",
    dataset_id="dataset_imbalanced",
    target_column="target",
)

result = graph.invoke(
    initial,
    config={"recursion_limit": 50},
)

print("status:", result.get("status"))

validation = result.get("validation")

print("validation type:", type(validation))

if validation:
    print("validation.valid:", validation.valid)

    print(
        "validation.checks:",
        [
            (
                c.check,
                c.passed,
                c.severity,
                c.message,
            )
            for c in validation.checks
        ],
    )

    print(
        "validation.warnings:",
        [
            (
                w.check,
                w.passed,
                w.severity,
                w.message,
            )
            for w in validation.warnings
        ],
    )

print()
print("=" * 60)
print("E. FINAL STATE KEYS")
print("=" * 60)

print(list(result.keys()))