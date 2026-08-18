"""
Storage layer.

DatasetStore, SplitStore, ModelStore (as InMemoryModelStore), and
RunStore (as InMemoryRunStore, see run_store.py) are all fully
implemented (in-memory, no persistence across process restarts —
matches every other store here).

Note: `ModelStore` (imported below from run_and_model_store) is the
ABSTRACT interface; `InMemoryModelStore` (from model_store) is the
real, tested implementation used everywhere in the codebase. This
mirrors the DatasetStore/InMemoryDatasetStore and
SplitStore/InMemorySplitStore pattern exactly.
"""

from app.storage.dataset_store import DatasetStore, InMemoryDatasetStore
from app.storage.split_store import InMemorySplitStore, SplitStore
from app.storage.model_store import InMemoryModelStore, ModelArtifact, ModelArtifactMetadata
from app.storage.exceptions import (
    DatasetNotFoundError,
    ExplorationNotFoundError,
    ModelNotFoundError,
    RunNotFoundError,
    SplitNotFoundError,
    StorageError,
)
from app.storage.exploration_store import ExplorationStore, InMemoryExplorationStore
from app.storage.run_and_model_store import ModelStore, RunStore
from app.storage.run_store import InMemoryRunStore, RunRecord

__all__ = [
    "DatasetStore",
    "InMemoryDatasetStore",
    "SplitStore",
    "InMemorySplitStore",
    "ModelStore",
    "InMemoryModelStore",
    "ModelArtifact",
    "ModelArtifactMetadata",
    "RunStore",
    "InMemoryRunStore",
    "RunRecord",
    "ExplorationStore",
    "InMemoryExplorationStore",
    "StorageError",
    "DatasetNotFoundError",
    "RunNotFoundError",
    "ModelNotFoundError",
    "SplitNotFoundError",
    "ExplorationNotFoundError",
]
