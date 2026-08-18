"""
Storage-layer exceptions.

Kept separate from ToolError (schemas/common.py) deliberately: these
are raised by the storage layer and caught by the tool layer, which
then translates them into a proper ToolResult(success=False, error=...).
The storage layer itself has no knowledge of ToolResult/ToolError.
"""

from __future__ import annotations


class StorageError(Exception):
    """Base class for all storage-layer errors."""


class DatasetNotFoundError(StorageError):
    def __init__(self, dataset_id: str):
        self.dataset_id = dataset_id
        super().__init__(f"Dataset '{dataset_id}' does not exist.")


class RunNotFoundError(StorageError):
    def __init__(self, run_id: str):
        self.run_id = run_id
        super().__init__(f"Run '{run_id}' does not exist.")


class ModelNotFoundError(StorageError):
    def __init__(self, model_id: str):
        self.model_id = model_id
        super().__init__(f"Model '{model_id}' does not exist.")


class SplitNotFoundError(StorageError):
    def __init__(self, split_id: str):
        self.split_id = split_id
        super().__init__(f"Split '{split_id}' does not exist.")


class ExplorationNotFoundError(StorageError):
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        super().__init__(f"Exploration '{experiment_id}' does not exist.")
