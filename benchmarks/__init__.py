"""Self-contained benchmark generators + registry."""
from .registry import get_dataset, list_datasets, Dataset

__all__ = ["get_dataset", "list_datasets", "Dataset"]
