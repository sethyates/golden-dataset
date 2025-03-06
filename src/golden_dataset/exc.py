from pathlib import Path
from typing import Any


class GoldenError(Exception):
    """Base exception for golden dataset errors."""

    pass


class DatasetNotFoundError(GoldenError):
    """Exception raised when a dataset cannot be found."""

    def __init__(self, dataset_name: str, path: Path):
        self.dataset_name = dataset_name
        self.path = path
        super().__init__(f"Dataset '{dataset_name}' not found at {path}")


class DatabaseError(GoldenError):
    """Base exception for database-related errors."""

    pass


class ModelNotFoundError(DatabaseError):
    """Exception raised when a model class cannot be found."""

    pass


class EntityImportError(DatabaseError):
    """Exception raised when an entity cannot be imported."""

    def __init__(self, table_name: str, entity_id: Any, original_error: Exception):
        self.table_name = table_name
        self.entity_id = entity_id
        self.original_error = original_error
        super().__init__(f"Error creating {table_name} with ID {entity_id}: {str(original_error)}")
