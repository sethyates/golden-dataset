"""
Golden dataset class for managing and exporting datasets.
"""

import datetime
import functools
import json
import logging
import uuid
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, Self, TypedDict, TypeVar, cast, overload

from pydantic import BaseModel, Field, PrivateAttr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    PyprojectTomlConfigSettingsSource,
    SettingsConfigDict,
)
from sqlalchemy import exc
from sqlalchemy.orm import DeclarativeMeta, Session

from .core import bulk_delete, bulk_import, get_function, is_same_class
from .exc import DatasetNotFoundError, GoldenError

# Set up logging
logger = logging.getLogger(__name__)


DEFAULT_SRC_DIR = "src"
DEFAULT_DATA_DIR = "tests/data/golden"
DEFAULT_GENERATOR_MODULE = "golden"


T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class GoldenSettings(BaseSettings):
    """Settings for golden datasets."""

    src_dir: str = Field(default=DEFAULT_SRC_DIR, alias="src")
    datasets_dir: str = Field(default=DEFAULT_DATA_DIR, alias="datasets")
    generators: str = Field(default=DEFAULT_GENERATOR_MODULE, alias="generators")
    base_class_name: str = Field(default="Base", alias="base-class")
    engine_name: str = Field(default="engine", alias="engine")
    session_factory_name: str = Field(default="Session", alias="session-factory")

    model_config = SettingsConfigDict(
        env_prefix="GOLDEN_",
        pyproject_toml_table_header=("tool", "golden"),
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PyprojectTomlConfigSettingsSource, PydanticBaseSettingsSource, PydanticBaseSettingsSource]:
        return (PyprojectTomlConfigSettingsSource(settings_cls), env_settings, dotenv_settings)


class TableInfo(TypedDict):
    """Information about a table in a dataset."""

    name: str
    count: int


class GoldenDataset(BaseModel):
    """
    Golden dataset class for managing and exporting datasets.
    """

    name: str
    revision: str = "1"
    title: str | None = Field(default=None)
    description: str = Field(default="")
    dependencies: list[str] = Field(default_factory=list)
    tables: dict[str, int] = Field(default_factory=lambda: defaultdict(int))
    exported_at: datetime.datetime | None = Field(default=None)

    _objects: dict[str, dict[Any, Any]] = PrivateAttr(default_factory=lambda: defaultdict(dict))

    def remove_from_session(self, base: DeclarativeMeta, session: Session) -> dict[str, int]:
        """
        Remove all objects in the dataset from a SQLAlchemy session.

        Args:
            base: SQLAlchemy declarative base class containing model definitions
            session: An open SQLAlchemy session

        Returns:
            Dictionary mapping table names to number of successfully deleted entities

        Raises:
            ModelNotFoundError: If a model class cannot be found for a table
        """
        return bulk_delete(self._objects, base, session)

    def add_to_session(self, base: DeclarativeMeta, session: Session) -> dict[str, int]:
        """
        Add all objects in the dataset to a SQLAlchemy session.

        Args:
            base: SQLAlchemy declarative base class containing model definitions
            session: An open SQLAlchemy session

        Returns:
            Dictionary mapping table names to number of successfully imported entities

        Raises:
            ModelNotFoundError: If a model class cannot be found for a table
            EntityImportError: If an entity cannot be imported
        """
        return bulk_import(self._objects, base, session)

    def clear(self) -> None:
        """Clear all data from the dataset."""
        self.tables = dict()
        self._objects = defaultdict(dict)

    def add[ModelType](self, obj: ModelType) -> ModelType:
        """
        Add an object to the dataset.

        Args:
            obj: The object to add to the dataset.

        Returns:
            The added object (for method chaining)
        """
        table_name = self._get_table_name(obj)

        # Generate an ID if not present
        obj_id = self._get_or_create_id(obj)

        # Store the object in the appropriate table
        self._objects[table_name][obj_id] = obj
        self.tables[table_name] = self.tables[table_name] + 1

        return obj

    def import_data(self, data: dict[str, dict[Any, Any]]) -> None:
        """
        Import data from a dictionary into the dataset.

        Args:
            data: Dictionary mapping table names to dictionaries of objects
        """
        self._objects = data
        self._recalculate_tables()

    def refresh(self, obj: Any) -> None:
        """
        Simulate refreshing an object from the database.

        In SQLAlchemy, this would refresh the object from the database.
        Here, it's a no-op since we're not actually querying a database.

        Args:
            obj: The object to refresh
        """
        pass

    def query[ModelType](self, model_cls: type[ModelType]) -> list[ModelType]:
        """
        Query objects of a specific model type.

        Args:
            model_cls: The model class to query.

        Returns:
            List of objects of the specified model type.
        """
        table_name = self._get_model_table_name(model_cls)
        return list(self._objects.get(table_name, {}).values())

    def get[ModelType](self, model_cls: type[ModelType], obj_id: Any) -> ModelType | None:
        """
        Get an object by its ID.

        Args:
            model_cls: The model class to query.
            obj_id: The ID of the object to get.

        Returns:
            The object with the specified ID, or None if not found.
        """
        table_name = self._get_model_table_name(model_cls)
        return self._objects.get(table_name, {}).get(obj_id)

    def _recalculate_tables(self) -> None:
        """Update the tables dictionary based on the current objects."""
        self.tables = {table_name: len(objects) for table_name, objects in self._objects.items()}

    def get_tables(self) -> dict[str, int]:
        """
        Get information about the tables in the dataset.

        Returns:
            A dictionary mapping table names to the number of objects in each table.
        """
        self._recalculate_tables()
        return self.tables.copy()

    def get_table(self, name: str) -> dict[str, list[dict[str, Any]]]:
        """
        Get all data in a table as a dictionary suitable for serialization.

        Args:
            name: The name of the table.

        Returns:
            A dictionary with a "data" key containing a list of serialized objects.
        """
        objects = self._objects[name]
        serialized_objects = []

        for obj in list(objects.values()):
            serialized_objects.append(self._serialize_object(obj))

        return {"data": serialized_objects}

    def _get_or_create_id(self, obj: Any) -> Any:
        """
        Get or create an ID for an object.

        Args:
            obj: The object to get or create an ID for.

        Returns:
            The ID of the object.
        """
        # Try common primary key attribute names
        for attr in ["id", "ID", "Id", "primary_key"]:
            if hasattr(obj, attr) and getattr(obj, attr) is not None:
                return getattr(obj, attr)

        # For SQLModel/SQLAlchemy models
        if hasattr(obj, "__table__") and hasattr(obj.__table__, "primary_key"):
            primary_keys = obj.__table__.primary_key.columns.keys()
            if len(primary_keys) == 1:
                pk_name = primary_keys[0]
                if hasattr(obj, pk_name) and getattr(obj, pk_name) is not None:
                    return getattr(obj, pk_name)

        # Generate a UUID if no ID is found
        new_id = uuid.uuid4()
        if hasattr(obj, "id"):
            obj.id = new_id

        return new_id

    def _serialize_object(self, obj: Any) -> dict[str, Any]:
        """
        Serialize an object to a dictionary.

        Args:
            obj: The object to serialize.

        Returns:
            A dictionary representation of the object.
        """
        # For SQLAlchemy objects
        result: dict[str, Any] = {}

        # For Pydantic/SQLModel objects
        match obj:
            case o if hasattr(o, "model_dump"):
                # Modern Pydantic v2
                result = o.model_dump()
            case o if hasattr(o, "dict"):
                # Legacy Pydantic v1
                result = o.dict()
            case o if hasattr(o, "__table__"):
                # SQLAlchemy models
                for column in o.__table__.columns:
                    result[column.name] = getattr(o, column.name)
            case _:
                # For dataclasses and other objects
                for attr_name in dir(obj):
                    # Skip private attributes, methods, and callables
                    if not attr_name.startswith("_") and not callable(getattr(obj, attr_name)):
                        value = getattr(obj, attr_name)

                        # Handle special case for UUID, datetime, etc.
                        if hasattr(value, "__str__"):
                            result[attr_name] = str(value)
                        else:
                            result[attr_name] = value

        return result

    def _get_table_name(self, obj: Any) -> str:
        """
        Get the table name for an object.

        Args:
            obj: The object to get the table name for.

        Returns:
            The table name for the object.
        """
        # Try SQLModel/SQLAlchemy pattern
        if hasattr(obj, "__table__") and hasattr(obj.__table__, "name"):
            name: str = obj.__table__.name
        else:
            name = obj.__class__.__name__.lower()
        return name

    def _get_model_table_name(self, model_cls: type) -> str:
        """
        Get the table name for a model class.

        Args:
            model_cls: The model class to get the table name for.

        Returns:
            The table name for the model class.
        """
        # Try SQLModel/SQLAlchemy pattern
        if hasattr(model_cls, "__tablename__"):
            name: str = model_cls.__tablename__

        # Try SQLModel/SQLAlchemy metadata pattern
        elif hasattr(model_cls, "__table__") and hasattr(model_cls.__table__, "name"):
            name = model_cls.__table__.name

        # Fall back to class name
        else:
            name = model_cls.__name__.lower()

        return name


class GoldenSession:
    """
    A session-like interface for golden datasets that mimics SQLAlchemy/SQLModel sessions.

    This class provides a minimal interface compatible with SQLAlchemy's session
    with methods like add(), refresh(), etc. but stores objects in memory.
    """

    dataset: GoldenDataset | None

    def __init__(self, dataset: GoldenDataset | None = None):
        """Initialize a new golden session."""
        self.dataset = dataset

    def add[ModelType](self, obj: ModelType) -> ModelType:
        """
        Add an object to the session.

        Args:
            obj: The object to add to the session.

        Returns:
            The added object (for method chaining)
        """
        if self.dataset is None:
            raise exc.InvalidRequestError("Object cannot be added to a closed session")

        return self.dataset.add(obj)

    def add_all(self, objects: list[Any]) -> None:
        """
        Add multiple objects to the session.

        Args:
            objects: List of objects to add to the session.
        """
        if self.dataset is None:
            raise exc.InvalidRequestError("Objects cannot be added to a closed session")

        for obj in objects:
            self.dataset.add(obj)

    def refresh(self, obj: Any) -> None:
        """
        Simulate refreshing an object from the database.

        In SQLAlchemy, this would refresh the object from the database.
        Here, it's a no-op since we're not actually querying a database.

        Args:
            obj: The object to refresh

        Raises:
            InvalidRequestError: If the session is closed
        """
        if self.dataset is None:
            raise exc.InvalidRequestError("Object cannot be refreshed on a closed session")

        self.dataset.refresh(obj)

    def query[ModelType](self, model_cls: type[ModelType]) -> list[ModelType]:
        """
        Query objects of a specific model type.

        Args:
            model_cls: The model class to query.

        Returns:
            List of objects of the specified model type.

        Raises:
            InvalidRequestError: If the session is closed
        """
        if self.dataset is None:
            raise exc.InvalidRequestError("Object cannot be queried on a closed session")

        return self.dataset.query(model_cls)

    def get[ModelType](self, model_cls: type[ModelType], obj_id: Any) -> ModelType | None:
        """
        Get an object by its ID.

        Args:
            model_cls: The model class to query.
            obj_id: The ID of the object to get.

        Returns:
            The object with the specified ID, or None if not found.

        Raises:
            InvalidRequestError: If the session is closed
        """
        if self.dataset is None:
            raise exc.InvalidRequestError("Object cannot be queried on a closed session")

        return self.dataset.get(model_cls, obj_id)

    def commit(self) -> None:
        """
        Simulate committing changes to the database.

        In SQLAlchemy, this would commit the transaction.
        Here, it's a no-op since we're not actually using a database.

        Raises:
            InvalidRequestError: If the session is closed
        """
        if self.dataset is None:
            raise exc.InvalidRequestError("Cannot commit a closed session")

    def rollback(self) -> None:
        """
        Simulate rolling back changes to the database.

        In SQLAlchemy, this would roll back the transaction.
        Here, we clear the dataset to simulate a rollback.

        Raises:
            InvalidRequestError: If the session is closed
        """
        if self.dataset is None:
            raise exc.InvalidRequestError("Cannot rollback a closed session")

        self.dataset.clear()

    def close(self) -> None:
        """
        Simulate closing the session.

        In SQLAlchemy, this would close the session.
        Here, we simply set the dataset to None.

        Raises:
            InvalidRequestError: If the session is already closed
        """
        if self.dataset is None:
            raise exc.InvalidRequestError("Cannot close a closed session")

        self.dataset = None

    def __enter__(self) -> Self:
        """Support for context manager interface."""
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        """Support for context manager interface."""
        if exc_type is not None:
            self.rollback()
        else:
            try:
                self.commit()
            except:
                self.rollback()
                raise
        self.close()


class GoldenManager:
    """Manager for golden datasets."""

    def __init__(self, settings: GoldenSettings):
        """
        Initialize the golden manager with settings.

        Args:
            settings: Settings for the golden manager
        """
        default_settings = GoldenSettings()
        self.datasets_dir = Path(settings.datasets_dir or default_settings.datasets_dir)
        self.src_dir = Path(settings.src_dir or default_settings.src_dir or ".")
        self.generators = settings.generators or default_settings.generators

    def session(self, dataset: GoldenDataset) -> GoldenSession:
        """
        Create a new session for a dataset.

        Args:
            dataset: Dataset to create a session for

        Returns:
            A new golden session
        """
        return GoldenSession(dataset)

    def dataset(
        self, name: str, title: str | None = None, description: str | None = None, dependencies: list[str] | None = None
    ) -> GoldenDataset:
        """
        Create a new dataset.

        Args:
            name: Name of the dataset
            title: Title of the dataset (defaults to capitalized name)
            description: Description of the dataset
            dependencies: List of dataset dependencies

        Returns:
            A new golden dataset
        """
        return GoldenDataset(
            name=name, title=title or name.title() or "", description=description or "", dependencies=dependencies or []
        )

    def list_datasets(self) -> list[GoldenDataset]:
        """
        List all available golden datasets.

        Returns:
            List of golden datasets

        Raises:
            FileNotFoundError: If the datasets directory does not exist
            NotADirectoryError: If the datasets directory is not a directory
        """
        if not self.datasets_dir.exists():
            raise FileNotFoundError(f"Directory {self.datasets_dir} does not exist")

        if not self.datasets_dir.is_dir():
            raise NotADirectoryError(f"Expected {self.datasets_dir} to be a dir")

        # Find all datasets
        datasets = []
        for item in self.datasets_dir.iterdir():
            if item.is_dir():
                try:
                    datasets.append(self.open_dataset(item.name))
                except Exception as e:
                    logger.warning(f"Could not open dataset {item.name}: {e}")

        return datasets

    def open_dataset(self, dataset_name: str) -> GoldenDataset:
        """
        Open a dataset from the datasets directory.

        Args:
            dataset_name: Name of Dataset to open

        Returns:
            The opened Golden Dataset

        Raises:
            DatasetNotFoundError: If the dataset does not exist
            FileNotFoundError: If the metadata file does not exist
            ValueError: If the metadata file is invalid
        """
        item = self.datasets_dir / dataset_name

        if not item.exists() or not item.is_dir():
            raise DatasetNotFoundError(dataset_name, item)

        metadata_file = item / "_metadata.json"

        if not metadata_file.exists():
            raise FileNotFoundError(f"Expected metadata file {metadata_file} to exist")

        try:
            with open(metadata_file) as f:
                return GoldenDataset.model_validate_json(f.read())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid metadata file {metadata_file}: {e}") from e
        except Exception as e:
            raise ValueError(f"Could not open dataset {dataset_name}: {e}") from e

    def load_dataset(self, dataset_name: str) -> GoldenDataset:
        """
        Load a dataset, including all its data.

        Args:
            dataset_name: Name of the dataset to load

        Returns:
            The loaded dataset with all its data

        Raises:
            DatasetNotFoundError: If the dataset does not exist
        """
        dataset = self.open_dataset(dataset_name)

        data: dict[str, dict[Any, Any]] = defaultdict(dict)
        for table_name in dataset.tables:
            file_path = self.datasets_dir / dataset_name / f"{table_name}.json"
            if not file_path.exists():
                logger.warning(f"Table file {file_path} does not exist")
                continue

            try:
                with open(file_path) as f:
                    table_data = json.load(f)

                    # Use adapter to convert dictionaries to model instances
                    for item in table_data.get("data", []):
                        data[table_name][item.get("id")] = item
            except Exception as e:
                logger.error(f"Error loading table {table_name}: {e}")

        dataset.import_data(data)
        return dataset

    def dump_dataset(self, dataset: GoldenDataset) -> None:
        """
        Dump the dataset to JSON files.

        Args:
            dataset: Dataset to export

        Raises:
            IOError: If the dataset directory cannot be created
        """
        # Create directory for this dataset
        dataset_dir = self.datasets_dir / dataset.name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Write each table to a separate file
        for table_name in dataset.get_tables():
            table_data = dataset.get_table(table_name)
            file_path = dataset_dir / f"{table_name}.json"
            try:
                with open(file_path, "w") as f:
                    json.dump(table_data, f, indent=2, default=str)
            except Exception as e:
                logger.error(f"Error writing table {table_name}: {e}")
                raise OSError(f"Could not write table {table_name}") from e

        # Set export timestamp
        dataset.exported_at = datetime.datetime.now(datetime.UTC)

        # Create metadata file
        metadata_path = dataset_dir / "_metadata.json"
        try:
            with open(metadata_path, "w") as f:
                json.dump(dataset.model_dump(mode="json"), f, indent=2)
        except Exception as e:
            logger.error(f"Error writing metadata: {e}")
            raise OSError("Could not write metadata") from e

    def generate_dataset(self, fn: str) -> GoldenDataset:
        """
        Generate a dataset from a generator function.

        Args:
            fn: Path to the generator function in the format "module.path:function_name"

        Returns:
            The generated dataset

        Raises:
            GoldenError: If the generator function does not have the correct signature
        """
        func, name, args = get_function(fn, str(self.generators), str(self.src_dir))

        first_arg = args.pop(0)
        if first_arg["name"] != "session" or not (
            is_same_class(first_arg["type"], GoldenSession) or is_same_class(first_arg["type"], Any)
        ):
            raise GoldenError(f"Generator {fn} must have session as first parameter")

        title = ""
        description = ""
        dependencies = [arg["name"] for arg in args]
        if getattr(func, "__golden__", False):
            name = func.__name__ or name
            title = getattr(func, "__title__", "")
            description = getattr(func, "__description__", "")
            dependencies.extend(getattr(func, "__dependencies__", []) or [])

        dataset = self.dataset(name=name, title=title, description=description, dependencies=list(set(dependencies)))

        session = self.session(dataset)
        try:
            func_args = [arg.get("default", None) for arg in args]
            func(session, *func_args)
            session.commit()
        except Exception as e:
            session.rollback()
            raise GoldenError(f"Error generating dataset: {e}") from e

        return dataset


# Define a protocol for the wrapped function that will have our custom attributes
class GoldenWrapped(Protocol):
    __golden__: bool
    __name__: str
    __title__: str
    __description__: str
    __dependencies__: list[str]

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


class Golden(GoldenManager):
    """
    Decorator for golden dataset generators.

    Can be used as:

    @golden
    def func(session): ...

    or

    @golden(dependencies=['dataset1'])
    def func(session): ...
    """

    def __init__(self, settings: GoldenSettings | None = None):
        """Initialize the golden decorator with settings."""
        super().__init__(settings=settings or GoldenSettings())

    # First overload: when used as @golden
    @overload
    def __call__(
        self,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        dependencies: list[str] | None = None,
    ) -> GoldenWrapped: ...

    # Second overload: when used as @golden(title="", dependencies=[...])
    @overload
    def __call__(
        self,
        func: None = None,
        *,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        dependencies: list[str] | None = None,
    ) -> Callable[[Callable[..., Any]], GoldenWrapped]: ...

    def __call__(
        self,
        func: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        dependencies: list[str] | None = None,
    ) -> GoldenWrapped | Callable[[Callable[..., Any]], GoldenWrapped]:
        """
        Supports both:
        @golden
        def func(): ...

        and

        @golden(dependencies=['dataset1'])
        def func(): ...
        """
        # Called as @golden
        if callable(func):
            return self._decorate(
                func, name=name or "", title=title or "", description=description or "", dependencies=[]
            )

        # Called as @golden(dependencies=...)
        else:
            deps = dependencies or []

            def decorator(fn: Callable[..., Any]) -> GoldenWrapped:
                return self._decorate(
                    fn, name=name or "", title=title or "", description=description or "", dependencies=deps
                )

            return decorator

    def _decorate(
        self,
        func: Callable[..., Any],
        name: str,
        title: str,
        description: str,
        dependencies: list[str],
    ) -> GoldenWrapped:
        """
        Decorate a function to mark it as a golden dataset generator.

        Args:
            func: The function to decorate
            name: Name for the dataset
            title: Title for the dataset
            description: Description for the dataset
            dependencies: List of dataset dependencies

        Returns:
            The decorated function
        """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)
            return result

        # Add attributes to the wrapper
        wrapper.__golden__ = True  # type: ignore
        wrapper.__name__ = name or func.__name__
        wrapper.__title__ = title  # type: ignore
        wrapper.__description__ = description  # type: ignore
        wrapper.__dependencies__ = dependencies  # type: ignore

        return cast(GoldenWrapped, wrapper)  # Cast to satisfy type checker


# Create a default instance
golden = Golden()
