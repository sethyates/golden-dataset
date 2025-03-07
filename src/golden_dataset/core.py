"""
Core functionality for golden datasets.
"""

import contextlib
import datetime
import importlib
import inspect
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict, TypeVar, get_type_hints

from sqlalchemy import Engine
from sqlalchemy import inspect as sqlalchemy_inspect
from sqlalchemy.ext.declarative import DeclarativeMeta
from sqlalchemy.orm import Query, Session, sessionmaker

from .exc import EntityImportError, GoldenError, ModelNotFoundError

# Set up logging
logger = logging.getLogger(__name__)


T = TypeVar("T")  # Generic type for return values
ModelType = TypeVar("ModelType")  # Type for SQLAlchemy models


class ParamInfo(TypedDict):
    """Type definition for parameter information dictionaries."""

    name: str
    kind: str
    has_default: bool
    default: Any
    type: Any  # Using Any because Type is too restrictive


def is_same_class(obj_class: Any, proto_class: Any) -> bool:
    """
    Check if two classes are the same by comparing their names and modules.

    Args:
        obj_class: First class to compare
        proto_class: Second class to compare

    Returns:
        True if classes are the same, False otherwise
    """
    # Check if we're dealing with typing special forms
    if obj_class is Any or proto_class is Any:
        return obj_class is proto_class

    # Handle regular class types
    if (
        hasattr(obj_class, "__name__")
        and hasattr(proto_class, "__name__")
        and hasattr(obj_class, "__module__")
        and hasattr(proto_class, "__module__")
    ):
        return obj_class.__name__ == proto_class.__name__ and obj_class.__module__ == proto_class.__module__ and True

    # If we can't compare them in a structured way, just check equality
    return obj_class == proto_class and True


def parse_import_path(import_path: str) -> tuple[str, str]:
    """
    Parse an import path in the format "module.path:object_name" or just "object_name".

    Args:
        import_path: A string either in the format "module.path:object_name" or just "object_name"

    Returns:
        A tuple containing (module_path, object_name)
        - If no colon is present, returns ("", import_path)
    """
    if ":" not in import_path:
        return "", import_path

    module_path, object_name = import_path.split(":", 1)

    # Validate the object name is not empty
    if not object_name:
        raise ValueError(f"Empty object name in import: {import_path}")

    return module_path, object_name


def get_function(
    module_path_and_func: str,
    generator_root: str | None = None,
    source_root: str | None = None,
) -> tuple[Callable, str, list[ParamInfo]]:
    """
    Dynamically imports and returns a function along with its name and parameter information.

    Args:
        module_path_and_func: String in the format 'module.path:function_name'
        generator_root: Optional generator root directory for package imports
        source_root: Optional source root directory to add to sys.path

    Returns:
        Tuple containing:
        - The callable function
        - The function name (string)
        - List of parameters with their types and information

    Raises:
        ValueError: If the module path format is invalid
        ImportError: If the module cannot be imported
        AttributeError: If the function cannot be found in the module
        TypeError: If the function is not callable
        GoldenError: For other processing errors
    """
    # Record original sys.path to restore later
    original_path = sys.path.copy()

    try:
        # Add source_root to path if provided
        if source_root:
            source_path = Path(source_root).resolve()
            if str(source_path) not in sys.path:
                sys.path.insert(0, str(source_path))
        sys.path.insert(0, "")

        # Parse the module path and function name
        if ":" not in module_path_and_func:
            raise ValueError(f"Invalid format: {module_path_and_func}. Expected 'module.path:function_name'")

        module_path, func_name = parse_import_path(module_path_and_func)

        try:
            # Import the module
            if module_path:
                module = importlib.import_module(f".{module_path}", package=generator_root)
            elif generator_root:
                module = importlib.import_module(generator_root, package=generator_root)
            else:
                raise GoldenError("Must specify either a generator root or module path to import")

            # Get the function
            if not hasattr(module, func_name):
                raise AttributeError(f"Function '{func_name}' not found in module '{module_path}'")

            func = getattr(module, func_name)

            if not callable(func):
                raise TypeError(f"'{func_name}' is not a callable function")

            # Get parameter information
            sig = inspect.signature(func)
            params = []
            type_hints = get_type_hints(func)

            for name, param in sig.parameters.items():
                # Skip 'self' and 'cls' for methods
                if name in ("self", "cls") and param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                    continue

                param_info: ParamInfo = {
                    "name": name,
                    "kind": str(param.kind),
                    "has_default": param.default is not param.empty,
                    "default": None if param.default is param.empty else param.default,
                    "type": Any,
                }

                # Get type information if available
                if name in type_hints:
                    param_info["type"] = type_hints[name]

                params.append(param_info)

            return func, func_name, params

        except ImportError as e:
            logger.error(f"Failed to import module '{module_path}': {e}")
            raise ImportError(f"Failed to import module '{module_path}': {e}") from e
        except (AttributeError, TypeError) as e:
            logger.error(f"Error with function '{func_name}' in module '{module_path}': {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing function '{module_path_and_func}': {e}")
            raise GoldenError(f"Error processing function '{module_path_and_func}': {e}") from e

    finally:
        # Restore original path
        sys.path = original_path


def find_class_by_name(class_name: str, search_paths: list[str], package: str | None = None) -> type | None:
    """
    Find a class by name in a list of search paths.

    Args:
        class_name: Name of the class to find
        search_paths: List of module paths to search
        package: Optional package prefix for relative imports

    Returns:
        The class if found, None otherwise
    """
    for module_name in search_paths:
        logger.debug(f"searching for {class_name} in {module_name} with package {package}")
        with contextlib.suppress(ImportError, AttributeError):
            module = importlib.import_module(f".{module_name}", package=package)
            return getattr(module, class_name)  # type: ignore

    return None


def get_sqlalchemy_base(
    base_class_name: str = "Base",
    search_path: list[str] | None = None,
    package: str | None = None,
) -> DeclarativeMeta | None:
    """
    Find the SQLAlchemy Base class in the project.

    Args:
        base_class_name: Name of the Base class to find (default: "Base")
        search_path: List of module paths to search
        package: Optional package prefix for relative imports

    Returns:
        The Base class if found, None otherwise
    """
    search_path = search_path or [
        "app.models",
        "app.db.models",
        "models",
        "db.models",
        "myapp.models",
        "project.models",
        "src.models",
    ]

    models_module, base_class_name = parse_import_path(base_class_name)

    if models_module:
        search_path.insert(0, models_module)

    # First try finding the base class directly
    base = find_class_by_name(base_class_name, search_path, package)
    if base:
        return base  # type: ignore

    # Then try finding SQLModel which is common in newer projects
    base = find_class_by_name("SQLModel", search_path, package)
    if base:
        return base  # type: ignore

    logger.warning(f"Could not find SQLAlchemy Base class '{base_class_name}' in any of the search paths")
    return None


def get_sqlalchemy_engine(
    engine_name: str = "engine",
    search_path: list[str] | None = None,
    package: str | None = None,
) -> Engine | None:
    """
    Find the SQLAlchemy engine in the project.

    Args:
        engine_name: Name of the engine variable (default: "engine")
        search_path: List of module paths to search
        package: Optional package prefix for relative imports

    Returns:
        The engine if found, None otherwise
    """
    search_path = search_path or [
        "app.db",
        "app.database",
        "db",
        "database",
        "app.db.engine",
        "app.database.engine",
        "db.engine",
        "config",
        "app.config",
    ]

    engine_module, engine_name = parse_import_path(engine_name)

    if engine_module:
        search_path.insert(0, engine_module)

    engine = find_class_by_name(engine_name, search_path, package)
    if engine:
        return engine  # type: ignore

    logger.warning(f"Could not find SQLAlchemy engine '{engine_name}' in any of the search paths")
    return None


def get_sqlalchemy_session_factory(
    session_factory_name: str = "Session",
    search_path: list[str] | None = None,
    package: str | None = None,
) -> Callable[[], Session] | None:
    """
    Find the SQLAlchemy session factory in the project.

    Args:
        session_factory_name: Name of the session factory (default: "Session")
        search_path: List of module paths to search
        package: Optional package prefix for relative imports

    Returns:
        The session factory if found, None otherwise
    """
    search_path = search_path or [
        "app.db",
        "app.database",
        "db",
        "database",
        "app.db.session",
        "app.database.session",
        "db.session",
    ]

    session_factory_module, session_factory_name = parse_import_path(session_factory_name)
    if session_factory_module:
        search_path.insert(0, session_factory_module)

    # Try to find the session factory
    for module_name in search_path:
        with contextlib.suppress(ImportError, AttributeError):
            logger.debug(f"Searching for session factory in {module_name}")
            module = importlib.import_module(f".{module_name}", package=package)
            if hasattr(module, session_factory_name):
                session = getattr(module, session_factory_name)
                if callable(session):
                    logger.debug(f"Found SQLAlchemy session factory in {module_name}")
                    return session  # type: ignore

    # If we couldn't find a Session, create a basic one from the engine
    logger.info(f"Could not find SQLAlchemy session factory '{session_factory_name}'. Attempting to create one.")

    engine = get_sqlalchemy_engine()
    if engine:
        logger.info("Created session factory from engine")
        factory = sessionmaker(bind=engine)
        return factory

    logger.warning("Could not create a session factory - no engine found")
    return None


class DatetimeFormat(TypedDict, total=False):
    """Represents a datetime format pattern."""

    format: str
    description: str


def parse_datetime(value: str) -> datetime.datetime:
    """
    Parse a datetime string in various formats.

    Args:
        value: String representation of a datetime

    Returns:
        Parsed datetime object

    Raises:
        ValueError: If the datetime cannot be parsed in any known format
    """
    # Try ISO format first (with timezone)
    with contextlib.suppress(ValueError):
        # Handle 'Z' timezone indicator
        match value:
            case str(s) if s.endswith("Z"):
                value = s[:-1] + "+00:00"
            case _:
                pass
        return datetime.datetime.fromisoformat(value)

    # Try common formats
    formats: list[DatetimeFormat] = [
        {"format": "%Y-%m-%d %H:%M:%S%z", "description": "With timezone"},
        {"format": "%Y-%m-%d %H:%M:%S", "description": "Without timezone"},
        {"format": "%Y-%m-%dT%H:%M:%S%z", "description": "ISO-like with timezone"},
        {"format": "%Y-%m-%dT%H:%M:%S", "description": "ISO-like without timezone"},
        {"format": "%Y-%m-%d", "description": "Just date"},
    ]

    for fmt in formats:
        with contextlib.suppress(ValueError):
            return datetime.datetime.strptime(value, fmt["format"])

    # If we get here, none of our formats worked
    raise ValueError(f"Could not parse datetime string: {value}")


def coerce_attributes(attributes: dict[str, Any], model_class: type[Any]) -> dict[str, Any]:
    """
    Coerce attribute values to the correct types expected by the model.
    Uses SQLAlchemy's type system to handle conversion.

    Args:
        attributes: Dictionary containing entity attributes
        model_class: SQLAlchemy model class with column definitions

    Returns:
        Dictionary with values coerced to the correct types
    """
    result = attributes.copy()

    # Get the mapper for this model
    mapper = sqlalchemy_inspect(model_class)

    # Process each column
    for column_prop in mapper.column_attrs:
        column = column_prop.columns[0]
        key = column.key

        # Skip if the attribute is not provided or is None
        if key not in result or result[key] is None:
            continue

        # Get the column type and value
        sa_type = column.type
        value = result[key]
        type_name = sa_type.__class__.__name__.upper()

        # Handle specific types based on SQLAlchemy type
        with contextlib.suppress(ValueError, TypeError):
            # Handle datetimes - this is the most common issue with JSON data
            if type_name in ("DATETIME", "TIMESTAMP", "DATE") and isinstance(value, str):
                result[key] = parse_datetime(value)

            # Handle booleans
            elif type_name == "BOOLEAN" and isinstance(value, str):
                if value.lower() in ("true", "t", "yes", "y", "1"):
                    result[key] = True
                elif value.lower() in ("false", "f", "no", "n", "0"):
                    result[key] = False

            # Handle integers
            elif type_name in ("INTEGER", "BIGINT", "SMALLINT") and not isinstance(value, int):
                result[key] = int(value)

            # Handle floats
            elif type_name in ("FLOAT", "REAL", "NUMERIC", "DECIMAL") and not isinstance(value, float):
                result[key] = float(value)

    return result


def get_all_subclasses(cls: type) -> list[type]:
    all_subclasses: list[type] = []
    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))
    return all_subclasses


def bulk_delete(
    dataset: dict[str, dict[str, dict[str, Any]]],
    base: DeclarativeMeta,
    session: Session,
) -> dict[str, int]:
    """
    Remove all objects in the dataset from a SQLAlchemy session.

    Args:
        dataset: Dictionary where keys are table names and values are dictionaries of entities.
                The entity dictionaries are keyed by identity column and contain attribute dictionaries.
        base: SQLAlchemy declarative base class containing model definitions
        session: An open SQLAlchemy session

    Returns:
        Dictionary mapping table names to number of successfully deleted entities

    Raises:
        ModelNotFoundError: If a model class cannot be found for a table
    """
    removal_counts: dict[str, int] = {}

    # Get all model classes from the base
    models: dict[str, type] = {}

    for cls in get_all_subclasses(base):
        if hasattr(cls, "__tablename__"):
            models[cls.__tablename__] = cls

    # If no models found from subclasses, try to find them in the base's module
    if not models:
        base_module = inspect.getmodule(base)
        if base_module:
            for _, obj in inspect.getmembers(base_module):
                if inspect.isclass(obj) and issubclass(obj, base) and obj != base and hasattr(obj, "__tablename__"):
                    models[obj.__tablename__] = obj

    # Process tables in reverse order of metadata to handle foreign key constraints
    for table in reversed(base.metadata.sorted_tables):
        table_name = table.name
        if table_name not in dataset:
            continue

        if table_name not in models:
            raise ModelNotFoundError(f"No model class found for table {table_name}")

        model_class = models[table_name]
        entities = dataset[table_name]

        # Get primary key column(s)
        primary_key_cols = [col.name for col in table.primary_key.columns]

        count = 0
        for _, entity_data in entities.items():
            # Build a filter condition for each primary key
            filter_conditions = []
            for pk_col in primary_key_cols:
                if pk_col in entity_data:
                    filter_conditions.append(getattr(model_class, pk_col) == entity_data[pk_col])

            # If we have valid filter conditions, delete matching entities
            if filter_conditions:
                query: Query = session.query(model_class).filter(*filter_conditions)
                to_delete = query.all()

                for obj in to_delete:
                    session.delete(obj)
                    count += 1

        removal_counts[table_name] = count

    return removal_counts


def bulk_import(
    dataset: dict[str, dict[str, dict[str, Any]]], base: DeclarativeMeta, session: Session, fail_on_error: bool = False
) -> dict[str, int]:
    """
    Bulk import entities from a dictionary to SQLAlchemy models and commit to database.

    Args:
        dataset: Dictionary where keys are table names and values are dictionaries of entities.
                The entity dictionaries are keyed by identity column and contain attribute dictionaries.
        base: SQLAlchemy declarative base class containing model definitions
        session: An open SQLAlchemy session
        fail_on_error: Whether to fail on first error (True) or continue with remaining entities (False)

    Returns:
        Dictionary mapping table names to number of successfully imported entities

    Raises:
        GoldenError: If fail_on_error is True and an error occurs during import
        ModelNotFoundError: If a model class cannot be found for a table
    """
    success_counts: dict[str, int] = {}
    errors: list[EntityImportError] = []

    # Find all model classes defined using the base
    models: dict[str, type] = {}

    # First, check base's subclasses
    model_classes = get_all_subclasses(base)

    for cls in model_classes:
        if hasattr(cls, "__tablename__"):
            models[cls.__tablename__] = cls

    # If no models found from subclasses, try to find them in the base's module
    if not models:
        base_module = inspect.getmodule(base)
        if base_module:
            for _, obj in inspect.getmembers(base_module):
                if inspect.isclass(obj) and issubclass(obj, base) and obj != base and hasattr(obj, "__tablename__"):
                    models[obj.__tablename__] = obj

    if not models:
        logger.warning("No model classes found for bulk import")

    # Use SQLAlchemy's table ordering to handle dependencies correctly
    for table in base.metadata.sorted_tables:
        table_name = table.name
        if table_name not in dataset:
            logger.debug(f"Table {table_name} not in dataset, skipping")
            continue

        if table_name not in models:
            msg = f"No model class found for table {table_name}"
            if fail_on_error:
                raise ModelNotFoundError(msg)
            logger.warning(f"{msg}, skipping")
            continue

        logger.info(f"Processing table: {table_name}")
        entities = dataset[table_name]
        model_class = models[table_name]

        success_count = 0
        # Process each entity in the table
        for entity_id, attributes in entities.items():
            try:
                # Create a new model instance with the attributes
                coerced_attributes = coerce_attributes(attributes, model_class)
                model_instance = model_class(**coerced_attributes)
                session.merge(model_instance, load=True)
                success_count += 1
            except Exception as e:
                error = EntityImportError(table_name, entity_id, e)
                if fail_on_error:
                    raise error from e
                errors.append(error)
                logger.error(str(error))

        success_counts[table_name] = success_count

    # If we collected errors but didn't fail immediately, now raise them as a group
    if errors and not fail_on_error:
        # In Python 3.11+, we could use ExceptionGroup
        # but we'll just log them for now
        logger.error(f"Encountered {len(errors)} errors during import")

    return success_counts


def get_model_class[T](model_name: str, search_paths: list[str], package: str | None = None) -> type[T] | None:
    """
    Find a model class by name in a list of search paths.

    Args:
        model_name: Name of the model class to find
        search_paths: List of module paths to search
        package: Optional package prefix for relative imports

    Returns:
        The model class if found, None otherwise
    """
    return find_class_by_name(model_name, search_paths, package)  # type: ignore


def sum_dicts(dict1: dict[str, int], dict2: dict[str, int]) -> dict[str, int]:
    from collections import Counter

    return dict(Counter(dict1) + Counter(dict2))
