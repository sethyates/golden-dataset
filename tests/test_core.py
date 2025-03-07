import datetime
import importlib
import sys
from typing import Any
from unittest import mock

import pytest
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from golden_dataset import core


# Create test fixtures
@pytest.fixture
def sample_params() -> list[core.ParamInfo]:
    return [
        {
            "name": "test_param",
            "kind": "POSITIONAL_OR_KEYWORD",
            "has_default": True,
            "default": "default_value",
            "type": str,
        }
    ]


@pytest.fixture
def create_test_module():
    """Create a temporary module for testing imports."""

    def _create_module(module_name, content):
        spec = importlib.util.spec_from_loader(
            module_name, loader=importlib.machinery.SourceFileLoader(module_name, "<string>")
        )
        if spec is None:
            return None

        module = importlib.util.module_from_spec(spec)
        exec(content, module.__dict__)
        sys.modules[module_name] = module
        return module

    return _create_module


@pytest.fixture
def cleanup_modules():
    """Fixture to clean up test modules after tests."""
    modules_before = set(sys.modules.keys())

    yield

    # Remove modules added during the test
    for module_name in set(sys.modules.keys()) - modules_before:
        if module_name.startswith("test_module"):
            del sys.modules[module_name]


@pytest.fixture
def sqlalchemy_base():
    """Create a SQLAlchemy Base for testing."""
    return declarative_base()


@pytest.fixture
def sqlalchemy_models(sqlalchemy_base):
    """Define test models using the SQLAlchemy Base."""

    class User(sqlalchemy_base):  # type: ignore
        __tablename__ = "users"
        id = Column(Integer, primary_key=True)
        name = Column(String)
        email = Column(String)
        created_at = Column(DateTime)
        is_active = Column(Boolean)
        score = Column(Float)

    class Post(sqlalchemy_base):  # type: ignore
        __tablename__ = "posts"
        id = Column(Integer, primary_key=True)
        user_id = Column(Integer)
        title = Column(String)
        content = Column(String)

    return {"User": User, "Post": Post}


@pytest.fixture
def sqlalchemy_engine():
    """Create an in-memory SQLite engine for testing."""
    return create_engine("sqlite:///:memory:")


@pytest.fixture
def sqlalchemy_session(sqlalchemy_engine, sqlalchemy_models, sqlalchemy_base):
    """Create a session and set up/tear down the database."""
    sqlalchemy_base.metadata.create_all(sqlalchemy_engine)
    SessionFactory = sessionmaker(bind=sqlalchemy_engine)
    session = SessionFactory()

    yield session

    session.close()
    sqlalchemy_base.metadata.drop_all(sqlalchemy_engine)


@pytest.fixture
def sample_dataset(sqlalchemy_models):
    """Create a sample dataset for testing import/delete."""
    return {
        "users": {
            "user1": {
                "id": 1,
                "name": "John Doe",
                "email": "john@example.com",
                "created_at": "2023-01-01T12:00:00",
                "is_active": "true",
                "score": "95.5",
            },
            "user2": {
                "id": 2,
                "name": "Jane Smith",
                "email": "jane@example.com",
                "created_at": "2023-01-02T14:30:00",
                "is_active": "false",
                "score": "87.2",
            },
        },
        "posts": {
            "post1": {"id": 1, "user_id": 1, "title": "First Post", "content": "Hello World"},
            "post2": {"id": 2, "user_id": 2, "title": "Another Post", "content": "Testing"},
        },
    }


# Test functions
def test_is_same_class():
    """Test the is_same_class function with various inputs."""

    # Test with regular classes
    class TestClass1:
        pass

    class TestClass2:
        pass

    # Same class comparison
    assert core.is_same_class(TestClass1, TestClass1) is True

    # Different class comparison
    assert core.is_same_class(TestClass1, TestClass2) is False

    # Compare with Any
    assert core.is_same_class(Any, Any) is True
    assert core.is_same_class(TestClass1, Any) is False

    # Classes with same name but different modules
    cls1 = type("SameNameClass", (), {"__module__": "module1"})
    cls2 = type("SameNameClass", (), {"__module__": "module2"})

    assert core.is_same_class(cls1, cls2) is False


def test_parse_import_path():
    """Test the parse_import_path function."""
    # Test with module and object
    module_path, object_name = core.parse_import_path("module.path:object_name")
    assert module_path == "module.path"
    assert object_name == "object_name"

    # Test with just object name
    module_path, object_name = core.parse_import_path("object_name")
    assert module_path == ""
    assert object_name == "object_name"

    # Test with empty object name
    with pytest.raises(ValueError):
        core.parse_import_path("module.path:")


def test_get_function(create_test_module, cleanup_modules):
    """Test the get_function function."""
    # Test getting a function
    func, name, params = core.get_function("test_module_function:test_function", generator_root="tests.fixtures")

    assert callable(func)
    assert name == "test_function"
    assert len(params) == 2
    assert params[0]["name"] == "param1"
    assert params[0]["type"] is str
    assert params[1]["name"] == "param2"
    assert params[1]["type"] is int
    assert params[1]["has_default"] is True
    assert params[1]["default"] == 10

    # Test function not found
    with pytest.raises(AttributeError):
        core.get_function("test_module_function:nonexistent_function", generator_root="tests.fixtures")

    # Test not a callable
    with pytest.raises(TypeError):
        core.get_function("test_module_function:test_not_callable", generator_root="tests.fixtures")


def test_find_class_by_name(create_test_module, cleanup_modules):
    """Test the find_class_by_name function."""
    # Test finding a class
    cls = core.find_class_by_name("TestClass1", ["test_module1", "test_module2"], package="tests.fixtures")
    assert cls is not None
    assert cls.__name__ == "TestClass1"

    # Test finding a class in the second module
    cls = core.find_class_by_name("TestClass2", ["test_module1", "test_module2"], package="tests.fixtures")
    assert cls is not None
    assert cls.__name__ == "TestClass2"

    # Test class not found
    cls = core.find_class_by_name("NonexistentClass", ["test_module1", "test_module2"], package="tests.fixtures")
    assert cls is None


def test_get_sqlalchemy_base(create_test_module, cleanup_modules):
    """Test the get_sqlalchemy_base function."""
    # Mock find_class_by_name to return the Base from our test module
    # Test getting the Base class
    base = core.get_sqlalchemy_base(search_path=["test_module_models"], package="tests.fixtures")
    assert base is not None
    assert base.__name__ == "Base"

    # Test Base not found
    base = core.get_sqlalchemy_base(search_path=["nonexistent_module"], package="tests.fixtures")
    assert base is None


def test_parse_datetime():
    """Test the parse_datetime function with various formats."""
    # Test ISO format
    dt = core.parse_datetime("2023-01-01T12:00:00")
    assert dt == datetime.datetime(2023, 1, 1, 12, 0, 0)

    # Test ISO format with Z timezone
    dt = core.parse_datetime("2023-01-01T12:00:00Z")
    assert dt == datetime.datetime(2023, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)

    # Test ISO format with explicit timezone
    dt = core.parse_datetime("2023-01-01T12:00:00+00:00")
    assert dt == datetime.datetime(2023, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)

    # Test common format with space
    dt = core.parse_datetime("2023-01-01 12:00:00")
    assert dt == datetime.datetime(2023, 1, 1, 12, 0, 0)

    # Test date only
    dt = core.parse_datetime("2023-01-01")
    assert dt == datetime.datetime(2023, 1, 1, 0, 0, 0)

    # Test invalid format
    with pytest.raises(ValueError):
        core.parse_datetime("not-a-date")


def test_coerce_attributes(sqlalchemy_models):
    """Test the coerce_attributes function."""
    User = sqlalchemy_models["User"]

    # Test with various attribute types
    attributes = {
        "id": "1",  # Should be converted to int
        "name": "Test User",  # String stays string
        "email": "test@example.com",
        "created_at": "2023-01-01T12:00:00",  # Should be converted to datetime
        "is_active": "true",  # Should be converted to bool
        "score": "95.5",  # Should be converted to float
    }

    coerced = core.coerce_attributes(attributes, User)

    assert isinstance(coerced["id"], int)
    assert coerced["id"] == 1

    assert isinstance(coerced["name"], str)
    assert coerced["name"] == "Test User"

    assert isinstance(coerced["created_at"], datetime.datetime)
    assert coerced["created_at"] == datetime.datetime(2023, 1, 1, 12, 0, 0)

    assert isinstance(coerced["is_active"], bool)
    assert coerced["is_active"] is True

    assert isinstance(coerced["score"], float)
    assert coerced["score"] == 95.5


def test_get_all_subclasses():
    """Test the get_all_subclasses function."""

    class BaseClass:
        pass

    class Level1A(BaseClass):
        pass

    class Level1B(BaseClass):
        pass

    class Level2(Level1A):
        pass

    subclasses = core.get_all_subclasses(BaseClass)

    assert len(subclasses) == 3
    assert Level1A in subclasses
    assert Level1B in subclasses
    assert Level2 in subclasses


def test_bulk_import(sqlalchemy_base, sqlalchemy_session, sample_dataset):
    """Test the bulk_import function."""
    # Test successful import
    result = core.bulk_import(sample_dataset, sqlalchemy_base, sqlalchemy_session)

    assert "users" in result
    assert result["users"] == 2
    assert "posts" in result
    assert result["posts"] == 2

    # Verify data was imported correctly
    User = sqlalchemy_base.metadata.tables["users"]
    users = sqlalchemy_session.query(User).all()
    assert len(users) == 2

    # Test with nonexistent model
    # bad_dataset = {"nonexistent_table": {"entity1": {"id": 1}}}
    #
    # with pytest.raises(core.ModelNotFoundError):
    #     core.bulk_import(bad_dataset, sqlalchemy_base, sqlalchemy_session, fail_on_error=True)
    #
    # # Test with fail_on_error=False
    # result = core.bulk_import(bad_dataset, sqlalchemy_base, sqlalchemy_session, fail_on_error=False)
    # assert result == {}


def test_bulk_delete(sqlalchemy_base, sqlalchemy_session, sample_dataset):
    """Test the bulk_delete function."""
    # First import data
    core.bulk_import(sample_dataset, sqlalchemy_base, sqlalchemy_session)

    # Then test deletion
    result = core.bulk_delete(sample_dataset, sqlalchemy_base, sqlalchemy_session)

    assert "users" in result
    assert result["users"] == 2
    assert "posts" in result
    assert result["posts"] == 2

    # Verify data was deleted
    User = sqlalchemy_base.metadata.tables["users"]
    users = sqlalchemy_session.query(User).all()
    assert len(users) == 0

    # Test with nonexistent model
    # bad_dataset = {"nonexistent_table": {"entity1": {"id": 1}}}
    #
    # with pytest.raises(core.ModelNotFoundError):
    #     core.bulk_delete(bad_dataset, sqlalchemy_base, sqlalchemy_session)


def test_get_model_class(create_test_module, cleanup_modules):
    """Test the get_model_class function."""
    # Create a test module with a model class
    # Test finding a model class
    model_class = core.get_model_class("TestModel", ["test_module_models"], package="tests.fixtures")
    assert model_class is not None
    assert model_class.__name__ == "TestModel"

    # Test model not found
    model_class = core.get_model_class("NonexistentModel", ["test_module_models"], package="tests.fixtures")
    assert model_class is None


def test_sum_dicts():
    """Test the sum_dicts function."""
    dict1 = {"a": 1, "b": 2, "c": 3}
    dict2 = {"b": 3, "c": 4, "d": 5}

    result = core.sum_dicts(dict1, dict2)

    assert result == {"a": 1, "b": 5, "c": 7, "d": 5}

    # Test with empty dicts
    assert core.sum_dicts({}, {}) == {}
    assert core.sum_dicts(dict1, {}) == dict1
    assert core.sum_dicts({}, dict2) == dict2


# Additional tests for edge cases and error handling
def test_get_sqlalchemy_engine(create_test_module, cleanup_modules):
    """Test the get_sqlalchemy_engine function."""
    # Test finding the engine
    engine = core.get_sqlalchemy_engine(search_path=["test_module_engine"], package="tests.fixtures")
    assert engine is not None

    # Test engine not found
    engine = core.get_sqlalchemy_engine(search_path=["nonexistent_module"], package="tests.fixtures")
    assert engine is None


def test_get_sqlalchemy_session_factory(create_test_module, cleanup_modules, sqlalchemy_engine):
    """Test the get_sqlalchemy_session_factory function."""
    # Test finding the session factory
    with mock.patch("golden_dataset.core.find_class_by_name") as mock_find_class:
        mock_find_class.side_effect = (
            lambda name, path, package: sys.modules["test_module_session"].Session if name == "Session" else None
        )

        session_factory = core.get_sqlalchemy_session_factory(
            search_path=["test_module_session"], package="tests.fixtures"
        )
        assert session_factory is not None
        assert callable(session_factory)

        # Test with engine fallback
        mock_find_class.return_value = None
        with mock.patch("golden_dataset.core.get_sqlalchemy_engine") as mock_get_engine:
            mock_get_engine.return_value = sqlalchemy_engine
            session_factory = core.get_sqlalchemy_session_factory(
                search_path=["nonexistent_module"], package="tests.fixtures"
            )
            assert session_factory is not None
            assert callable(session_factory)

            # Test with no engine available
            mock_get_engine.return_value = None
            session_factory = core.get_sqlalchemy_session_factory(
                search_path=["nonexistent_module"], package="tests.fixtures"
            )
            assert session_factory is None
