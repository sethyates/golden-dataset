import json
import tempfile
from pathlib import Path

import pytest

from golden_dataset import GoldenDataset, GoldenManager, GoldenSession, GoldenSettings, golden
from golden_dataset.exc import DatasetNotFoundError, GoldenError

# Import the module to test
from .fixtures.mock_module import Post, SimpleUser, User, UserModel


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def test_settings(temp_dir):
    """Create test settings with temporary directories."""
    datasets_dir = temp_dir / "datasets"
    src_dir = "tests"

    datasets_dir.mkdir(exist_ok=True)

    settings = GoldenSettings()
    settings.datasets_dir = str(datasets_dir)
    settings.src_dir = str(src_dir)
    settings.generators = "tests.fixtures"
    settings.base_class_name = "test_module_models:Base"
    settings.engine_name = "test_module_engine:engine"
    settings.session_factory_name = "test_module_engine:Session"
    return settings


@pytest.fixture
def test_manager(test_settings):
    """Create a test golden manager."""
    print(test_settings)
    return GoldenManager(settings=test_settings)


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    return GoldenDataset(
        name="test_dataset", title="Test Dataset", description="A test dataset", tables={"users": 2, "posts": 3}
    )


@pytest.fixture
def populated_dataset():
    """Create a dataset with data."""
    dataset = GoldenDataset(name="populated_dataset", title="Populated Dataset", description="A dataset with data")

    # Add SQLAlchemy objects
    user1 = User(id=1, name="John Doe", email="john@example.com")
    user2 = User(id=2, name="Jane Smith", email="jane@example.com")

    post1 = Post(id=1, user_id=1, title="First Post", content="Hello World")
    post2 = Post(id=2, user_id=1, title="Second Post", content="Another post")
    post3 = Post(id=3, user_id=2, title="Jane's Post", content="Hi there")

    dataset.add(user1)
    dataset.add(user2)
    dataset.add(post1)
    dataset.add(post2)
    dataset.add(post3)

    return dataset


@pytest.fixture
def setup_dataset_dir(temp_dir, populated_dataset, test_manager):
    """Set up a dataset directory with files for testing."""
    dataset_dir = temp_dir / "datasets" / populated_dataset.name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Save the dataset
    test_manager.dump_dataset(populated_dataset)

    return dataset_dir


# Tests for GoldenSettings
def test_golden_settings_defaults():
    """Test that GoldenSettings has correct defaults."""
    settings = GoldenSettings()

    if settings.generators == "golden":
        assert settings.src_dir == "src"
        assert settings.datasets_dir == "tests/data/golden"
        assert settings.generators == "golden"
        assert settings.base_class_name == "Base"
        assert settings.engine_name == "engine"
        assert settings.session_factory_name == "Session"
    else:
        assert settings.src_dir == "examples"
        assert settings.datasets_dir == "tests/data/golden"
        assert settings.generators == "db.datasets"
        assert settings.base_class_name == "db.models:Base"
        assert settings.engine_name == "db.models:engine"
        assert settings.session_factory_name == "db.models:Session"


# Tests for GoldenDataset
def test_golden_dataset_initialization():
    """Test initializing a golden dataset."""
    dataset = GoldenDataset(name="test")

    assert dataset.name == "test"
    assert dataset.revision == "1"
    assert dataset.title is None
    assert dataset.description == ""
    assert dataset.dependencies == []
    assert dataset.tables == {}
    assert dataset.exported_at is None


def test_golden_dataset_add_object():
    """Test adding objects to a dataset."""
    dataset = GoldenDataset(name="test")

    # Add a SQLAlchemy object
    user = User(id=1, name="Test User", email="test@example.com")
    result = dataset.add(user)

    # Verify result is the same object
    assert result is user

    # Verify tables are updated
    assert dataset.tables == {"users": 1}

    # Add another object
    post = Post(id=1, user_id=1, title="Test Post", content="Hello")
    dataset.add(post)

    assert dataset.tables == {"users": 1, "posts": 1}


def test_golden_dataset_query():
    """Test querying objects from a dataset."""
    dataset = GoldenDataset(name="test")

    # Add objects
    user1 = User(id=1, name="User 1", email="user1@example.com")
    user2 = User(id=2, name="User 2", email="user2@example.com")
    post = Post(id=1, user_id=1, title="Post", content="Content")

    dataset.add(user1)
    dataset.add(user2)
    dataset.add(post)

    # Query users
    users = dataset.query(User)
    assert len(users) == 2
    assert all(isinstance(u, User) for u in users)

    # Query posts
    posts = dataset.query(Post)
    assert len(posts) == 1
    assert all(isinstance(p, Post) for p in posts)


def test_golden_dataset_get():
    """Test getting an object by ID."""
    dataset = GoldenDataset(name="test")

    # Add objects
    user1 = User(id=1, name="User 1", email="user1@example.com")
    user2 = User(id=2, name="User 2", email="user2@example.com")

    dataset.add(user1)
    dataset.add(user2)

    # Get by ID
    result = dataset.get(User, 1)
    assert result is user1

    # Get nonexistent ID
    result = dataset.get(User, 99)
    assert result is None


def test_golden_dataset_clear():
    """Test clearing a dataset."""
    dataset = GoldenDataset(name="test")

    # Add objects
    user = User(id=1, name="User", email="user@example.com")
    post = Post(id=1, user_id=1, title="Post", content="Content")

    dataset.add(user)
    dataset.add(post)

    assert dataset.tables == {"users": 1, "posts": 1}

    # Clear the dataset
    dataset.clear()

    assert dataset.tables == {}
    assert dataset.query(User) == []
    assert dataset.query(Post) == []


def test_golden_dataset_serialize_object_types():
    """Test serializing different types of objects."""
    dataset = GoldenDataset(name="test")

    # Test SQLAlchemy object
    sa_user = User(id=1, name="SA User", email="sa@example.com")
    sa_result = dataset._serialize_object(sa_user)
    assert sa_result == {"id": 1, "name": "SA User", "email": "sa@example.com"}

    # Test Pydantic object
    pydantic_user = UserModel(id=2, name="Pydantic User", email="pydantic@example.com")
    pydantic_result = dataset._serialize_object(pydantic_user)
    assert pydantic_result == {"id": 2, "name": "Pydantic User", "email": "pydantic@example.com"}

    # Test regular class
    simple_user = SimpleUser(3, "Simple User", "simple@example.com")
    simple_result = dataset._serialize_object(simple_user)
    assert simple_result == {"id": "3", "name": "Simple User", "email": "simple@example.com"}


def test_golden_dataset_get_or_create_id():
    """Test getting or creating an ID for an object."""
    dataset = GoldenDataset(name="test")

    # Object with ID
    user = User(id=1, name="User", email="user@example.com")
    assert dataset._get_or_create_id(user) == 1

    # Object without ID
    user_no_id = User(name="No ID", email="noid@example.com")
    id_result = dataset._get_or_create_id(user_no_id)
    assert id_result is not None  # Should generate a UUID

    # Simple object
    class NoIdObject:
        pass

    obj = NoIdObject()
    id_result = dataset._get_or_create_id(obj)
    assert id_result is not None  # Should generate a UUID


def test_golden_dataset_get_table_name():
    """Test getting the table name for different objects."""
    dataset = GoldenDataset(name="test")

    # SQLAlchemy model
    assert dataset._get_table_name(User(id=1)) == "users"

    # Regular class
    class TestClass:
        pass

    assert dataset._get_table_name(TestClass()) == "testclass"


def test_golden_dataset_get_model_table_name():
    """Test getting the table name for model classes."""
    dataset = GoldenDataset(name="test")

    # SQLAlchemy model with __tablename__
    assert dataset._get_model_table_name(User) == "users"

    # Regular class
    class TestClass:
        pass

    assert dataset._get_model_table_name(TestClass) == "testclass"


# Tests for GoldenSession
def test_golden_session_initialization():
    """Test initializing a golden session."""
    dataset = GoldenDataset(name="test")
    session = GoldenSession(dataset)

    assert session.dataset is dataset


def test_golden_session_add():
    """Test adding an object to a session."""
    dataset = GoldenDataset(name="test")
    session = GoldenSession(dataset)

    user = User(id=1, name="Test User", email="test@example.com")
    result = session.add(user)

    assert result is user
    assert dataset.tables == {"users": 1}


def test_golden_session_add_all():
    """Test adding multiple objects to a session."""
    dataset = GoldenDataset(name="test")
    session = GoldenSession(dataset)

    user1 = User(id=1, name="User 1", email="user1@example.com")
    user2 = User(id=2, name="User 2", email="user2@example.com")

    session.add_all([user1, user2])

    assert dataset.tables == {"users": 2}
    assert len(dataset.query(User)) == 2


def test_golden_session_query():
    """Test querying objects from a session."""
    dataset = GoldenDataset(name="test")
    session = GoldenSession(dataset)

    user = User(id=1, name="Test User", email="test@example.com")
    session.add(user)

    users = session.query(User)
    assert len(users) == 1
    assert users[0] is user


def test_golden_session_get():
    """Test getting an object by ID from a session."""
    dataset = GoldenDataset(name="test")
    session = GoldenSession(dataset)

    user = User(id=1, name="Test User", email="test@example.com")
    session.add(user)

    result = session.get(User, 1)
    assert result is user

    result = session.get(User, 99)
    assert result is None


def test_golden_session_operations_on_closed_session():
    """Test operations on a closed session raise exceptions."""
    dataset = GoldenDataset(name="test")
    session = GoldenSession(dataset)

    # Close the session
    session.close()

    # All operations should raise InvalidRequestError
    with pytest.raises(Exception, match="closed session"):
        session.add(User(id=1))

    with pytest.raises(Exception, match="closed session"):
        session.add_all([User(id=1)])

    with pytest.raises(Exception, match="closed session"):
        session.refresh(User(id=1))

    with pytest.raises(Exception, match="closed session"):
        session.query(User)

    with pytest.raises(Exception, match="closed session"):
        session.get(User, 1)

    with pytest.raises(Exception, match="closed session"):
        session.commit()

    with pytest.raises(Exception, match="closed session"):
        session.rollback()

    with pytest.raises(Exception, match="closed session"):
        session.close()


def test_golden_session_context_manager():
    """Test using a golden session as a context manager."""
    dataset = GoldenDataset(name="test")

    # Normal execution
    with GoldenSession(dataset) as session:
        user = User(id=1, name="Test User", email="test@example.com")
        session.add(user)

    # Session should be closed after context
    assert session.dataset is None

    # Exception in context
    try:
        with GoldenSession(dataset) as session:
            session.add(User(id=2))
            raise ValueError("Test exception")
    except ValueError:
        pass

    # Session should be closed and rolled back
    assert session.dataset is None
    assert dataset.tables == {}  # Should be empty due to rollback


# Tests for GoldenManager
def test_golden_manager_initialization(test_settings):
    """Test initializing a golden manager."""
    manager = GoldenManager(settings=test_settings)

    assert manager.datasets_dir == Path(test_settings.datasets_dir)
    assert manager.src_dir == Path(test_settings.src_dir)
    assert manager.generators == "tests.fixtures"


def test_golden_manager_dataset():
    """Test creating a dataset with a golden manager."""
    settings = GoldenSettings()
    manager = GoldenManager(settings=settings)

    dataset = manager.dataset(
        name="test", title="Test Dataset", description="A test dataset", dependencies=["dep1", "dep2"]
    )

    assert dataset.name == "test"
    assert dataset.title == "Test Dataset"
    assert dataset.description == "A test dataset"
    assert dataset.dependencies == ["dep1", "dep2"]


def test_golden_manager_session(sample_dataset):
    """Test creating a session with a golden manager."""
    settings = GoldenSettings()
    manager = GoldenManager(settings=settings)

    session = manager.session(sample_dataset)

    assert isinstance(session, GoldenSession)
    assert session.dataset is sample_dataset


def test_golden_manager_dump_dataset(test_manager, populated_dataset):
    """Test dumping a dataset to files."""
    # Dump the dataset
    test_manager.dump_dataset(populated_dataset)

    # Check that files were created
    dataset_dir = Path(test_manager.datasets_dir) / populated_dataset.name
    assert dataset_dir.exists()

    # Check metadata file
    metadata_file = dataset_dir / "_metadata.json"
    assert metadata_file.exists()

    with open(metadata_file) as f:
        metadata = json.load(f)
        assert metadata["name"] == populated_dataset.name
        assert metadata["title"] == populated_dataset.title
        assert metadata["description"] == populated_dataset.description

    # Check table files
    users_file = dataset_dir / "users.json"
    posts_file = dataset_dir / "posts.json"

    assert users_file.exists()
    assert posts_file.exists()

    with open(users_file) as f:
        users_data = json.load(f)
        assert len(users_data["data"]) == 2

    with open(posts_file) as f:
        posts_data = json.load(f)
        assert len(posts_data["data"]) == 3


def test_golden_manager_open_dataset(test_manager, setup_dataset_dir):
    """Test opening a dataset from files."""
    # Open the dataset
    dataset = test_manager.open_dataset("populated_dataset")

    assert dataset.name == "populated_dataset"
    assert dataset.title == "Populated Dataset"
    assert dataset.description == "A dataset with data"

    # Tables should be recorded but not loaded yet
    assert "users" in dataset.tables
    assert "posts" in dataset.tables


def test_golden_manager_load_dataset(test_manager, setup_dataset_dir):
    """Test loading a dataset with all its data."""
    # Load the dataset
    dataset = test_manager.load_dataset("populated_dataset")

    assert dataset.name == "populated_dataset"

    # Verify that data was loaded
    users = dataset.query(User)
    posts = dataset.query(Post)

    # Data should be loaded as dictionaries
    assert len(users) == 2
    assert len(posts) == 3


def test_golden_manager_list_datasets(test_manager, setup_dataset_dir):
    """Test listing available datasets."""
    # Create another dataset
    another_dataset = GoldenDataset(name="another_dataset", title="Another Dataset")
    test_manager.dump_dataset(another_dataset)

    # List datasets
    datasets = test_manager.list_datasets()

    assert len(datasets) == 2
    dataset_names = [d.name for d in datasets]
    assert "populated_dataset" in dataset_names
    assert "another_dataset" in dataset_names


def test_golden_manager_dataset_not_found(test_manager):
    """Test opening a nonexistent dataset."""
    with pytest.raises(DatasetNotFoundError):
        test_manager.open_dataset("nonexistent_dataset")


# Tests for generate_dataset
def test_golden_manager_generate_dataset(test_manager):
    """Test generating a dataset from a function."""
    dataset = test_manager.generate_dataset("mock_module:mock_generator")

    assert dataset.name == "mock_generator"
    assert "users" in dataset.tables
    assert dataset.tables["users"] == 1


def test_golden_manager_generate_dataset_invalid_signature(test_manager):
    """Test generating a dataset with an invalid function signature."""
    with pytest.raises(GoldenError, match="must have session as first parameter"):
        test_manager.generate_dataset("mock_module:invalid_func")


# Tests for Golden decorator
def test_golden_decorator_simple():
    """Test simple @golden decorator usage."""

    # Create a decorated function
    @golden
    def test_generator(session):
        user = User(id=1, name="Test User", email="test@example.com")
        session.add(user)

    # Check attributes
    assert test_generator.__golden__ is True
    assert test_generator.__name__ == "test_generator"
    assert test_generator.__dependencies__ == []


def test_golden_decorator_with_params():
    """Test @golden decorator with parameters."""

    # Create a decorated function with parameters
    @golden(name="custom_name", title="Custom Title", description="Custom description", dependencies=["dep1", "dep2"])
    def test_generator(session):
        user = User(id=1, name="Test User", email="test@example.com")
        session.add(user)

    # Check attributes
    assert test_generator.__golden__ is True
    assert test_generator.__name__ == "custom_name"
    assert test_generator.__title__ == "Custom Title"
    assert test_generator.__description__ == "Custom description"
    assert test_generator.__dependencies__ == ["dep1", "dep2"]


def test_golden_decorator_function_execution():
    """Test that decorated functions still execute normally."""

    # Create a decorated function
    @golden
    def test_generator(session, param1="default"):
        user = User(id=1, name=f"User {param1}", email="test@example.com")
        session.add(user)
        return user

    # Create a dataset and session
    dataset = GoldenDataset(name="test")
    session = GoldenSession(dataset)

    # Execute the function
    result = test_generator(session, param1="custom")

    # Check function execution
    assert result.name == "User custom"
    assert dataset.tables["users"] == 1


def test_golden_instance_settings():
    """Test creating a Golden instance with custom settings."""
    # Create custom settings
    custom_golden = GoldenSettings()
    custom_golden.datasets_dir = "/custom/datasets"
    custom_golden.generators = "custom.generators"
    custom_golden.src_dir = "/custom/src"

    # Check settings
    assert custom_golden.datasets_dir == "/custom/datasets"
    assert custom_golden.generators == "custom.generators"
    assert custom_golden.src_dir == "/custom/src"
