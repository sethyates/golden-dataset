import json
from pathlib import Path
from unittest import mock

import pytest
from typer.testing import CliRunner

import golden_dataset.cli
from golden_dataset import GoldenDataset, GoldenSettings
from golden_dataset.exc import DatasetNotFoundError, GoldenError


@pytest.fixture
def cli_runner():
    """Fixture for running CLI commands."""
    return CliRunner()


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test data."""
    return tmp_path


@pytest.fixture
def test_settings(temp_dir):
    """Create test settings with temporary directories."""
    datasets_dir = temp_dir / "datasets"
    src_dir = "tests"

    datasets_dir.mkdir(exist_ok=True)

    settings = GoldenSettings()
    settings.datasets_dir = str(datasets_dir)
    settings.src_dir = str(src_dir)
    settings.generators = "fixtures"
    settings.base_class_name = "test_module_models:Base"
    settings.engine_name = "test_module_engine:engine"
    settings.session_factory_name = "test_module_engine:Session"

    with mock.patch.object(golden_dataset.cli, "settings", settings):
        yield settings


@pytest.fixture
def sample_dataset(test_settings):
    """Create a sample dataset for testing."""
    dataset = GoldenDataset(
        name="test_dataset",
        title="Test Dataset",
        description="A sample dataset for testing",
        tables={"users": 2, "posts": 3},
    )

    # Create dataset directory and files
    dataset_dir = Path(test_settings.datasets_dir) / dataset.name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Create metadata file
    metadata_file = dataset_dir / "_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(dataset.model_dump(mode="json"), f, indent=2)

    # Create table files
    users_file = dataset_dir / "users.json"
    with open(users_file, "w") as f:
        json.dump(
            {
                "data": [
                    {"id": 1, "name": "User 1", "email": "user1@example.com"},
                    {"id": 2, "name": "User 2", "email": "user2@example.com"},
                ]
            },
            f,
            indent=2,
        )

    posts_file = dataset_dir / "posts.json"
    with open(posts_file, "w") as f:
        json.dump(
            {
                "data": [
                    {"id": 1, "user_id": 1, "title": "Post 1", "content": "Content 1"},
                    {"id": 2, "user_id": 1, "title": "Post 2", "content": "Content 2"},
                    {"id": 3, "user_id": 2, "title": "Post 3", "content": "Content 3"},
                ]
            },
            f,
            indent=2,
        )

    return dataset


@pytest.fixture
def dependency_dataset(test_settings):
    """Create a dependency dataset."""
    dataset = GoldenDataset(
        name="dependency_dataset",
        title="Dependency Dataset",
        description="A dependency dataset",
        tables={"categories": 1},
    )

    # Create dataset directory and files
    dataset_dir = Path(test_settings.datasets_dir) / dataset.name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Create metadata file
    metadata_file = dataset_dir / "_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(dataset.model_dump(mode="json"), f, indent=2)

    # Create table file
    categories_file = dataset_dir / "categories.json"
    with open(categories_file, "w") as f:
        json.dump({"data": [{"id": 1, "name": "Category 1"}]}, f, indent=2)

    return dataset


@pytest.fixture
def complex_dataset(test_settings, dependency_dataset):
    """Create a dataset with dependencies."""
    dataset = GoldenDataset(
        name="complex_dataset",
        title="Complex Dataset",
        description="A dataset with dependencies",
        dependencies=[dependency_dataset.name],
        tables={"articles": 2},
    )

    # Create dataset directory and files
    dataset_dir = Path(test_settings.datasets_dir) / dataset.name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Create metadata file
    metadata_file = dataset_dir / "_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(dataset.model_dump(mode="json"), f, indent=2)

    # Create table file
    articles_file = dataset_dir / "articles.json"
    with open(articles_file, "w") as f:
        json.dump(
            {
                "data": [
                    {"id": 1, "category_id": 1, "title": "Article 1", "content": "Content 1"},
                    {"id": 2, "category_id": 1, "title": "Article 2", "content": "Content 2"},
                ]
            },
            f,
            indent=2,
        )

    return dataset


@pytest.fixture
def mock_sqlalchemy_components():
    """Mock SQLAlchemy components for testing."""
    with (
        mock.patch.object(golden_dataset.cli, "get_sqlalchemy_engine") as mock_engine,
        mock.patch.object(golden_dataset.cli, "get_sqlalchemy_base") as mock_base,
        mock.patch.object(golden_dataset.cli, "get_sqlalchemy_session_factory") as mock_session,
    ):
        # Create mock session
        mock_session_instance = mock.MagicMock()
        mock_session.return_value = mock.MagicMock(return_value=mock_session_instance)

        # Mock database operations
        mock_session_instance.__enter__.return_value = mock_session_instance
        mock_session_instance.commit.return_value = None

        mock_engine.return_value = mock.MagicMock()
        mock_base.return_value = mock.MagicMock()

        yield {
            "engine": mock_engine,
            "base": mock_base,
            "session_factory": mock_session,
            "session": mock_session_instance,
        }


# Test the main callback
def test_main_callback(cli_runner, test_settings):
    """Test the main callback function."""
    result = cli_runner.invoke(golden_dataset.cli.app, [])
    assert result.exit_code == 2


# Test list_datasets command
def test_list_datasets_success(cli_runner, test_settings, sample_dataset):
    """Test listing datasets successfully."""
    with mock.patch.object(golden_dataset.cli, "GoldenManager") as mock_manager_cls:
        # Mock manager instance and list_datasets method
        mock_manager = mock.MagicMock()
        mock_manager_cls.return_value = mock_manager
        mock_manager.list_datasets.return_value = [sample_dataset]

        result = cli_runner.invoke(golden_dataset.cli.app, ["--datasets-dir", str(test_settings.datasets_dir), "list"])

        assert result.exit_code == 0
        assert "Golden Datasets" in result.stdout
        assert sample_dataset.name in result.stdout

        # Verify correct parameters were passed
        mock_manager_cls.assert_called_once()
        mock_manager.list_datasets.assert_called_once()


def test_list_datasets_empty(cli_runner, test_settings):
    """Test listing datasets when none exist."""
    with mock.patch.object(golden_dataset.cli, "GoldenManager") as mock_manager_cls:
        # Mock manager instance and list_datasets method
        mock_manager = mock.MagicMock()
        mock_manager_cls.return_value = mock_manager
        mock_manager.list_datasets.return_value = []

        result = cli_runner.invoke(golden_dataset.cli.app, ["--datasets-dir", str(test_settings.datasets_dir), "list"])

        assert result.exit_code == 0
        assert "No datasets found" in result.stdout


def test_list_datasets_error(cli_runner, test_settings):
    """Test listing datasets with an error."""
    with mock.patch.object(golden_dataset.cli, "GoldenManager") as mock_manager_cls:
        # Mock manager instance and list_datasets method to raise an error
        mock_manager = mock.MagicMock()
        mock_manager_cls.return_value = mock_manager
        mock_manager.list_datasets.side_effect = GoldenError("Test error")

        result = cli_runner.invoke(golden_dataset.cli.app, ["--datasets-dir", str(test_settings.datasets_dir), "list"])

        assert result.exit_code == 1


# Test show_dataset command
def test_show_dataset_success(cli_runner, test_settings, sample_dataset):
    """Test showing a dataset successfully."""
    with mock.patch.object(golden_dataset.cli, "GoldenManager") as mock_manager_cls:
        # Mock manager instance and open_dataset method
        mock_manager = mock.MagicMock()
        mock_manager_cls.return_value = mock_manager
        mock_manager.open_dataset.return_value = sample_dataset

        result = cli_runner.invoke(
            golden_dataset.cli.app, ["--datasets-dir", str(test_settings.datasets_dir), "show", sample_dataset.name]
        )

        assert result.exit_code == 0
        assert "Golden Dataset" in result.stdout
        assert "users" in result.stdout
        assert "Tables" in result.stdout

        # Verify correct parameters were passed
        mock_manager_cls.assert_called_once()
        mock_manager.open_dataset.assert_called_once_with(sample_dataset.name)


def test_show_dataset_not_found(cli_runner, test_settings):
    """Test showing a dataset that doesn't exist."""
    with mock.patch.object(golden_dataset.cli, "GoldenManager") as mock_manager_cls:
        # Mock manager instance and open_dataset method to raise an error
        mock_manager = mock.MagicMock()
        mock_manager_cls.return_value = mock_manager
        mock_manager.open_dataset.side_effect = DatasetNotFoundError("nonexistent", Path("/path"))

        result = cli_runner.invoke(
            golden_dataset.cli.app, ["--datasets-dir", str(test_settings.datasets_dir), "show", "nonexistent"]
        )

        assert result.exit_code == 1
        assert "not found" in result.stdout


# Test generate_dataset command
def test_generate_dataset_success(cli_runner, test_settings):
    """Test generating a dataset successfully."""
    with mock.patch.object(golden_dataset.cli, "GoldenManager") as mock_manager_cls:
        # Mock manager instance and generate_dataset/dump_dataset methods
        mock_manager = mock.MagicMock()
        mock_manager_cls.return_value = mock_manager

        # Create a mock dataset as the result
        mock_dataset = mock.MagicMock()
        mock_dataset.name = "sample_generator"
        mock_dataset.get_tables.return_value = {"users": 1}

        mock_manager.generate_dataset.return_value = mock_dataset

        result = cli_runner.invoke(
            golden_dataset.cli.app,
            [
                "--datasets-dir",
                str(test_settings.datasets_dir),
                "--src-dir",
                str(test_settings.src_dir),
                "generate",
                "test_generators.test_generator:sample_generator",
            ],
        )

        assert result.exit_code == 0
        assert "Dataset generated successfully" in result.stdout
        assert "sample_generator" in result.stdout

        # Verify correct parameters were passed
        mock_manager_cls.assert_called_once()
        mock_manager.generate_dataset.assert_called_once_with("test_generators.test_generator:sample_generator")
        mock_manager.dump_dataset.assert_called_once_with(mock_dataset)


def test_generate_dataset_error(cli_runner, test_settings):
    """Test generating a dataset with an error."""
    with mock.patch.object(golden_dataset.cli, "GoldenManager") as mock_manager_cls:
        # Mock manager instance and generate_dataset method to raise an error
        mock_manager = mock.MagicMock()
        mock_manager_cls.return_value = mock_manager
        mock_manager.generate_dataset.side_effect = GoldenError("Test error")

        result = cli_runner.invoke(
            golden_dataset.cli.app,
            ["--datasets-dir", str(test_settings.datasets_dir), "generate", "nonexistent_module:nonexistent_generator"],
        )

        assert result.exit_code == 1
        assert "Test error" in result.stdout


# Test load_dataset command
def test_load_dataset_success(cli_runner, test_settings, sample_dataset, mock_sqlalchemy_components):
    """Test loading a dataset successfully."""
    mocks = mock_sqlalchemy_components

    with (
        mock.patch.object(golden_dataset.cli, "GoldenManager") as mock_manager_cls,
        mock.patch.object(golden_dataset.cli, "recursively_load_datasets") as mock_load,
    ):
        # Mock manager instance
        mock_manager = mock.MagicMock()
        mock_manager_cls.return_value = mock_manager

        # Mock recursive loading to return table counts
        mock_load.return_value = {"users": 2, "posts": 3}

        result = cli_runner.invoke(
            golden_dataset.cli.app,
            [
                "--datasets-dir",
                str(test_settings.datasets_dir),
                "--src-dir",
                str(test_settings.src_dir),
                "load",
                sample_dataset.name,
            ],
        )

        assert result.exit_code == 0
        assert "Dataset test_dataset imported successfully" in result.stdout
        assert "Results" in result.stdout

        # Verify SQLAlchemy components were retrieved
        mocks["engine"].assert_called_once()
        mocks["base"].assert_called_once()
        mocks["session_factory"].assert_called_once()

        # Verify recursive loading was called with correct parameters
        mock_load.assert_called_once_with(
            sample_dataset.name,
            mocks["base"].return_value,
            mocks["session_factory"].return_value.return_value,
            recurse=True,
        )


def test_load_dataset_with_no_depends(cli_runner, test_settings, sample_dataset, mock_sqlalchemy_components):
    """Test loading a dataset without dependencies."""
    mocks = mock_sqlalchemy_components

    with (
        mock.patch.object(golden_dataset.cli, "GoldenManager") as mock_manager_cls,
        mock.patch.object(golden_dataset.cli, "recursively_load_datasets") as mock_load,
    ):
        # Mock manager instance
        mock_manager = mock.MagicMock()
        mock_manager_cls.return_value = mock_manager

        # Mock recursive loading to return table counts
        mock_load.return_value = {"users": 2, "posts": 3}

        result = cli_runner.invoke(
            golden_dataset.cli.app,
            [
                "--datasets-dir",
                str(test_settings.datasets_dir),
                "--src-dir",
                str(test_settings.src_dir),
                "load",
                sample_dataset.name,
                "--no-depends",
            ],
        )

        assert result.exit_code == 0

        # Verify recursive loading was called with recurse=False
        mock_load.assert_called_once_with(
            sample_dataset.name,
            mocks["base"].return_value,
            mocks["session_factory"].return_value.return_value,
            recurse=False,
        )


def test_load_dataset_error(cli_runner, test_settings, mock_sqlalchemy_components):
    """Test loading a dataset with an error."""
    with (
        mock.patch.object(golden_dataset.cli, "GoldenManager") as mock_manager_cls,
        mock.patch.object(golden_dataset.cli, "recursively_load_datasets") as mock_load,
    ):
        # Mock manager instance
        mock_manager = mock.MagicMock()
        mock_manager_cls.return_value = mock_manager

        # Mock recursive loading to raise an error
        mock_load.side_effect = GoldenError("Test error")

        result = cli_runner.invoke(
            golden_dataset.cli.app,
            [
                "--datasets-dir",
                str(test_settings.datasets_dir),
                "--src-dir",
                str(test_settings.src_dir),
                "load",
                "dataset_name",
            ],
        )

        assert result.exit_code == 1
        assert "Test error" in result.stdout


def test_load_dataset_missing_engine(cli_runner, test_settings, mock_sqlalchemy_components):
    """Test loading a dataset when engine cannot be found."""
    mock_sqlalchemy_components["engine"].return_value = None

    result = cli_runner.invoke(golden_dataset.cli.app, ["load", "dataset_name"])

    assert result.exit_code == 0
    assert "Could not find engine" in result.stdout


def test_load_dataset_missing_base(cli_runner, test_settings, mock_sqlalchemy_components):
    """Test loading a dataset when base cannot be found."""
    mock_sqlalchemy_components["engine"].return_value = mock.MagicMock()
    mock_sqlalchemy_components["base"].return_value = None

    result = cli_runner.invoke(
        golden_dataset.cli.app,
        [
            "--datasets-dir",
            str(test_settings.datasets_dir),
            "--src-dir",
            str(test_settings.src_dir),
            "load",
            "dataset_name",
        ],
    )

    assert result.exit_code == 0
    assert "Could not find Base Base" in result.stdout


def test_load_dataset_missing_session(cli_runner, test_settings, mock_sqlalchemy_components):
    """Test loading a dataset when session factory cannot be found."""
    mock_sqlalchemy_components["engine"].return_value = mock.MagicMock()
    mock_sqlalchemy_components["base"].return_value = mock.MagicMock()
    mock_sqlalchemy_components["session_factory"].return_value = None

    result = cli_runner.invoke(golden_dataset.cli.app, ["load", "dataset_name"])

    assert result.exit_code == 0
    assert "Could not find Session" in result.stdout


# Test unload_dataset command
def test_unload_dataset_success(cli_runner, test_settings, sample_dataset, mock_sqlalchemy_components):
    """Test unloading a dataset successfully."""
    mocks = mock_sqlalchemy_components

    with mock.patch.object(golden_dataset.cli, "GoldenManager") as mock_manager_cls:
        # Mock manager instance and load_dataset method
        mock_manager = mock.MagicMock()
        mock_manager_cls.return_value = mock_manager

        # Mock dataset with remove_from_session method
        mock_dataset = mock.MagicMock()
        mock_dataset.remove_from_session.return_value = {"users": 2, "posts": 3}
        mock_manager.load_dataset.return_value = mock_dataset

        result = cli_runner.invoke(
            golden_dataset.cli.app,
            [
                "--datasets-dir",
                str(test_settings.datasets_dir),
                "--src-dir",
                str(test_settings.src_dir),
                "unload",
                sample_dataset.name,
            ],
        )

        assert result.exit_code == 0
        assert f"Dataset {sample_dataset.name} removed successfully" in result.stdout
        assert "Results" in result.stdout

        # Verify SQLAlchemy components were retrieved
        mocks["engine"].assert_called_once()
        mocks["base"].assert_called_once()
        mocks["session_factory"].assert_called_once()

        # Verify dataset was loaded and removed
        mock_manager.load_dataset.assert_called_once_with(sample_dataset.name)
        mock_dataset.remove_from_session.assert_called_once_with(
            mocks["base"].return_value, mocks["session_factory"].return_value.return_value
        )


def test_unload_dataset_error(cli_runner, test_settings, mock_sqlalchemy_components):
    """Test unloading a dataset with an error."""
    with mock.patch.object(golden_dataset.cli, "GoldenManager") as mock_manager_cls:
        # Mock manager instance and load_dataset method to raise an error
        mock_manager = mock.MagicMock()
        mock_manager_cls.return_value = mock_manager
        mock_manager.load_dataset.side_effect = GoldenError("Test error")

        result = cli_runner.invoke(
            golden_dataset.cli.app,
            [
                "--datasets-dir",
                str(test_settings.datasets_dir),
                "--src-dir",
                str(test_settings.src_dir),
                "unload",
                "dataset_name",
            ],
        )

        assert result.exit_code == 1
        assert "Test error" in result.stdout


def test_unload_dataset_session_error(cli_runner, test_settings, sample_dataset, mock_sqlalchemy_components):
    """Test unloading a dataset with a session error."""
    mocks = mock_sqlalchemy_components

    with mock.patch.object(golden_dataset.cli, "GoldenManager") as mock_manager_cls:
        # Mock manager instance and load_dataset method
        mock_manager = mock.MagicMock()
        mock_manager_cls.return_value = mock_manager

        # Mock dataset with remove_from_session that raises an error
        mock_dataset = mock.MagicMock()
        mock_dataset.remove_from_session.side_effect = Exception("Session error")
        mock_manager.load_dataset.return_value = mock_dataset

        # Mock session to raise error on commit
        mocks["session"].__enter__.return_value = mocks["session"]
        mocks["session"].commit.side_effect = Exception("Commit error")

        result = cli_runner.invoke(
            golden_dataset.cli.app,
            [
                "--datasets-dir",
                str(test_settings.datasets_dir),
                "--src-dir",
                str(test_settings.src_dir),
                "unload",
                sample_dataset.name,
            ],
        )

        assert result.exit_code == 1
        assert "Error unloading dataset" in result.stdout

        # Verify rollback was called
        mocks["session"].rollback.assert_called_once()


# Test recursively_load_datasets function
def test_recursively_load_datasets(test_settings, sample_dataset, dependency_dataset, complex_dataset):
    """Test recursively loading datasets."""
    # Mock dependencies
    base_mock = mock.MagicMock()
    session_mock = mock.MagicMock()

    with mock.patch.object(golden_dataset.cli, "GoldenManager") as mock_manager_cls:
        # Mock manager to return our datasets
        mock_manager = mock.MagicMock()
        mock_manager_cls.return_value = mock_manager

        def mock_load_dataset(name):
            if name == sample_dataset.name:
                mock_dataset = mock.MagicMock()
                mock_dataset.dependencies = []
                mock_dataset.add_to_session.return_value = {"users": 2, "posts": 3}
                return mock_dataset
            elif name == dependency_dataset.name:
                mock_dataset = mock.MagicMock()
                mock_dataset.dependencies = []
                mock_dataset.add_to_session.return_value = {"categories": 1}
                return mock_dataset
            elif name == complex_dataset.name:
                mock_dataset = mock.MagicMock()
                mock_dataset.dependencies = [dependency_dataset.name]
                mock_dataset.add_to_session.return_value = {"articles": 2}
                return mock_dataset
            else:
                return None

        mock_manager.load_dataset.side_effect = mock_load_dataset

        # Test loading a dataset with no dependencies
        result = golden_dataset.cli.recursively_load_datasets(sample_dataset.name, base_mock, session_mock)
        assert result == {"users": 2, "posts": 3}

        # Test loading a dataset with dependencies
        result = golden_dataset.cli.recursively_load_datasets(complex_dataset.name, base_mock, session_mock)
        assert result == {"categories": 1, "articles": 2}

        # Test with circular dependencies
        mock_circular = mock.MagicMock()
        mock_circular.dependencies = ["circular"]
        mock_circular.add_to_session.return_value = {"circular": 1}

        def mock_load_circular(name):
            return mock_circular if name == "circular" else None

        mock_manager.load_dataset.side_effect = mock_load_circular

        result = golden_dataset.cli.recursively_load_datasets("circular", base_mock, session_mock)
        assert result == {"circular": 1}  # Should only be loaded once despite circularity
