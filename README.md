# golden-dataset

A Python library for creating, managing, and sharing golden datasets.

## Overview

Golden datasets are curated samples of data used for testing, development, and demonstration purposes. This library provides a simple way to create, export, and load golden datasets in a consistent format.

## Features

- Define dataset generators with a simple `@golden` decorator
- Track dependencies between datasets
- Support for both SQLModel and SQLAlchemy models
- Generate datasets programmatically or via CLI
- Export datasets to JSON files
- Import datasets into any SQLAlchemy or SQLModel session

## Installation

### From PyPI

```bash
# Basic installation
pip install golden-dataset

# With SQLModel support
pip install golden-dataset[sqlmodel]

# With SQLAlchemy support
pip install golden-dataset[sqlalchemy]

# With CLI support
pip install golden-dataset[cli]

# With async support
pip install golden-dataset[async]

# With all optional dependencies
pip install golden-dataset[all]
```

### From source

```bash
# Using uv (recommended)
uv pip install .

# With specific optional dependencies
uv pip install ".[sqlmodel]"
uv pip install ".[sqlalchemy]"
uv pip install ".[all]"

# Or using pip
pip install .
pip install ".[sqlmodel]"
```

## Usage

### Creating a golden dataset

```python
# base_dataset.py
from golden_dataset import golden
from myapp.models import EventType

@golden
def base(session):
    event_type1 = EventType.model_validate({"id": 0, "name": "Purchase"})
    event_type2 = EventType.model_validate({"id": 1, "name": "Landing page"})
    event_type3 = EventType.model_validate({"id": 2, "name": "Sign-Up"})
    event_type4 = EventType.model_validate({"id": 3, "name": "Lead"})
    event_type5 = EventType.model_validate({"id": 99, "name": "Other"})
    session.add(event_type1)
    session.add(event_type2)
    session.add(event_type3)
    session.add(event_type4)
    session.add(event_type5)
```

### Creating a dataset with dependencies

```python
# bloom_organics_dataset.py
from golden_dataset import golden
from myapp.models import Brand, Font, Brandkit

@golden(dependencies=["base"])
def bloom_organics(session):
    brand = Brand.model_validate({
        "code": "bloom_organics",
        "name": "Bloom Organics",
        "description": "Clean, plant-based skincare formulated with certified organic ingredients.",
        # ... more fields
    })
    session.add(brand)
    session.refresh(brand)
    
    font1 = Font.model_validate({
        "brand_id": brand.id,
        "font_family": "Cormorant Garamond",
        "font_weight": 500,
        "font_style": "normal",
        "source": "https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500&display=swap"
    })
    session.add(font1)
    # ... add more objects
```

### Generating datasets programmatically

```python
from golden_dataset import generate_dataset

# Generate a dataset and its dependencies
dataset = generate_dataset("bloom_organics_dataset:bloom_organics", "output_dir")
```

### Loading datasets into a database

```python
from golden_dataset import load_dataset
from sqlmodel import Session, create_engine

# Create a database engine
engine = create_engine("sqlite:///app.db")

# Define a session factory
def session_factory():
    return Session(engine)

# Load the dataset
load_dataset("output_dir/bloom_organics", session_factory)
```

### Using the CLI

```bash
# Generate a dataset
golden-dataset generate "bloom_organics_dataset:bloom_organics" --output-dir datasets

# List available datasets
golden-dataset list-datasets --datasets-dir datasets

# Import a dataset (demo only)
golden-dataset import-dataset bloom_organics --datasets-dir datasets
```

## Development

### Development setup

```bash
# Using uv (recommended)
uv venv
uv sync --all --dev

# Or using pip (traditional approach)
pip install -e ".[dev,all]"
```

### Code formatting and linting

```bash
# Format code with Ruff
ruff format src tests

# Run Ruff linter
ruff check src tests

# Auto-fix issues where possible
ruff check --fix src tests
```

### Running tests

```bash
pytest
```

### Running with code coverage

```bash
pytest --cov=golden_dataset
```

### Type checking

```bash
mypy src
```

## License

MIT
Dataset for testing" --version "1.0.0"
```

Add a record to a dataset:

```bash
golden-dataset add-record 1 content.json --metadata-file metadata.json
```

List all datasets:

```bash
golden-dataset list-datasets
```

Export a dataset to JSON:

```bash
golden-dataset export-dataset 1 dataset_export.json
```

### Python API

```python
from golden_dataset.database import Database
from golden_dataset.models import Dataset, Record

# Initialize the database
db = Database("sqlite:///golden.db")
db.create_tables()

# Create a new dataset
dataset = Dataset(
    name="My Test Dataset",
    description="A dataset for testing",
    version="1.0.0",
    metadata={"source": "manual", "category": "test"}
)
db.add(dataset)

# Add a record to the dataset
record = Record(
    content={"key": "value", "test": True, "count": 42},
    metadata={"quality": "high"},
    dataset_id=dataset.id
)
db.add(record)

# Query datasets
test_datasets = db.query(Dataset, name="My Test Dataset")
for dataset in test_datasets:
    print(f"Dataset: {dataset.name} (version {dataset.version})")
    print(f"Records: {len(dataset.records)}")
```

## Publishing to PyPI or a Private Registry

### Setting up for publishing

1. Create a PyPI account if publishing to the public PyPI
2. Generate an API token for the registry
3. Set up GitHub Actions for automated publishing (see `.github/workflows/publish.yml`)

### Publishing with uv

Using [uv](https://github.com/astral-sh/uv) for publishing:

```bash
# Build the package
uv build

# Publish to PyPI
uv publish

# Publish to a private registry
uv publish --repository-url https://your-private-registry/simple
```

### Publishing with twine

Alternatively, you can use twine:

```bash
# Build the package
python -m build

# Publish to PyPI
twine upload dist/*

# Publish to a private registry
twine upload --repository-url https://your-private-registry/simple dist/*
```

## Development

### Running tests

```bash
pytest
```

### Running with code coverage

```bash
pytest --cov=golden_dataset
```

### Code formatting and linting

```bash
# Format code with Ruff
ruff format src tests

# Run Ruff linter
ruff check src tests

# Auto-fix issues where possible
ruff check --fix src tests
```

### Type checking

```bash
mypy src
```

## License

MIT