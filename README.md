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

# With CLI support
pip install golden-dataset[cli]
```

### From source

```bash
# Using uv (recommended)
uv pip install .

# With specific optional dependencies
uv pip install ".[cli]"

# Or using pip
pip install .
pip install ".[cli]"
```

## Usage

### Creating a golden dataset

```python
# base_dataset.py
from golden_dataset import golden
from myapp.models import EventType

@golden
def base(session):
    event_type1 = EventType(id=0, name="Purchase")
    event_type2 = EventType(id=1, name="Landing page")
    event_type3 = EventType(id=2, name="Sign-Up")
    event_type4 = EventType(id=3, name="Lead")
    event_type5 = EventType(id=99, name="Other")
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

# without a decorator:
# def bloom_organics(session, base):

@golden(title="Bloom Organics", dependencies=["base"])
def bloom_organics(session):
    brand = Brand(
        code="bloom_organics",
        name="Bloom Organics",
        description="Clean, plant-based skincare formulated with certified organic ingredients.",
        # ... more fields
    )
    session.add(brand)
    session.refresh(brand)
    
    font1 = Font(
        brand_id=brand.id,
        font_family="Cormorant Garamond",
        font_weight=500,
        font_style="normal",
        source="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500&display=swap"
    )
    session.add(font1)
    # ... add more objects
```

### Generating datasets programmatically

```python
from golden_dataset import GoldenManager

# Generate a dataset and its dependencies
manager = GoldenManager()
dataset = manager.generate_dataset("bloom_organics_dataset:bloom_organics")
manager.dump_dataset(dataset)
```

### Loading datasets into a database

```python
from golden_dataset import GoldenManager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base

# Create a database engine
engine = create_engine("sqlite:///app.db")

session_factory = sessionmaker(bind=engine)

# Load the dataset
with session_factory() as session:
    manager = GoldenManager()
    manager.load_dataset("bloom_organics", Base, session, recurse=False)
```

### Using the CLI

```bash
# List available datasets
golden-dataset list

# Generate a dataset
golden-dataset generate "bloom_organics_dataset:bloom_organics"

# Show a dataset details
golden-dataset show bloom_organics

# Import a dataset
golden-dataset load bloom_organics

# Delete a dataset from the database
golden-dataset unload bloom_organics
```

## Development

### Development setup

```bash
# Using uv (recommended)
uv venv
uv sync --all --dev

# Or using pip (traditional approach)
pip install -e ".[dev]"
```

### Code formatting and linting

```bash
# Format code with Ruff
uv run ruff format src tests

# Run Ruff linter
uv run ruff check src tests

# Auto-fix issues where possible
uv run ruff check --fix src tests
```

### Running tests

```bash
uv run pytest
```

### Running with code coverage

```bash
uv run pytest --cov=golden_dataset
```

### Type checking

```bash
uv run mypy src
```

## License

MIT