"""
Pytest configuration for golden-dataset tests.
"""

import pytest

from golden_dataset import GoldenSession


@pytest.fixture
def empty_session():
    """Create an empty GoldenSession."""
    return GoldenSession()
