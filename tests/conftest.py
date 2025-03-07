"""
Pytest configuration for golden-dataset tests.
"""

import asyncio
from collections.abc import Generator

import pytest

from golden_dataset import GoldenSession


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    asyncio.set_event_loop(loop)  # Set this as the current event loop
    yield loop
    loop.close()


@pytest.fixture
def empty_session():
    """Create an empty GoldenSession."""
    return GoldenSession()
