import pytest
from sqlalchemy.orm import declarative_base

Base = declarative_base()


@pytest.mark.skip
class TestModel:
    pass
