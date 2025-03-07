import pytest


@pytest.mark.skip
def test_function(param1: str, param2: int = 10) -> str:
    return f"{param1}: {param2}"


@pytest.mark.skip
class TestClass:
    @pytest.mark.skip
    def test_method(self, param1: str) -> str:
        return param1


test_not_callable = "not a function"
