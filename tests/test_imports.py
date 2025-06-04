import importlib

import pytest

@pytest.mark.parametrize("module_name", ["toraniko", "numpy", "polars"])
def test_required_imports(module_name):
    """Ensure toraniko and its dependencies import correctly."""
    assert importlib.import_module(module_name)
