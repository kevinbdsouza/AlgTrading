import importlib

import pytest

@pytest.mark.parametrize("module_name", ["toraniko", "numpy", "polars"])
def test_required_imports(module_name):
    """Ensure toraniko and its dependencies import correctly."""
    assert importlib.import_module(module_name)


def test_macd_strategy_creation():
    import sys
    sys.path.append('code')
    from strategy_base import StrategyFactory

    strategy = StrategyFactory.create_strategy('macd')
    assert strategy is not None
