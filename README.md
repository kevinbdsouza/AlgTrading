# AlgTrading

A collection of tools for researching and running intraday trading strategies. It includes a modular trading system, backtester, risk management utilities and several example strategies.

## Features

- **Multiple strategies**: Moving average crossover, RSI mean reversion, Bollinger Bands and MACD (new).
- **Backtesting engine** with metrics such as total return, Sharpe ratio, drawdown and more.
- **Risk management**: position sizing, stop loss/take profit and daily loss limits.
- **Live trading support** via Interactive Brokers API with optional paper trading.
- **Factor model utilities** using [toraniko](https://github.com/0xfdf/toraniko) for quantitative analysis.

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Quick Start

Run a simple moving average cross backtest:

```python
from config import TradingConfig
from intraday_strategy import IntradayTradingSystem

system = IntradayTradingSystem(TradingConfig())
system.set_strategy('ma_cross')
results = system.run_backtest('SPY', '20230101', '20231231')
print(results.metrics)
```

For live paper trading (requires a running TWS/Gateway instance):

```python
system.connect_to_ibkr()
system.set_strategy('ma_cross')
system.run_live_trading('SPY', duration_minutes=60)
print(system.get_performance_metrics())
```

## Backtesting Metrics

The backtester records a suite of metrics including:

- Profit and loss (PnL)
- Win rate and profit factor
- Sharpe and Sortino ratios
- Maximum drawdown and volatility
- Trade expectancy and Calmar ratio

Results are returned as a dictionary for further analysis or comparison.

## Live Performance Tracking

During live (paper) trading all executed trades are logged and the system keeps running totals of realised and unrealised PnL. Performance metrics can be retrieved via `get_performance_metrics()` and may be used to automatically adjust strategy parameters.

## Examples and Tests

Additional examples can be found under `code/example_usage.py` and `docs/examples/`. To verify the installation run:

```bash
pytest -q
```

## Folder Structure

```
AlgTrading/
├── code/              # Trading system modules
├── docs/              # Books and example notebooks
├── tests/             # Basic import tests
└── requirements.txt
```

This project is for research purposes. Use at your own risk when trading live.
