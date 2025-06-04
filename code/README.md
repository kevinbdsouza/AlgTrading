# Improved Intraday Trading System

A comprehensive, production-ready algorithmic trading system with multiple strategies, risk management, backtesting, and live trading capabilities.

## Features

### 🚀 Core Features
- **Multiple Trading Strategies**: Moving Average Crossover, RSI, Bollinger Bands
- **Risk Management**: Position sizing, stop-loss, take-profit, daily loss limits
- **Backtesting Engine**: Comprehensive performance metrics and analysis
- **Live Trading**: Real-time trading with IBKR integration
- **Data Management**: Efficient data fetching, validation, and caching
- **Logging & Monitoring**: Structured logging and performance tracking

### 📊 Strategies Available
1. **Moving Average Crossover** (`ma_cross`): Trend-following strategy
2. **RSI Strategy** (`rsi`): Mean-reversion based on RSI levels
3. **Bollinger Bands** (`bollinger`): Mean-reversion using Bollinger Bands

### 🛡️ Risk Management
- Position size limits
- Stop-loss and take-profit orders
- Daily loss limits
- Real-time risk monitoring
- Portfolio exposure tracking

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables (optional):
```bash
export SHORT_MA_PERIOD=10
export LONG_MA_PERIOD=30
export INITIAL_CAPITAL=100000
export LOG_LEVEL=INFO
```

## Quick Start

### Backtesting
```python
from config import TradingConfig
from intraday_strategy import IntradayTradingSystem

# Create trading system
config = TradingConfig()
trading_system = IntradayTradingSystem(config=config)

# Set strategy
trading_system.set_strategy('ma_cross')

# Run backtest
results = trading_system.run_backtest(
    symbol='SPY',
    start_date='20230101',
    end_date='20231231'
)

print(f"Total Return: {results.metrics['total_return_pct']:.2f}%")
print(f"Sharpe Ratio: {results.metrics['sharpe_ratio']:.2f}")
```

### Live Trading
```python
# Connect to IBKR and run live trading
trading_system.connect_to_ibkr()
trading_system.set_strategy('ma_cross')
trading_system.run_live_trading('SPY', duration_minutes=60)
```

### Strategy Comparison
```python
strategy_configs = [
    {'name': 'ma_cross'},
    {'name': 'rsi', 'oversold_threshold': 25},
    {'name': 'bollinger', 'period': 20}
]

results = trading_system.compare_strategies(
    strategy_configs,
    'SPY',
    '20230101',
    '20231231'
)
```

## Command Line Usage

The system can be run from the command line:

```bash
# Run backtest
python intraday_strategy.py --mode backtest --symbol SPY --strategy ma_cross

# Run live trading
python intraday_strategy.py --mode live --symbol AAPL --duration 120

# Compare strategies
python intraday_strategy.py --mode compare --symbol SPY
```

## Configuration

### Environment Variables
- `SHORT_MA_PERIOD`: Short moving average period (default: 10)
- `LONG_MA_PERIOD`: Long moving average period (default: 30)
- `INITIAL_CAPITAL`: Initial capital for backtesting (default: 100000)
- `MAX_POSITION_SIZE`: Maximum position size (default: 100)
- `STOP_LOSS_PCT`: Stop loss percentage (default: 0.02)
- `TAKE_PROFIT_PCT`: Take profit percentage (default: 0.04)
- `LOG_LEVEL`: Logging level (default: INFO)

### Configuration File
```python
from config import TradingConfig

config = TradingConfig(
    short_ma_period=15,
    long_ma_period=35,
    initial_capital=50000,
    stop_loss_pct=0.015
)
```

## Architecture

### Core Components

1. **IntradayTradingSystem**: Main orchestrator class
2. **DataManager**: Handles data fetching and validation
3. **StrategyBase**: Abstract base for trading strategies
4. **RiskManager**: Manages risk and position sizing
5. **Backtester**: Backtesting engine with performance metrics
6. **Config**: Configuration management
7. **Logger**: Structured logging system

### Class Hierarchy
```
IntradayTradingSystem
├── DataManager
├── RiskManager
├── Backtester
└── BaseStrategy
    ├── MovingAverageCrossStrategy
    ├── RSIStrategy
    └── BollingerBandsStrategy
```

## Performance Metrics

The system calculates comprehensive performance metrics:

- **Returns**: Total return, annualized return
- **Risk Metrics**: Sharpe ratio, maximum drawdown, volatility
- **Trade Metrics**: Win rate, profit factor, average win/loss
- **Risk Management**: Position exposure, daily P&L tracking

## Error Handling

Robust error handling with custom exceptions:
- `TradingError`: Base trading exception
- `DataError`: Data-related errors
- `ConnectionError`: IBKR connection issues
- `OrderError`: Order execution problems
- `RiskManagementError`: Risk limit violations

## Logging

Structured logging with multiple levels:
- Console output for real-time monitoring
- File logging for historical analysis
- Configurable log levels (DEBUG, INFO, WARNING, ERROR)

## Testing

Run the system with dummy data for testing:
```python
# The system automatically generates dummy data when IBKR is not available
trading_system = IntradayTradingSystem()
trading_system.set_strategy('ma_cross')
results = trading_system.run_backtest('TEST', '20230101', '20231231')
```

## IBKR Integration

### Prerequisites
1. Install IBKR TWS or Gateway
2. Enable API connections in TWS/Gateway
3. Configure connection parameters in config

### Connection Settings
```python
config = TradingConfig(
    ibkr_host='127.0.0.1',
    ibkr_port=7497,  # 7497 for TWS, 4001 for Gateway
    ibkr_client_id=131
)
```

## Best Practices

1. **Risk Management**: Always set appropriate position sizes and stop losses
2. **Backtesting**: Test strategies thoroughly before live trading
3. **Monitoring**: Monitor logs and performance metrics regularly
4. **Configuration**: Use environment variables for sensitive settings
5. **Error Handling**: Implement proper error handling in custom strategies

## Extending the System

### Adding New Strategies
```python
from strategy_base import BaseStrategy, SignalType

class MyCustomStrategy(BaseStrategy):
    def generate_signals(self, data):
        # Implement your strategy logic
        df = data.copy()
        df['signal'] = SignalType.HOLD.value
        # Add your signal generation logic
        return df
    
    def get_strategy_params(self):
        return {'name': 'my_custom_strategy'}

# Register the strategy
StrategyFactory._strategies['my_custom'] = MyCustomStrategy
```

### Custom Risk Rules
```python
class CustomRiskManager(RiskManager):
    def validate_order(self, symbol, quantity, price, action):
        # Add custom risk validation logic
        is_valid, reason = super().validate_order(symbol, quantity, price, action)
        
        # Add your custom checks
        if my_custom_check():
            return False, "Custom risk rule violated"
        
        return is_valid, reason
```

## Troubleshooting

### Common Issues

1. **IBKR Connection Failed**
   - Check TWS/Gateway is running
   - Verify API is enabled
   - Check port and client ID

2. **No Data Received**
   - Verify market hours
   - Check symbol validity
   - Review data permissions

3. **Strategy Not Working**
   - Check strategy parameters
   - Verify data quality
   - Review signal generation logic

### Debug Mode
```bash
export LOG_LEVEL=DEBUG
python intraday_strategy.py --mode backtest
```

## License

This project is for educational and research purposes. Use at your own risk in live trading.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review logs for error details
3. Create an issue with detailed information