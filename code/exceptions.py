"""
Custom exceptions for the trading system.
"""


class TradingError(Exception):
    """Base exception for trading-related errors."""
    pass


class DataError(TradingError):
    """Exception raised for data-related errors."""
    pass


class ConnectionError(TradingError):
    """Exception raised for connection-related errors."""
    pass


class OrderError(TradingError):
    """Exception raised for order-related errors."""
    pass


class StrategyError(TradingError):
    """Exception raised for strategy-related errors."""
    pass


class RiskManagementError(TradingError):
    """Exception raised for risk management violations."""
    pass


class ConfigurationError(TradingError):
    """Exception raised for configuration-related errors."""
    pass


class BacktestError(TradingError):
    """Exception raised for backtesting-related errors."""
    pass