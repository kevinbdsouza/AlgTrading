"""
Configuration management for the trading system.
"""
from dataclasses import dataclass
from typing import Dict, Any
import os


@dataclass
class TradingConfig:
    """Configuration for trading parameters."""
    
    # Strategy parameters
    short_ma_period: int = 10
    long_ma_period: int = 30
    quantity: int = 1
    slippage_per_trade: float = 0.01
    
    # Risk management
    max_position_size: int = 100
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit
    max_daily_loss: float = 1000.0
    
    # Backtesting
    initial_capital: float = 100000.0
    shares_per_trade: int = 10
    
    # Data settings
    default_symbol: str = "SPY"
    default_timeframe: str = "5 mins"
    
    # IBKR connection
    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 7497
    ibkr_client_id: int = 131
    connection_timeout: int = 10
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "trading.log"
    
    @classmethod
    def from_env(cls) -> 'TradingConfig':
        """Create config from environment variables."""
        return cls(
            short_ma_period=int(os.getenv('SHORT_MA_PERIOD', 10)),
            long_ma_period=int(os.getenv('LONG_MA_PERIOD', 30)),
            quantity=int(os.getenv('QUANTITY', 1)),
            slippage_per_trade=float(os.getenv('SLIPPAGE_PER_TRADE', 0.01)),
            max_position_size=int(os.getenv('MAX_POSITION_SIZE', 100)),
            stop_loss_pct=float(os.getenv('STOP_LOSS_PCT', 0.02)),
            take_profit_pct=float(os.getenv('TAKE_PROFIT_PCT', 0.04)),
            max_daily_loss=float(os.getenv('MAX_DAILY_LOSS', 1000.0)),
            initial_capital=float(os.getenv('INITIAL_CAPITAL', 100000.0)),
            shares_per_trade=int(os.getenv('SHARES_PER_TRADE', 10)),
            default_symbol=os.getenv('DEFAULT_SYMBOL', 'SPY'),
            default_timeframe=os.getenv('DEFAULT_TIMEFRAME', '5 mins'),
            ibkr_host=os.getenv('IBKR_HOST', '127.0.0.1'),
            ibkr_port=int(os.getenv('IBKR_PORT', 7497)),
            ibkr_client_id=int(os.getenv('IBKR_CLIENT_ID', 131)),
            connection_timeout=int(os.getenv('CONNECTION_TIMEOUT', 10)),
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            log_file=os.getenv('LOG_FILE', 'trading.log')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'short_ma_period': self.short_ma_period,
            'long_ma_period': self.long_ma_period,
            'quantity': self.quantity,
            'slippage_per_trade': self.slippage_per_trade,
            'max_position_size': self.max_position_size,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'max_daily_loss': self.max_daily_loss,
            'initial_capital': self.initial_capital,
            'shares_per_trade': self.shares_per_trade,
            'default_symbol': self.default_symbol,
            'default_timeframe': self.default_timeframe,
            'ibkr_host': self.ibkr_host,
            'ibkr_port': self.ibkr_port,
            'ibkr_client_id': self.ibkr_client_id,
            'connection_timeout': self.connection_timeout,
            'log_level': self.log_level,
            'log_file': self.log_file
        }