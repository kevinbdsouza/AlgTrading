"""
Base strategy class and signal generation.
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from enum import Enum

from logger import get_logger
from exceptions import StrategyError
from config import TradingConfig


class SignalType(Enum):
    """Signal types for trading decisions."""
    BUY = 1
    SELL = -1
    HOLD = 0


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""
    
    def __init__(self, config: Optional[TradingConfig] = None):
        self.config = config or TradingConfig()
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.name = self.__class__.__name__
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on the data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals added
        """
        pass
    
    @abstractmethod
    def get_strategy_params(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        pass
    
    def validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data."""
        if data.empty:
            raise StrategyError("Empty DataFrame provided")
        
        required_columns = ['Close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise StrategyError(f"Missing required columns: {missing_columns}")
        
        if data['Close'].isna().all():
            raise StrategyError("All Close prices are NaN")


class MovingAverageCrossStrategy(BaseStrategy):
    """Moving Average Crossover Strategy."""
    
    def __init__(self, config: Optional[TradingConfig] = None):
        super().__init__(config)
        self.short_period = self.config.short_ma_period
        self.long_period = self.config.long_ma_period
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on moving average crossover."""
        self.validate_data(data)
        
        df = data.copy()
        
        # Calculate moving averages
        df['short_ma'] = df['Close'].rolling(window=self.short_period, min_periods=1).mean()
        df['long_ma'] = df['Close'].rolling(window=self.long_period, min_periods=1).mean()
        
        # Initialize signal column
        df['signal'] = SignalType.HOLD.value
        
        # Calculate previous values for crossover detection
        df['prev_short_ma'] = df['short_ma'].shift(1)
        df['prev_long_ma'] = df['long_ma'].shift(1)
        
        # Generate buy signals (short MA crosses above long MA)
        buy_condition = (
            (df['short_ma'] > df['long_ma']) & 
            (df['prev_short_ma'] <= df['prev_long_ma'])
        )
        df.loc[buy_condition, 'signal'] = SignalType.BUY.value
        
        # Generate sell signals (short MA crosses below long MA)
        sell_condition = (
            (df['short_ma'] < df['long_ma']) & 
            (df['prev_short_ma'] >= df['prev_long_ma'])
        )
        df.loc[sell_condition, 'signal'] = SignalType.SELL.value
        
        # Add signal strength (distance between MAs)
        df['signal_strength'] = abs(df['short_ma'] - df['long_ma']) / df['long_ma']
        
        # Clean up temporary columns
        df.drop(['prev_short_ma', 'prev_long_ma'], axis=1, inplace=True)
        
        self.logger.info(f"Generated {(df['signal'] != 0).sum()} signals")
        
        return df
    
    def get_strategy_params(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return {
            'name': self.name,
            'short_period': self.short_period,
            'long_period': self.long_period,
            'type': 'trend_following'
        }


class RSIStrategy(BaseStrategy):
    """RSI-based mean reversion strategy."""
    
    def __init__(self, config: Optional[TradingConfig] = None, rsi_period: int = 14, 
                 oversold_threshold: float = 30, overbought_threshold: float = 70):
        super().__init__(config)
        self.rsi_period = rsi_period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on RSI levels."""
        self.validate_data(data)
        
        df = data.copy()
        
        # Calculate RSI
        df['rsi'] = self._calculate_rsi(df['Close'], self.rsi_period)
        
        # Initialize signal column
        df['signal'] = SignalType.HOLD.value
        
        # Generate buy signals (RSI oversold)
        buy_condition = df['rsi'] < self.oversold_threshold
        df.loc[buy_condition, 'signal'] = SignalType.BUY.value
        
        # Generate sell signals (RSI overbought)
        sell_condition = df['rsi'] > self.overbought_threshold
        df.loc[sell_condition, 'signal'] = SignalType.SELL.value
        
        # Add signal strength (distance from threshold)
        df['signal_strength'] = np.where(
            df['signal'] == SignalType.BUY.value,
            (self.oversold_threshold - df['rsi']) / self.oversold_threshold,
            np.where(
                df['signal'] == SignalType.SELL.value,
                (df['rsi'] - self.overbought_threshold) / (100 - self.overbought_threshold),
                0
            )
        )
        
        self.logger.info(f"Generated {(df['signal'] != 0).sum()} RSI signals")
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def get_strategy_params(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return {
            'name': self.name,
            'rsi_period': self.rsi_period,
            'oversold_threshold': self.oversold_threshold,
            'overbought_threshold': self.overbought_threshold,
            'type': 'mean_reversion'
        }


class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands strategy."""
    
    def __init__(self, config: Optional[TradingConfig] = None, period: int = 20, 
                 std_dev: float = 2.0):
        super().__init__(config)
        self.period = period
        self.std_dev = std_dev
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on Bollinger Bands."""
        self.validate_data(data)
        
        df = data.copy()
        
        # Calculate Bollinger Bands
        sma = df['Close'].rolling(window=self.period).mean()
        std = df['Close'].rolling(window=self.period).std()
        
        df['bb_upper'] = sma + (std * self.std_dev)
        df['bb_lower'] = sma - (std * self.std_dev)
        df['bb_middle'] = sma
        
        # Initialize signal column
        df['signal'] = SignalType.HOLD.value
        
        # Generate buy signals (price touches lower band)
        buy_condition = df['Close'] <= df['bb_lower']
        df.loc[buy_condition, 'signal'] = SignalType.BUY.value
        
        # Generate sell signals (price touches upper band)
        sell_condition = df['Close'] >= df['bb_upper']
        df.loc[sell_condition, 'signal'] = SignalType.SELL.value
        
        # Add signal strength (distance from middle band)
        df['signal_strength'] = abs(df['Close'] - df['bb_middle']) / (df['bb_upper'] - df['bb_lower'])
        
        self.logger.info(f"Generated {(df['signal'] != 0).sum()} Bollinger Bands signals")
        
        return df
    
    def get_strategy_params(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return {
            'name': self.name,
            'period': self.period,
            'std_dev': self.std_dev,
            'type': 'mean_reversion'
        }


class MACDStrategy(BaseStrategy):
    """Moving Average Convergence Divergence strategy."""

    def __init__(self, config: Optional[TradingConfig] = None,
                 fast_period: int = 12, slow_period: int = 26,
                 signal_period: int = 9):
        super().__init__(config)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on MACD crossovers."""
        self.validate_data(data)

        df = data.copy()

        df['ema_fast'] = df['Close'].ewm(span=self.fast_period, adjust=False).mean()
        df['ema_slow'] = df['Close'].ewm(span=self.slow_period, adjust=False).mean()
        df['macd'] = df['ema_fast'] - df['ema_slow']
        df['signal_line'] = df['macd'].ewm(span=self.signal_period, adjust=False).mean()

        df['signal'] = SignalType.HOLD.value

        buy_condition = (
            (df['macd'] > df['signal_line']) &
            (df['macd'].shift(1) <= df['signal_line'].shift(1))
        )
        sell_condition = (
            (df['macd'] < df['signal_line']) &
            (df['macd'].shift(1) >= df['signal_line'].shift(1))
        )

        df.loc[buy_condition, 'signal'] = SignalType.BUY.value
        df.loc[sell_condition, 'signal'] = SignalType.SELL.value

        df['signal_strength'] = abs(df['macd'] - df['signal_line']) / df['Close']

        self.logger.info(f"Generated {(df['signal'] != 0).sum()} MACD signals")

        return df

    def get_strategy_params(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return {
            'name': self.name,
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'signal_period': self.signal_period,
            'type': 'trend_following'
        }


class StrategyFactory:
    """Factory for creating strategy instances."""
    
    _strategies = {
        'ma_cross': MovingAverageCrossStrategy,
        'rsi': RSIStrategy,
        'bollinger': BollingerBandsStrategy,
        'macd': MACDStrategy
    }
    
    @classmethod
    def create_strategy(cls, strategy_name: str, config: Optional[TradingConfig] = None, 
                       **kwargs) -> BaseStrategy:
        """Create a strategy instance."""
        if strategy_name not in cls._strategies:
            available = ', '.join(cls._strategies.keys())
            raise StrategyError(f"Unknown strategy: {strategy_name}. Available: {available}")
        
        strategy_class = cls._strategies[strategy_name]
        return strategy_class(config=config, **kwargs)
    
    @classmethod
    def list_strategies(cls) -> list:
        """List available strategies."""
        return list(cls._strategies.keys())