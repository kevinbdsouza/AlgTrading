"""
Data management for the trading system.
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import time

from logger import get_logger
from exceptions import DataError
from config import TradingConfig


class DataManager:
    """Manages data fetching, validation, and processing."""
    
    def __init__(self, ib_client=None, config: Optional[TradingConfig] = None):
        self.ib_client = ib_client
        self.config = config or TradingConfig()
        self.logger = get_logger(f"{__name__}.DataManager")
        self._data_cache: Dict[str, pd.DataFrame] = {}
        
    def fetch_historical_data(
        self,
        symbol: str,
        duration_str: str = "1 D",
        bar_size_setting: str = "5 mins",
        end_date_time_str: str = "",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical data from IBKR or return cached data.
        
        Args:
            symbol: Trading symbol
            duration_str: Duration string for IBKR API
            bar_size_setting: Bar size setting
            end_date_time_str: End date time string
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with historical data
            
        Raises:
            DataError: If data fetching fails
        """
        cache_key = f"{symbol}_{duration_str}_{bar_size_setting}_{end_date_time_str}"
        
        if use_cache and cache_key in self._data_cache:
            self.logger.info(f"Using cached data for {symbol}")
            return self._data_cache[cache_key]
        
        if not self.ib_client:
            self.logger.warning("IB client not available, generating dummy data")
            return self._generate_dummy_data(symbol, duration_str, bar_size_setting)
        
        try:
            df = self._fetch_from_ibkr(symbol, duration_str, bar_size_setting, end_date_time_str)
            
            if use_cache:
                self._data_cache[cache_key] = df
                
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch data for {symbol}: {e}")
            raise DataError(f"Data fetching failed: {e}")
    
    def _fetch_from_ibkr(
        self,
        symbol: str,
        duration_str: str,
        bar_size_setting: str,
        end_date_time_str: str
    ) -> pd.DataFrame:
        """Fetch data from IBKR API."""
        from trade_utils import stock_order
        
        contract = stock_order(symbol)
        self.ib_client.data = []
        
        req_id = 100  # Could be made configurable
        
        self.ib_client.reqHistoricalData(
            reqId=req_id,
            contract=contract,
            endDateTime=end_date_time_str,
            durationStr=duration_str,
            barSizeSetting=bar_size_setting,
            whatToShow="TRADES",
            useRTH=1,
            formatDate=1,
            keepUpToDate=False
        )
        
        self.logger.info(f"Requested historical data for {symbol}")
        
        # Wait for data with timeout
        max_wait_time = 30
        start_time = time.time()
        
        while not self.ib_client.data and (time.time() - start_time) < max_wait_time:
            time.sleep(0.5)
        
        if not self.ib_client.data:
            raise DataError(f"Failed to fetch data for {symbol} within timeout")
        
        # Process the data
        df = pd.DataFrame(self.ib_client.data, columns=['DateTime', 'Close'])
        df = self._validate_and_clean_data(df)
        
        self.logger.info(f"Successfully fetched {len(df)} bars for {symbol}")
        return df
    
    def _generate_dummy_data(
        self,
        symbol: str,
        duration_str: str,
        bar_size_setting: str
    ) -> pd.DataFrame:
        """Generate dummy data for testing purposes."""
        self.logger.info(f"Generating dummy data for {symbol}")
        
        # Parse duration to determine number of periods
        duration_days = self._parse_duration_string(duration_str)
        
        # Generate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=duration_days)
        
        if 'min' in bar_size_setting:
            freq = f"{bar_size_setting.split()[0]}T"
        elif 'hour' in bar_size_setting:
            freq = f"{bar_size_setting.split()[0]}H"
        else:  # daily
            freq = 'D'
        
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # Generate realistic price data
        initial_price = 100 + np.random.randint(-10, 10)
        volatility = 0.01
        drift = 0.0005
        
        log_returns = np.random.normal(drift, volatility, len(dates))
        prices = initial_price * np.exp(log_returns.cumsum())
        
        df = pd.DataFrame({
            'DateTime': dates,
            'Close': prices
        })
        
        return self._validate_and_clean_data(df)
    
    def _validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the data."""
        if df.empty:
            raise DataError("Empty DataFrame provided")
        
        if 'DateTime' not in df.columns or 'Close' not in df.columns:
            raise DataError("Required columns (DateTime, Close) missing")
        
        # Convert DateTime to datetime if it's not already
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df.set_index('DateTime', inplace=True)
        
        # Convert Close to numeric and handle errors
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        
        # Remove rows with NaN close prices
        initial_len = len(df)
        df.dropna(subset=['Close'], inplace=True)
        
        if len(df) < initial_len:
            self.logger.warning(f"Removed {initial_len - len(df)} rows with invalid close prices")
        
        if df.empty:
            raise DataError("No valid data remaining after cleaning")
        
        # Check for reasonable price values
        if (df['Close'] <= 0).any():
            self.logger.warning("Found non-positive prices, removing them")
            df = df[df['Close'] > 0]
        
        # Sort by datetime
        df.sort_index(inplace=True)
        
        return df
    
    def _parse_duration_string(self, duration_str: str) -> int:
        """Parse IBKR duration string to days."""
        duration_str = duration_str.strip().upper()
        
        if 'D' in duration_str:
            return int(duration_str.replace('D', '').strip())
        elif 'W' in duration_str:
            return int(duration_str.replace('W', '').strip()) * 7
        elif 'M' in duration_str:
            return int(duration_str.replace('M', '').strip()) * 30
        elif 'Y' in duration_str:
            return int(duration_str.replace('Y', '').strip()) * 365
        else:
            return 1  # Default to 1 day
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the DataFrame."""
        if df.empty or 'Close' not in df.columns:
            raise DataError("Invalid DataFrame for technical indicators")
        
        df = df.copy()
        
        # Moving averages
        df['SMA_10'] = df['Close'].rolling(window=10, min_periods=1).mean()
        df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
        df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
        
        # Exponential moving averages
        df['EMA_10'] = df['Close'].ewm(span=10).mean()
        df['EMA_20'] = df['Close'].ewm(span=20).mean()
        
        # RSI
        df['RSI'] = self._calculate_rsi(df['Close'])
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        sma = df['Close'].rolling(window=bb_period).mean()
        std = df['Close'].rolling(window=bb_period).std()
        df['BB_Upper'] = sma + (std * bb_std)
        df['BB_Lower'] = sma - (std * bb_std)
        df['BB_Middle'] = sma
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the latest price for a symbol."""
        try:
            df = self.fetch_historical_data(symbol, "1 D", "1 min")
            if not df.empty:
                return float(df['Close'].iloc[-1])
        except Exception as e:
            self.logger.error(f"Failed to get latest price for {symbol}: {e}")
        
        return None
    
    def clear_cache(self):
        """Clear the data cache."""
        self._data_cache.clear()
        self.logger.info("Data cache cleared")