"""
Improved Intraday Trading Strategy System.

This module provides a comprehensive trading system with:
- Multiple strategy support
- Risk management
- Backtesting capabilities
- Live trading integration
- Performance monitoring
"""
import threading
import time
from datetime import datetime
from typing import Optional, Dict, Any, List

from config import TradingConfig
from logger import setup_logger, get_logger
from exceptions import TradingError, ConnectionError, OrderError
from data_manager import DataManager
from strategy_base import BaseStrategy, StrategyFactory, SignalType
from risk_manager import RiskManager
from backtester import Backtester, BacktestResults
from ibkr_client import IBapi
import trade_utils


class IntradayTradingSystem:
    """
    Comprehensive intraday trading system.
    
    Features:
    - Multiple strategy support
    - Risk management
    - Live trading and backtesting
    - Performance monitoring
    - Error handling and logging
    """
    
    def __init__(self, config: Optional[TradingConfig] = None, ib_client: Optional[IBapi] = None):
        """
        Initialize the trading system.
        
        Args:
            config: Trading configuration
            ib_client: IBKR client instance
        """
        self.config = config or TradingConfig()
        self.ib_client = ib_client
        
        # Setup logging
        self.logger = setup_logger(
            name="trading_system",
            level=self.config.log_level,
            log_file=self.config.log_file
        )
        
        # Initialize components
        self.data_manager = DataManager(ib_client, self.config)
        self.risk_manager = RiskManager(self.config)
        self.backtester = Backtester(self.config)
        
        # Trading state
        self.is_trading = False
        self.current_strategy: Optional[BaseStrategy] = None
        self.performance_metrics: Dict[str, Any] = {}
        
        self.logger.info("Intraday Trading System initialized")
    
    def set_strategy(self, strategy_name: str, **strategy_params) -> None:
        """
        Set the trading strategy.
        
        Args:
            strategy_name: Name of the strategy to use
            **strategy_params: Additional strategy parameters
        """
        try:
            self.current_strategy = StrategyFactory.create_strategy(
                strategy_name, self.config, **strategy_params
            )
            self.logger.info(f"Strategy set to: {self.current_strategy.name}")
        except Exception as e:
            self.logger.error(f"Failed to set strategy: {e}")
            raise TradingError(f"Strategy setup failed: {e}")
    
    def connect_to_ibkr(self) -> bool:
        """
        Connect to IBKR if not already connected.
        
        Returns:
            True if connected successfully
        """
        if self.ib_client and hasattr(self.ib_client, 'isConnected') and self.ib_client.isConnected():
            self.logger.info("Already connected to IBKR")
            return True
        
        if not self.ib_client:
            self.ib_client = IBapi()
        
        try:
            self.ib_client.connect(
                self.config.ibkr_host,
                self.config.ibkr_port,
                self.config.ibkr_client_id
            )
            
            # Start API thread
            api_thread = threading.Thread(target=self.ib_client.run, daemon=True)
            api_thread.start()
            
            # Wait for connection
            start_time = time.time()
            while not (hasattr(self.ib_client, 'nextorderId') and self.ib_client.nextorderId is not None):
                time.sleep(0.5)
                if time.time() - start_time > self.config.connection_timeout:
                    raise ConnectionError("Connection timeout")
            
            self.logger.info(f"Connected to IBKR. Next Order ID: {self.ib_client.nextorderId}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to IBKR: {e}")
            raise ConnectionError(f"IBKR connection failed: {e}")
    
    def run_live_trading(self, symbol: str, duration_minutes: int = 60) -> None:
        """
        Run live trading for a specified duration.
        
        Args:
            symbol: Trading symbol
            duration_minutes: Trading duration in minutes
        """
        if not self.current_strategy:
            raise TradingError("No strategy set")
        
        if not self.ib_client or not hasattr(self.ib_client, 'nextorderId'):
            raise ConnectionError("IBKR not connected")
        
        self.logger.info(f"Starting live trading for {symbol} using {self.current_strategy.name}")
        self.is_trading = True
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        try:
            while self.is_trading and time.time() < end_time:
                # Fetch latest data
                data = self.data_manager.fetch_historical_data(
                    symbol, "2 D", self.config.default_timeframe
                )
                
                if data.empty:
                    self.logger.warning("No data received, skipping iteration")
                    time.sleep(30)  # Wait before next iteration
                    continue
                
                # Generate signals
                signals_df = self.current_strategy.generate_signals(data)
                
                if signals_df.empty or 'signal' not in signals_df.columns:
                    self.logger.warning("No signals generated")
                    time.sleep(30)
                    continue
                
                # Get latest signal
                latest_signal = signals_df['signal'].iloc[-1]
                latest_price = signals_df['Close'].iloc[-1]
                
                if latest_signal != SignalType.HOLD.value:
                    self._execute_live_trade(symbol, latest_signal, latest_price)
                
                # Check risk management
                self._check_risk_management(symbol, latest_price)
                
                # Wait before next iteration
                time.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            self.logger.info("Live trading interrupted by user")
        except Exception as e:
            self.logger.error(f"Live trading error: {e}")
            raise TradingError(f"Live trading failed: {e}")
        finally:
            self.is_trading = False
            self.logger.info("Live trading stopped")
    
    def _execute_live_trade(self, symbol: str, signal: int, price: float) -> None:
        """Execute a live trade based on signal."""
        try:
            action = "BUY" if signal == SignalType.BUY.value else "SELL"
            quantity = self.risk_manager.calculate_position_size(symbol, price)
            
            # Validate order with risk manager
            is_valid, reason = self.risk_manager.validate_order(symbol, quantity, price, action)
            
            if not is_valid:
                self.logger.warning(f"Order rejected by risk manager: {reason}")
                return
            
            # Create and place order
            contract = trade_utils.stock_order(symbol)
            order = trade_utils.get_order(action=action, quantity=quantity, order_type="MKT")
            order.orderId = self.ib_client.nextorderId
            self.ib_client.nextorderId += 1
            
            self.ib_client.placeOrder(order.orderId, contract, order)
            
            # Update risk manager
            self.risk_manager.update_position(symbol, quantity, price, action)
            
            self.logger.info(f"Executed {action} order: {quantity} {symbol} at {price:.2f}")
            
        except Exception as e:
            self.logger.error(f"Failed to execute trade: {e}")
            raise OrderError(f"Trade execution failed: {e}")
    
    def _check_risk_management(self, symbol: str, current_price: float) -> None:
        """Check risk management rules."""
        action = self.risk_manager.check_stop_loss_take_profit(symbol, current_price)
        
        if action:
            self.logger.info(f"Risk management triggered {action} for {symbol}")
            self._execute_live_trade(symbol, SignalType.SELL.value, current_price)
    
    def run_backtest(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        bar_size: str = "1 day",
        initial_capital: float = None
    ) -> BacktestResults:
        """
        Run a backtest for the current strategy.
        
        Args:
            symbol: Trading symbol
            start_date: Start date (YYYYMMDD)
            end_date: End date (YYYYMMDD)
            bar_size: Bar size setting
            initial_capital: Initial capital (uses config default if None)
            
        Returns:
            Backtest results
        """
        if not self.current_strategy:
            raise TradingError("No strategy set")
        
        initial_capital = initial_capital or self.config.initial_capital
        
        self.logger.info(f"Starting backtest: {symbol} from {start_date} to {end_date}")
        
        try:
            # Fetch historical data
            duration_str = self._calculate_duration(start_date, end_date)
            end_datetime = end_date + " 23:59:59"
            
            data = self.data_manager.fetch_historical_data(
                symbol, duration_str, bar_size, end_datetime, use_cache=False
            )
            
            if data.empty:
                raise TradingError("No data available for backtest period")
            
            # Run backtest
            results = self.backtester.run_backtest(
                self.current_strategy,
                data,
                initial_capital=initial_capital,
                commission=1.0,
                slippage=self.config.slippage_per_trade
            )
            
            # Store performance metrics
            self.performance_metrics = results.metrics
            
            self._log_backtest_results(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            raise TradingError(f"Backtest execution failed: {e}")
    
    def _calculate_duration(self, start_date: str, end_date: str) -> str:
        """Calculate duration string for IBKR API."""
        from datetime import datetime
        
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        duration_days = (end_dt - start_dt).days + 1
        
        return f"{duration_days} D"
    
    def _log_backtest_results(self, results: BacktestResults) -> None:
        """Log backtest results."""
        metrics = results.metrics
        
        self.logger.info("=== BACKTEST RESULTS ===")
        self.logger.info(f"Strategy: {self.current_strategy.name}")
        self.logger.info(f"Total Return: {metrics['total_return_pct']:.2f}%")
        self.logger.info(f"Number of Trades: {metrics['num_trades']}")
        self.logger.info(f"Win Rate: {metrics['win_rate']:.2f}%")
        self.logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        self.logger.info(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        self.logger.info(f"Profit Factor: {metrics['profit_factor']:.2f}")
        
        if metrics['num_trades'] > 0:
            self.logger.info(f"Average Win: ${metrics['avg_win']:.2f}")
            self.logger.info(f"Average Loss: ${metrics['avg_loss']:.2f}")
            self.logger.info(f"Average Trade Duration: {metrics['avg_trade_duration_days']:.1f} days")
    
    def compare_strategies(
        self,
        strategy_configs: List[Dict[str, Any]],
        symbol: str,
        start_date: str,
        end_date: str,
        **backtest_kwargs
    ) -> Dict[str, Any]:
        """
        Compare multiple strategies.
        
        Args:
            strategy_configs: List of strategy configurations
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            **backtest_kwargs: Additional backtest parameters
            
        Returns:
            Comparison results
        """
        strategies = []
        
        for config in strategy_configs:
            strategy_name = config.pop('name')
            strategy = StrategyFactory.create_strategy(strategy_name, self.config, **config)
            strategies.append(strategy)
        
        # Fetch data once
        duration_str = self._calculate_duration(start_date, end_date)
        end_datetime = end_date + " 23:59:59"
        
        data = self.data_manager.fetch_historical_data(
            symbol, duration_str, backtest_kwargs.get('bar_size', '1 day'), end_datetime
        )
        
        # Compare strategies
        comparison_df = self.backtester.compare_strategies(strategies, data, **backtest_kwargs)
        
        self.logger.info(f"Strategy comparison completed for {len(strategies)} strategies")
        
        return {
            'comparison_table': comparison_df,
            'best_strategy': comparison_df.loc[comparison_df['total_return_pct'].idxmax()].name if not comparison_df.empty else None
        }
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary."""
        return self.risk_manager.get_portfolio_summary()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.performance_metrics.copy()
    
    def stop_trading(self) -> None:
        """Stop live trading."""
        self.is_trading = False
        self.logger.info("Trading stop requested")
    
    def disconnect(self) -> None:
        """Disconnect from IBKR."""
        if self.ib_client and hasattr(self.ib_client, 'disconnect'):
            self.ib_client.disconnect()
            self.logger.info("Disconnected from IBKR")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_trading()
        self.disconnect()


# Example usage and main execution
def main():
    """Main execution function demonstrating the trading system."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Intraday Trading System')
    parser.add_argument('--mode', choices=['backtest', 'live', 'compare'], 
                       default='backtest', help='Trading mode')
    parser.add_argument('--symbol', default='SPY', help='Trading symbol')
    parser.add_argument('--strategy', default='ma_cross', help='Strategy name')
    parser.add_argument('--start-date', default='20230101', help='Start date (YYYYMMDD)')
    parser.add_argument('--end-date', default='20231231', help='End date (YYYYMMDD)')
    parser.add_argument('--duration', type=int, default=60, help='Live trading duration (minutes)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = TradingConfig.from_env()
    
    # Create trading system
    with IntradayTradingSystem(config=config) as trading_system:
        
        # Set strategy
        trading_system.set_strategy(args.strategy)
        
        if args.mode == 'backtest':
            # Run backtest
            results = trading_system.run_backtest(
                symbol=args.symbol,
                start_date=args.start_date,
                end_date=args.end_date
            )
            
            print("\n=== BACKTEST SUMMARY ===")
            print(f"Total Return: {results.metrics['total_return_pct']:.2f}%")
            print(f"Sharpe Ratio: {results.metrics['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {results.metrics['max_drawdown_pct']:.2f}%")
            
        elif args.mode == 'live':
            # Connect to IBKR and run live trading
            if trading_system.connect_to_ibkr():
                trading_system.run_live_trading(args.symbol, args.duration)
            else:
                print("Failed to connect to IBKR")
                
        elif args.mode == 'compare':
            # Compare multiple strategies
            strategy_configs = [
                {'name': 'ma_cross'},
                {'name': 'rsi', 'oversold_threshold': 25, 'overbought_threshold': 75},
                {'name': 'bollinger', 'period': 20, 'std_dev': 2.0}
            ]
            
            results = trading_system.compare_strategies(
                strategy_configs,
                args.symbol,
                args.start_date,
                args.end_date
            )
            
            print("\n=== STRATEGY COMPARISON ===")
            print(results['comparison_table'][['total_return_pct', 'sharpe_ratio', 'max_drawdown_pct']])
            print(f"\nBest Strategy: {results['best_strategy']}")


if __name__ == "__main__":
    main()
