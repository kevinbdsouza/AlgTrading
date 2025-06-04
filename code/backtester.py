"""
Backtesting engine for trading strategies.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass

from logger import get_logger
from exceptions import BacktestError
from config import TradingConfig
from strategy_base import BaseStrategy, SignalType


@dataclass
class Trade:
    """Represents a completed trade."""
    symbol: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    quantity: int
    side: str  # 'LONG' or 'SHORT'
    pnl: float
    pnl_pct: float
    duration_days: float


@dataclass
class BacktestResults:
    """Results from a backtest run."""
    trades: List[Trade]
    portfolio_history: pd.DataFrame
    metrics: Dict[str, Any]
    strategy_params: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            'trades': [
                {
                    'symbol': t.symbol,
                    'entry_date': t.entry_date.isoformat(),
                    'exit_date': t.exit_date.isoformat(),
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'quantity': t.quantity,
                    'side': t.side,
                    'pnl': t.pnl,
                    'pnl_pct': t.pnl_pct,
                    'duration_days': t.duration_days
                }
                for t in self.trades
            ],
            'portfolio_history': self.portfolio_history.to_dict('records'),
            'metrics': self.metrics,
            'strategy_params': self.strategy_params
        }


class Backtester:
    """Backtesting engine for trading strategies."""
    
    def __init__(self, config: Optional[TradingConfig] = None):
        self.config = config or TradingConfig()
        self.logger = get_logger(f"{__name__}.Backtester")
        
    def run_backtest(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        initial_capital: float = 100000.0,
        commission: float = 1.0,
        slippage: float = 0.01
    ) -> BacktestResults:
        """
        Run a backtest for the given strategy and data.
        
        Args:
            strategy: Trading strategy to test
            data: Historical price data
            initial_capital: Starting capital
            commission: Commission per trade
            slippage: Slippage per share
            
        Returns:
            BacktestResults object
        """
        try:
            self.logger.info(f"Starting backtest for {strategy.name}")
            
            # Validate inputs
            self._validate_inputs(data, initial_capital)
            
            # Generate signals
            signals_df = strategy.generate_signals(data)
            
            # Run simulation
            trades, portfolio_history = self._simulate_trading(
                signals_df, initial_capital, commission, slippage
            )
            
            # Calculate metrics
            metrics = self._calculate_metrics(trades, portfolio_history, initial_capital)
            
            # Create results
            results = BacktestResults(
                trades=trades,
                portfolio_history=portfolio_history,
                metrics=metrics,
                strategy_params=strategy.get_strategy_params()
            )
            
            self.logger.info(f"Backtest completed: {len(trades)} trades, "
                           f"Final return: {metrics['total_return_pct']:.2f}%")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            raise BacktestError(f"Backtest execution failed: {e}")
    
    def _validate_inputs(self, data: pd.DataFrame, initial_capital: float) -> None:
        """Validate backtest inputs."""
        if data.empty:
            raise BacktestError("Empty data provided")
        
        if 'Close' not in data.columns:
            raise BacktestError("Data must contain 'Close' column")
        
        if initial_capital <= 0:
            raise BacktestError("Initial capital must be positive")
        
        if data['Close'].isna().all():
            raise BacktestError("All close prices are NaN")
    
    def _simulate_trading(
        self,
        signals_df: pd.DataFrame,
        initial_capital: float,
        commission: float,
        slippage: float
    ) -> Tuple[List[Trade], pd.DataFrame]:
        """Simulate trading based on signals."""
        trades = []
        portfolio_history = []
        
        # Trading state
        cash = initial_capital
        position = 0  # 0: flat, >0: long position size
        entry_price = 0.0
        entry_date = None
        
        for timestamp, row in signals_df.iterrows():
            current_price = row['Close']
            signal = row.get('signal', 0)
            
            # Skip if price is invalid
            if pd.isna(current_price) or current_price <= 0:
                portfolio_history.append({
                    'date': timestamp,
                    'cash': cash,
                    'position_value': position * current_price if not pd.isna(current_price) else 0,
                    'total_value': cash + (position * current_price if not pd.isna(current_price) else 0),
                    'position_size': position
                })
                continue
            
            # Apply slippage
            buy_price = current_price + slippage
            sell_price = current_price - slippage
            
            # Process signals
            if signal == SignalType.BUY.value and position == 0:
                # Enter long position
                shares_to_buy = int((cash - commission) / buy_price)
                if shares_to_buy > 0:
                    cost = shares_to_buy * buy_price + commission
                    cash -= cost
                    position = shares_to_buy
                    entry_price = buy_price
                    entry_date = timestamp
                    
                    self.logger.debug(f"BUY: {shares_to_buy} shares at {buy_price:.2f}")
            
            elif signal == SignalType.SELL.value and position > 0:
                # Exit long position
                proceeds = position * sell_price - commission
                cash += proceeds
                
                # Record trade
                trade = Trade(
                    symbol="BACKTEST",  # Could be parameterized
                    entry_date=entry_date,
                    exit_date=timestamp,
                    entry_price=entry_price,
                    exit_price=sell_price,
                    quantity=position,
                    side="LONG",
                    pnl=proceeds - (position * entry_price + commission),
                    pnl_pct=((sell_price - entry_price) / entry_price) * 100,
                    duration_days=(timestamp - entry_date).days if entry_date else 0
                )
                trades.append(trade)
                
                self.logger.debug(f"SELL: {position} shares at {sell_price:.2f}, "
                                f"PnL: {trade.pnl:.2f}")
                
                position = 0
                entry_price = 0.0
                entry_date = None
            
            # Record portfolio state
            position_value = position * current_price
            total_value = cash + position_value
            
            portfolio_history.append({
                'date': timestamp,
                'cash': cash,
                'position_value': position_value,
                'total_value': total_value,
                'position_size': position,
                'price': current_price
            })
        
        # Close any remaining position at the end
        if position > 0:
            final_price = signals_df['Close'].iloc[-1]
            final_sell_price = final_price - slippage
            proceeds = position * final_sell_price - commission
            cash += proceeds
            
            trade = Trade(
                symbol="BACKTEST",
                entry_date=entry_date,
                exit_date=signals_df.index[-1],
                entry_price=entry_price,
                exit_price=final_sell_price,
                quantity=position,
                side="LONG",
                pnl=proceeds - (position * entry_price + commission),
                pnl_pct=((final_sell_price - entry_price) / entry_price) * 100,
                duration_days=(signals_df.index[-1] - entry_date).days if entry_date else 0
            )
            trades.append(trade)
            
            # Update final portfolio state
            portfolio_history[-1]['cash'] = cash
            portfolio_history[-1]['position_value'] = 0
            portfolio_history[-1]['total_value'] = cash
            portfolio_history[-1]['position_size'] = 0
        
        portfolio_df = pd.DataFrame(portfolio_history)
        if not portfolio_df.empty:
            portfolio_df.set_index('date', inplace=True)
        
        return trades, portfolio_df
    
    def _calculate_metrics(
        self,
        trades: List[Trade],
        portfolio_history: pd.DataFrame,
        initial_capital: float
    ) -> Dict[str, Any]:
        """Calculate performance metrics."""
        if portfolio_history.empty:
            return self._empty_metrics()
        
        # Basic metrics
        final_value = portfolio_history['total_value'].iloc[-1]
        total_return = final_value - initial_capital
        total_return_pct = (total_return / initial_capital) * 100
        
        # Trade-based metrics
        if trades:
            trade_pnls = [t.pnl for t in trades]
            winning_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl < 0]
            
            win_rate = (len(winning_trades) / len(trades)) * 100
            avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
            profit_factor = abs(sum([t.pnl for t in winning_trades]) / sum([t.pnl for t in losing_trades])) if losing_trades else np.inf
            
            max_win = max(trade_pnls) if trade_pnls else 0
            max_loss = min(trade_pnls) if trade_pnls else 0
            avg_trade_duration = np.mean([t.duration_days for t in trades])
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            max_win = 0
            max_loss = 0
            avg_trade_duration = 0
        
        # Portfolio-based metrics
        returns = portfolio_history['total_value'].pct_change().dropna()
        
        if len(returns) > 1:
            # Sharpe ratio (assuming daily data, risk-free rate = 0)
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

            # Sortino ratio
            downside = returns[returns < 0]
            sortino_ratio = (returns.mean() / downside.std()) * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 0
            
            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min() * 100
            
            # Volatility
            volatility = returns.std() * np.sqrt(252) * 100
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
            max_drawdown = 0
            volatility = 0
        
        # Calmar ratio
        calmar_ratio = (total_return_pct / abs(max_drawdown)) if max_drawdown != 0 else 0

        expectancy = ((win_rate / 100) * avg_win) + (((100 - win_rate) / 100) * avg_loss)

        # Trades per year approximation
        if not portfolio_history.empty:
            days = (portfolio_history.index[-1] - portfolio_history.index[0]).days
            years = days / 365 if days > 0 else 1
        else:
            years = 1
        trades_per_year = len(trades) / years if years else len(trades)
        
        return {
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'final_value': final_value,
            'num_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_win': max_win,
            'max_loss': max_loss,
            'avg_trade_duration_days': avg_trade_duration,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown_pct': max_drawdown,
            'volatility_pct': volatility,
            'calmar_ratio': calmar_ratio,
            'expectancy': expectancy,
            'trades_per_year': trades_per_year
        }
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics for failed backtests."""
        return {
            'total_return': 0,
            'total_return_pct': 0,
            'final_value': 0,
            'num_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'max_win': 0,
            'max_loss': 0,
            'avg_trade_duration_days': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'max_drawdown_pct': 0,
            'volatility_pct': 0,
            'calmar_ratio': 0,
            'expectancy': 0,
            'trades_per_year': 0
        }
    
    def compare_strategies(
        self,
        strategies: List[BaseStrategy],
        data: pd.DataFrame,
        **backtest_kwargs
    ) -> pd.DataFrame:
        """
        Compare multiple strategies on the same data.
        
        Args:
            strategies: List of strategies to compare
            data: Historical data
            **backtest_kwargs: Additional arguments for backtesting
            
        Returns:
            DataFrame with comparison metrics
        """
        results = []
        
        for strategy in strategies:
            try:
                result = self.run_backtest(strategy, data, **backtest_kwargs)
                metrics = result.metrics.copy()
                metrics['strategy_name'] = strategy.name
                results.append(metrics)
            except Exception as e:
                self.logger.error(f"Failed to backtest {strategy.name}: {e}")
                # Add empty result to maintain comparison structure
                empty_metrics = self._empty_metrics()
                empty_metrics['strategy_name'] = strategy.name
                results.append(empty_metrics)
        
        comparison_df = pd.DataFrame(results)
        if not comparison_df.empty:
            comparison_df.set_index('strategy_name', inplace=True)
        
        return comparison_df