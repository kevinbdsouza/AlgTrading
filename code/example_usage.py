#!/usr/bin/env python3
"""
Example usage of the Improved Intraday Trading System.

This script demonstrates various features of the trading system:
- Backtesting with different strategies
- Strategy comparison
- Risk management
- Performance analysis
"""

import pandas as pd
import polars as pl
from datetime import datetime, timedelta

from config import TradingConfig
from intraday_strategy import IntradayTradingSystem
from logger import setup_logger
from factor_model import FactorModel


def example_basic_backtest():
    """Example: Basic backtesting with moving average strategy."""
    print("=== EXAMPLE 1: Basic Backtesting ===")
    
    # Create configuration
    config = TradingConfig(
        short_ma_period=10,
        long_ma_period=30,
        initial_capital=100000,
        slippage_per_trade=0.01
    )
    
    # Create trading system
    trading_system = IntradayTradingSystem(config=config)
    
    # Set strategy
    trading_system.set_strategy('ma_cross')
    
    # Run backtest
    results = trading_system.run_backtest(
        symbol='SPY',
        start_date='20230101',
        end_date='20231231',
        bar_size='1 day'
    )
    
    # Display results
    metrics = results.metrics
    print(f"Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"Number of Trades: {metrics['num_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2f}%")
    
    return results


def example_strategy_comparison():
    """Example: Compare multiple strategies."""
    print("\n=== EXAMPLE 2: Strategy Comparison ===")
    
    config = TradingConfig(initial_capital=100000)
    trading_system = IntradayTradingSystem(config=config)
    
    # Define strategies to compare
    strategy_configs = [
        {
            'name': 'ma_cross',
            # Uses default MA periods from config
        },
        {
            'name': 'rsi',
            'rsi_period': 14,
            'oversold_threshold': 30,
            'overbought_threshold': 70
        },
        {
            'name': 'bollinger',
            'period': 20,
            'std_dev': 2.0
        }
    ]
    
    # Run comparison
    results = trading_system.compare_strategies(
        strategy_configs,
        symbol='SPY',
        start_date='20230101',
        end_date='20231231',
        initial_capital=100000
    )
    
    # Display comparison
    comparison_df = results['comparison_table']
    print("\nStrategy Comparison Results:")
    print(comparison_df[['total_return_pct', 'sharpe_ratio', 'max_drawdown_pct', 'win_rate']].round(2))
    
    print(f"\nBest Strategy (by return): {results['best_strategy']}")
    
    return results


def example_custom_strategy():
    """Example: Using RSI strategy with custom parameters."""
    print("\n=== EXAMPLE 3: Custom RSI Strategy ===")
    
    config = TradingConfig(initial_capital=50000)
    trading_system = IntradayTradingSystem(config=config)
    
    # Set RSI strategy with custom parameters
    trading_system.set_strategy(
        'rsi',
        rsi_period=21,
        oversold_threshold=25,
        overbought_threshold=75
    )
    
    # Run backtest
    results = trading_system.run_backtest(
        symbol='AAPL',
        start_date='20230601',
        end_date='20231231',
        bar_size='1 day'
    )
    
    # Analyze trades
    trades_df = pd.DataFrame([trade.__dict__ for trade in results.trades])
    
    if not trades_df.empty:
        print(f"Total Trades: {len(trades_df)}")
        print(f"Average Trade Duration: {trades_df['duration_days'].mean():.1f} days")
        print(f"Best Trade: ${trades_df['pnl'].max():.2f}")
        print(f"Worst Trade: ${trades_df['pnl'].min():.2f}")
        
        # Show first few trades
        print("\nFirst 5 Trades:")
        print(trades_df[['entry_date', 'exit_date', 'entry_price', 'exit_price', 'pnl']].head())
    
    return results


def example_risk_management():
    """Example: Demonstrate risk management features."""
    print("\n=== EXAMPLE 4: Risk Management ===")
    
    # Configure with strict risk management
    config = TradingConfig(
        initial_capital=100000,
        max_position_size=50,  # Limit position size
        stop_loss_pct=0.02,    # 2% stop loss
        take_profit_pct=0.04,  # 4% take profit
        max_daily_loss=1000    # Max $1000 daily loss
    )
    
    trading_system = IntradayTradingSystem(config=config)
    trading_system.set_strategy('ma_cross')
    
    # Get risk manager for demonstration
    risk_manager = trading_system.risk_manager
    
    # Simulate some position updates
    print("Simulating position updates...")
    
    # Simulate buying 30 shares of SPY at $400
    risk_manager.update_position('SPY', 30, 400.0, 'BUY')
    
    # Update market price
    risk_manager.update_market_prices({'SPY': 410.0})
    
    # Get portfolio summary
    portfolio = risk_manager.get_portfolio_summary()
    print(f"Positions: {portfolio['positions_count']}")
    print(f"Total Market Value: ${portfolio['total_market_value']:.2f}")
    print(f"Unrealized P&L: ${portfolio['total_unrealized_pnl']:.2f}")
    
    # Check risk metrics
    risk_metrics = risk_manager.get_risk_metrics()
    print(f"Total Exposure: ${risk_metrics['total_exposure']:.2f}")
    print(f"Unrealized P&L %: {risk_metrics['unrealized_pnl_pct']:.2f}%")
    
    # Test order validation
    is_valid, reason = risk_manager.validate_order('AAPL', 25, 150.0, 'BUY')
    print(f"Order validation: {is_valid}, Reason: {reason}")
    
    # Test position sizing
    position_size = risk_manager.calculate_position_size('MSFT', 300.0)
    print(f"Recommended position size for MSFT: {position_size} shares")


def example_performance_analysis():
    """Example: Detailed performance analysis."""
    print("\n=== EXAMPLE 5: Performance Analysis ===")
    
    config = TradingConfig(initial_capital=100000)
    trading_system = IntradayTradingSystem(config=config)
    trading_system.set_strategy('ma_cross')
    
    # Run backtest
    results = trading_system.run_backtest(
        symbol='QQQ',
        start_date='20230101',
        end_date='20231231'
    )
    
    # Detailed analysis
    metrics = results.metrics
    portfolio_df = results.portfolio_history
    
    print("=== PERFORMANCE METRICS ===")
    print(f"Initial Capital: ${config.initial_capital:,.2f}")
    print(f"Final Value: ${metrics['final_value']:,.2f}")
    print(f"Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"Annualized Return: {(metrics['total_return_pct'] / 100) * (365 / 365):.2f}%")  # Simplified
    
    print(f"\n=== RISK METRICS ===")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"Volatility: {metrics['volatility_pct']:.2f}%")
    print(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")
    
    print(f"\n=== TRADE METRICS ===")
    print(f"Total Trades: {metrics['num_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2f}%")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Average Win: ${metrics['avg_win']:.2f}")
    print(f"Average Loss: ${metrics['avg_loss']:.2f}")
    print(f"Best Trade: ${metrics['max_win']:.2f}")
    print(f"Worst Trade: ${metrics['max_loss']:.2f}")
    print(f"Average Trade Duration: {metrics['avg_trade_duration_days']:.1f} days")
    
    # Portfolio evolution
    if not portfolio_df.empty:
        print(f"\n=== PORTFOLIO EVOLUTION ===")
        print(f"Starting Value: ${portfolio_df['total_value'].iloc[0]:,.2f}")
        print(f"Peak Value: ${portfolio_df['total_value'].max():,.2f}")
        print(f"Final Value: ${portfolio_df['total_value'].iloc[-1]:,.2f}")
        
        # Calculate monthly returns (simplified)
        monthly_returns = portfolio_df['total_value'].resample('M').last().pct_change().dropna()
        if len(monthly_returns) > 0:
            print(f"Best Month: {monthly_returns.max():.2%}")
            print(f"Worst Month: {monthly_returns.min():.2%}")
            print(f"Average Monthly Return: {monthly_returns.mean():.2%}")


def example_factor_model():
    """Example: Using toraniko factor model utilities."""
    print("\n=== EXAMPLE 6: Factor Model ===")

    # Load sample data from docs examples
    df = pl.read_csv("docs/examples/simulated_daily_data.csv", columns=[
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    ])
    df = df.rename({"Date": "date", "Close": "close_price"})
    df = df.with_columns(pl.lit("SIM").alias("symbol"))
    df = df.sort("date").with_columns(
        asset_returns=pl.col("close_price").pct_change().over("symbol")
    ).drop_nulls("asset_returns")

    returns_df = df.select(["date", "symbol", "asset_returns"])

    # Calculate momentum scores with FactorModel
    style_df = FactorModel.momentum_scores(returns_df, trailing_days=60)
    print(style_df.head())


def main():
    """Run all examples."""
    # Setup logging
    logger = setup_logger("examples", "INFO")
    logger.info("Starting trading system examples")
    
    try:
        # Run examples
        example_basic_backtest()
        example_strategy_comparison()
        example_custom_strategy()
        example_risk_management()
        example_performance_analysis()
        example_factor_model()
        
        print("\n=== ALL EXAMPLES COMPLETED ===")
        print("The improved trading system demonstrates:")
        print("✓ Multiple strategy support")
        print("✓ Comprehensive backtesting")
        print("✓ Risk management")
        print("✓ Performance analysis")
        print("✓ Modular architecture")
        print("✓ Error handling and logging")
        
    except Exception as e:
        logger.error(f"Example execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
