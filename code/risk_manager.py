"""
Risk management for the trading system.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from logger import get_logger
from exceptions import RiskManagementError
from config import TradingConfig
from backtester import Trade


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    quantity: int
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    @property
    def market_value(self) -> float:
        """Current market value of the position."""
        return self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L of the position."""
        return (self.current_price - self.entry_price) * self.quantity
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L percentage."""
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price * 100


class RiskManager:
    """Manages trading risk and position sizing."""
    
    def __init__(self, config: Optional[TradingConfig] = None):
        self.config = config or TradingConfig()
        self.logger = get_logger(f"{__name__}.RiskManager")
        
        # Risk tracking
        self.positions: Dict[str, Position] = {}
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset_date = datetime.now().date()
        
        # Risk limits
        self.max_position_size = self.config.max_position_size
        self.max_daily_loss = self.config.max_daily_loss
        self.stop_loss_pct = self.config.stop_loss_pct
        self.take_profit_pct = self.config.take_profit_pct
        
    def validate_order(self, symbol: str, quantity: int, price: float, 
                      action: str) -> Tuple[bool, str]:
        """
        Validate if an order meets risk management criteria.
        
        Args:
            symbol: Trading symbol
            quantity: Order quantity
            price: Order price
            action: Order action (BUY/SELL)
            
        Returns:
            Tuple of (is_valid, reason)
        """
        try:
            self._reset_daily_counters_if_needed()
            
            # Check daily loss limit
            if self.daily_pnl <= -self.max_daily_loss:
                return False, f"Daily loss limit exceeded: {self.daily_pnl:.2f}"
            
            # Check position size limits
            if action.upper() == "BUY":
                current_quantity = self.positions.get(symbol, Position(symbol, 0, 0, datetime.now())).quantity
                new_total_quantity = current_quantity + quantity
                
                if new_total_quantity > self.max_position_size:
                    return False, f"Position size limit exceeded: {new_total_quantity} > {self.max_position_size}"
            
            # Check if we have enough quantity to sell
            elif action.upper() == "SELL":
                current_quantity = self.positions.get(symbol, Position(symbol, 0, 0, datetime.now())).quantity
                
                if quantity > current_quantity:
                    return False, f"Insufficient position to sell: {quantity} > {current_quantity}"
            
            # Additional risk checks can be added here
            # - Correlation limits
            # - Sector exposure limits
            # - Volatility-based position sizing
            
            return True, "Order validated"
            
        except Exception as e:
            self.logger.error(f"Error validating order: {e}")
            return False, f"Validation error: {e}"
    
    def calculate_position_size(self, symbol: str, price: float, 
                               risk_per_trade: float = 0.01) -> int:
        """
        Calculate optimal position size based on risk management.
        
        Args:
            symbol: Trading symbol
            price: Entry price
            risk_per_trade: Risk per trade as percentage of portfolio
            
        Returns:
            Recommended position size
        """
        try:
            # Simple position sizing based on fixed risk per trade
            # In a real system, this would consider:
            # - Portfolio value
            # - Volatility
            # - Stop loss distance
            # - Correlation with existing positions
            
            portfolio_value = self.config.initial_capital  # Simplified
            risk_amount = portfolio_value * risk_per_trade
            
            # Assume 2% stop loss for position sizing
            stop_distance = price * self.stop_loss_pct
            position_size = int(risk_amount / stop_distance)
            
            # Apply maximum position size limit
            position_size = min(position_size, self.max_position_size)
            
            self.logger.info(f"Calculated position size for {symbol}: {position_size}")
            return max(1, position_size)  # Minimum 1 share
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 1  # Default to 1 share
    
    def update_position(self, symbol: str, quantity: int, price: float,
                       action: str, timestamp: Optional[datetime] = None) -> Optional[Trade]:
        """
        Update position after a trade.
        
        Args:
            symbol: Trading symbol
            quantity: Trade quantity
            price: Trade price
            action: Trade action (BUY/SELL)
            timestamp: Trade timestamp
        """
        try:
            timestamp = timestamp or datetime.now()
            action = action.upper()
            
            if symbol not in self.positions:
                if action == "BUY":
                    # Create new position
                    stop_loss = price * (1 - self.stop_loss_pct)
                    take_profit = price * (1 + self.take_profit_pct)

                    self.positions[symbol] = Position(
                        symbol=symbol,
                        quantity=quantity,
                        entry_price=price,
                        entry_time=timestamp,
                        current_price=price,
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )

                    self.logger.info(f"Opened new position: {symbol} {quantity}@{price:.2f}")
                    self.daily_trades += 1
                    return None
                else:
                    self.logger.warning(f"Cannot sell {symbol}: no existing position")
                    return None
            else:
                position = self.positions[symbol]
                
                if action == "BUY":
                    # Add to existing position (average price)
                    total_cost = (position.quantity * position.entry_price) + (quantity * price)
                    total_quantity = position.quantity + quantity
                    
                    position.entry_price = total_cost / total_quantity
                    position.quantity = total_quantity
                    position.current_price = price
                    
                    # Update stop loss and take profit
                    position.stop_loss = position.entry_price * (1 - self.stop_loss_pct)
                    position.take_profit = position.entry_price * (1 + self.take_profit_pct)
                    
                    self.logger.info(
                        f"Added to position: {symbol} {quantity}@{price:.2f}, new avg: {position.entry_price:.2f}")
                    return None
                
                elif action == "SELL":
                    # Reduce or close position
                    if quantity >= position.quantity:
                        # Close entire position
                        realized_pnl = (price - position.entry_price) * position.quantity
                        self.daily_pnl += realized_pnl

                        self.logger.info(
                            f"Closed position: {symbol} {position.quantity}@{price:.2f}, PnL: {realized_pnl:.2f}")

                        trade = Trade(
                            symbol=symbol,
                            entry_date=position.entry_time,
                            exit_date=timestamp,
                            entry_price=position.entry_price,
                            exit_price=price,
                            quantity=position.quantity,
                            side="LONG",
                            pnl=realized_pnl,
                            pnl_pct=((price - position.entry_price) / position.entry_price) * 100,
                            duration_days=(timestamp - position.entry_time).days
                        )

                        del self.positions[symbol]
                        self.daily_trades += 1
                        return trade
                    else:
                        # Partial close
                        realized_pnl = (price - position.entry_price) * quantity
                        self.daily_pnl += realized_pnl
                        position.quantity -= quantity
                        position.current_price = price

                        self.logger.info(
                            f"Partially closed position: {symbol} {quantity}@{price:.2f}, PnL: {realized_pnl:.2f}, remaining: {position.quantity}")

                        trade = Trade(
                            symbol=symbol,
                            entry_date=position.entry_time,
                            exit_date=timestamp,
                            entry_price=position.entry_price,
                            exit_price=price,
                            quantity=quantity,
                            side="LONG",
                            pnl=realized_pnl,
                            pnl_pct=((price - position.entry_price) / position.entry_price) * 100,
                            duration_days=(timestamp - position.entry_time).days
                        )

                        self.daily_trades += 1
                        return trade

            self.daily_trades += 1
            return None
            
        except Exception as e:
            self.logger.error(f"Error updating position: {e}")
            raise RiskManagementError(f"Position update failed: {e}")
    
    def check_stop_loss_take_profit(self, symbol: str, current_price: float) -> Optional[str]:
        """
        Check if position should be closed due to stop loss or take profit.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            
        Returns:
            Action to take ('SELL' or None)
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        position.current_price = current_price
        
        # Check stop loss
        if position.stop_loss and current_price <= position.stop_loss:
            self.logger.warning(f"Stop loss triggered for {symbol}: {current_price:.2f} <= {position.stop_loss:.2f}")
            return "SELL"
        
        # Check take profit
        if position.take_profit and current_price >= position.take_profit:
            self.logger.info(f"Take profit triggered for {symbol}: {current_price:.2f} >= {position.take_profit:.2f}")
            return "SELL"
        
        return None
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary with risk metrics."""
        total_market_value = sum(pos.market_value for pos in self.positions.values())
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        return {
            'positions_count': len(self.positions),
            'total_market_value': total_market_value,
            'total_unrealized_pnl': total_unrealized_pnl,
            'daily_realized_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'positions': {
                symbol: {
                    'quantity': pos.quantity,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'market_value': pos.market_value,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'unrealized_pnl_pct': pos.unrealized_pnl_pct,
                    'stop_loss': pos.stop_loss,
                    'take_profit': pos.take_profit
                }
                for symbol, pos in self.positions.items()
            }
        }
    
    def _reset_daily_counters_if_needed(self) -> None:
        """Reset daily counters if it's a new day."""
        current_date = datetime.now().date()
        
        if current_date > self.last_reset_date:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset_date = current_date
            self.logger.info("Reset daily counters for new trading day")
    
    def update_market_prices(self, prices: Dict[str, float]) -> None:
        """Update current market prices for all positions."""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price
    
    def get_risk_metrics(self) -> Dict[str, float]:
        """Calculate portfolio risk metrics."""
        if not self.positions:
            return {
                'total_exposure': 0.0,
                'largest_position_pct': 0.0,
                'unrealized_pnl_pct': 0.0,
                'daily_pnl_pct': 0.0
            }
        
        total_market_value = sum(pos.market_value for pos in self.positions.values())
        largest_position = max(pos.market_value for pos in self.positions.values())
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        portfolio_value = self.config.initial_capital  # Simplified
        
        return {
            'total_exposure': total_market_value,
            'largest_position_pct': (largest_position / total_market_value * 100) if total_market_value > 0 else 0,
            'unrealized_pnl_pct': (total_unrealized_pnl / portfolio_value * 100) if portfolio_value > 0 else 0,
            'daily_pnl_pct': (self.daily_pnl / portfolio_value * 100) if portfolio_value > 0 else 0
        }