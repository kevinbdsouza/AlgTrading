import datetime
import time
import pandas as pd
import numpy as np # For Sharpe Ratio calculation
from ibkr_client import IBapi
from trade_utils import stock_order, get_order, analysis # Make sure trade_utils is importable
import threading

class IntradayStrategy:
    def __init__(self, ib_client=None, trade_utils_module=None): # Allow None for backtesting if ib_client not needed
        self.ib_client = ib_client
        self.trade_utils = trade_utils_module
        self.strategy_name = "SimpleMA_Cross"
        self.symbol = "SPY"
        self.timeframe = "5 mins"
        self.short_ma_period = 10
        self.long_ma_period = 30
        self.quantity = 1 # Default quantity for live trades
        self.current_position = 0 # Live trading: 0 = flat, 1 = long, -1 = short
        self.historical_data_req_id = 100
        self.live_data_req_id = 101
        self.portfolio_value = 100000 # Initial portfolio value for backtesting
        self.slippage_per_trade = 0.01 # Example slippage per share per trade

    def fetch_historical_data(self, symbol, duration_str="1 D", bar_size_setting="5 mins", end_date_time_str=""):
        if not self.ib_client:
            print("IB client not available for fetching historical data (possibly in backtest mode).")
            # In a pure backtest scenario without ib_client, this method would load data from a CSV or DB.
            # For this integrated example, we'll assume ib_client is needed.
            # If you want to support backtesting from CSV, this method needs modification.
            return pd.DataFrame()

        contract = self.trade_utils.stock_order(symbol)
        self.ib_client.data = [] 
        self.ib_client.reqHistoricalData(
            reqId=self.historical_data_req_id,
            contract=contract,
            endDateTime=end_date_time_str, 
            durationStr=duration_str,
            barSizeSetting=bar_size_setting,
            whatToShow="TRADES",
            useRTH=1,
            formatDate=1,
            keepUpToDate=False
        )
        print(f"Requested historical data for {symbol} (duration: {duration_str}, end: {end_date_time_str}, bar: {bar_size_setting}). Waiting for data...")
        max_wait_time = 30 
        start_time = time.time()
        while not self.ib_client.data and (time.time() - start_time) < max_wait_time:
            time.sleep(0.5)

        if not self.ib_client.data:
            print(f"Failed to fetch historical data for {symbol} within timeout.")
            return pd.DataFrame()

        # Correctly handle data format from IB: list of BarData objects or list of lists
        # Assuming self.ib_client.data is populated as [[time, close, open, high, low, volume, ...]]
        # Or if it's BarData objects: [[bar.date, bar.close, bar.open, ...]]
        # The provided IBapi historicalData callback appends [bar.date, bar.close]

        df = pd.DataFrame(self.ib_client.data, columns=['DateTime', 'Close']) # Adjust if more columns are added by historicalData
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df.set_index('DateTime', inplace=True)
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df.dropna(subset=['Close'], inplace=True) # Ensure no NaN close prices
        print(f"Successfully fetched {len(df)} bars for {symbol}.")
        return df

    def calculate_signals(self, data_df):
        if data_df.empty or 'Close' not in data_df.columns:
            print("Data is empty or 'Close' column is missing for signal calculation.")
            return data_df.assign(short_ma=pd.NA, long_ma=pd.NA, signal=0)


        signals_df = data_df.copy()
        signals_df['short_ma'] = signals_df['Close'].rolling(window=self.short_ma_period, min_periods=1).mean()
        signals_df['long_ma'] = signals_df['Close'].rolling(window=self.long_ma_period, min_periods=1).mean()
        
        signals_df['signal'] = 0
        signals_df['prev_short_ma'] = signals_df['short_ma'].shift(1)
        signals_df['prev_long_ma'] = signals_df['long_ma'].shift(1)

        signals_df.loc[(signals_df['short_ma'] > signals_df['long_ma']) & (signals_df['prev_short_ma'] <= signals_df['prev_long_ma']), 'signal'] = 1
        signals_df.loc[(signals_df['short_ma'] < signals_df['long_ma']) & (signals_df['prev_short_ma'] >= signals_df['prev_long_ma']), 'signal'] = -1
        
        return signals_df

    def execute_trade(self, signal, symbol, price_at_signal=None): # price_at_signal for live, not used here yet
        # This method is for LIVE TRADING. Backtester will have its own logic.
        if not self.ib_client or not hasattr(self.ib_client, 'nextorderId') or self.ib_client.nextorderId is None:
            print("IB client not ready for live trading.")
            return

        contract = self.trade_utils.stock_order(symbol)
        action_taken = False
        
        if signal == 1 and self.current_position <= 0:
            if self.current_position == -1: # Close short
                order = self.trade_utils.get_order(action="BUY", quantity=self.quantity, order_type="MKT")
                order.orderId = self.ib_client.nextorderId
                self.ib_client.nextorderId +=1
                self.ib_client.placeOrder(order.orderId, contract, order)
                print(f"Live: Closing short position for {symbol}")
            
            order = self.trade_utils.get_order(action="BUY", quantity=self.quantity, order_type="MKT")
            order.orderId = self.ib_client.nextorderId
            self.ib_client.nextorderId +=1
            self.ib_client.placeOrder(order.orderId, contract, order)
            self.current_position = 1
            print(f"Live: Executed BUY for {self.quantity} of {symbol}. New Position: LONG")
            action_taken = True

        elif signal == -1 and self.current_position >= 0:
            if self.current_position == 1: # Close long
                order = self.trade_utils.get_order(action="SELL", quantity=self.quantity, order_type="MKT")
                order.orderId = self.ib_client.nextorderId
                self.ib_client.nextorderId +=1
                self.ib_client.placeOrder(order.orderId, contract, order)
                print(f"Live: Closing long position for {symbol}")

            # Assuming strategy is long-only for now, so -1 signal means exit to flat
            self.current_position = 0 
            print(f"Live: Exited position for {symbol}. New Position: FLAT")
            action_taken = True
            # If shorting is allowed:
            # order = self.trade_utils.get_order(action="SELL", quantity=self.quantity, order_type="MKT")
            # self.ib_client.placeOrder(self.ib_client.nextorderId, contract, order)
            # self.ib_client.nextorderId +=1
            # self.current_position = -1
            # print(f"Live: Executed SELL (short) for {self.quantity} of {symbol}. New Position: SHORT")

        if not action_taken:
            print(f"Live: No trade action for signal {signal}, current position {self.current_position}.")

    def run_strategy(self, symbol):
        print(f"Running LIVE strategy {self.strategy_name} for {symbol}")
        if not self.ib_client or not hasattr(self.ib_client, 'nextorderId') or self.ib_client.nextorderId is None:
            print("IB client not ready for trading. Cannot run live strategy.")
            return

        # For live, you'd typically fetch data, then subscribe to updates or run on a schedule
        # This is a simplified one-shot for demonstration
        historical_df = self.fetch_historical_data(symbol, duration_str="2 D", bar_size_setting="5 mins")
        if historical_df.empty:
            print(f"Could not run live strategy for {symbol} due to lack of initial data.")
            return

        signals_df = self.calculate_signals(historical_df)
        if signals_df.empty or 'signal' not in signals_df.columns or signals_df['signal'].empty:
            print(f"Could not calculate signals for {symbol} for live strategy.")
            return
            
        latest_signal = signals_df['signal'].iloc[-1]
        price_at_signal = signals_df['Close'].iloc[-1] # Use close of the bar that generated signal
        print(f"Live: Latest signal for {symbol} on {signals_df.index[-1]} is {latest_signal} at price {price_at_signal}")
        
        self.execute_trade(latest_signal, symbol, price_at_signal=price_at_signal)
        print(f"Live strategy execution attempt for {symbol} complete.")

    def backtest_strategy(self, symbol, start_date_str, end_date_str, bar_size_setting="1 day", initial_capital=100000, shares_per_trade=10):
        print(f"\n--- Starting Backtest for {symbol} ---")
        print(f"Period: {start_date_str} to {end_date_str}, Bar Size: {bar_size_setting}")
        print(f"Initial Capital: ${initial_capital}, Shares per Trade: {shares_per_trade}")
        print(f"MA Periods: Short={self.short_ma_period}, Long={self.long_ma_period}")
        print(f"Slippage per share per trade: ${self.slippage_per_trade:.2f}")

        # --- Data Fetching/Loading ---
        hist_data_df = pd.DataFrame()
        if self.ib_client and hasattr(self.ib_client, 'isConnected') and self.ib_client.isConnected():
            print("Attempting to fetch historical data via IBKR for backtest...")
            # Calculate duration for IBKR. This is tricky. For simplicity, if duration is long, this might fail or be slow.
            # IB's reqHistoricalData `durationStr` is more suitable for shorter periods.
            # `endDateTime` should be in 'YYYYMMDD HH:MM:SS [TZ]' format.
            # For longer backtests, it's better to query year by year or use pre-downloaded data.
            # For this example, let's try to fetch data using endDateTime and a calculated duration.
            try:
                start_dt = datetime.datetime.strptime(start_date_str, "%Y%m%d")
                end_dt = datetime.datetime.strptime(end_date_str, "%Y%m%d")
                delta = end_dt - start_dt
                
                if delta.days < 0:
                    print(f"Error: Start date {start_date_str} is after end date {end_date_str}.")
                    return None, None

                # Adjust duration string based on bar size for IB sanity
                # This logic is simplified. IB has specific rules for duration vs bar size.
                duration_days_for_ib = delta.days + 1 # Include the end date

                if bar_size_setting == "1 day":
                    # Max duration for daily bars is often 1 year or 365 D
                     if duration_days_for_ib > 360: # Keep it under typical limits
                         print(f"Warning: Requested duration {duration_days_for_ib} D for daily data is long. Fetching up to ~1 year chunks might be more robust if this fails.")
                         # For simplicity, we'll try one shot. A real implementation would loop.
                # Add more sophisticated duration calculation if needed for other bar sizes.
                
                ib_duration_str = f"{duration_days_for_ib} D"
                ib_end_date_time_str = end_date_str + " 23:59:59" # End of the last day

                hist_data_df = self.fetch_historical_data(symbol, duration_str=ib_duration_str, bar_size_setting=bar_size_setting, end_date_time_str=ib_end_date_time_str)
            except Exception as e:
                print(f"Error fetching data via IB for backtest: {e}. Falling back to dummy data if enabled.")
        
        if hist_data_df.empty:
            print("Failed to fetch data from IB or IB client not available/connected. Using dummy data for backtest.")
            # Create dummy data for testing if no client or fetch failed
            dates = pd.date_range(start=start_date_str, end=end_date_str, freq='B') # Business days
            if dates.empty:
                if pd.to_datetime(start_date_str) > pd.to_datetime(end_date_str):
                    print(f"Start date {start_date_str} is after end date {end_date_str}. No data to backtest.")
                    return None, None
                else: # if start and end are same or non-business day
                    dates = pd.DatetimeIndex([pd.to_datetime(start_date_str)])
            
            if len(dates) == 0:
                 print ("No dates generated for the given range. Cannot proceed with backtest.")
                 return None, None
            
            # Generate somewhat realistic price series
            price_volatility = 0.01 # Daily volatility
            log_returns = np.random.normal(loc=0.0005, scale=price_volatility, size=len(dates)) # Small positive drift
            initial_price = 100 + np.random.randint(-10,10)
            close_prices = initial_price * np.exp(log_returns.cumsum())
            
            hist_data_df = pd.DataFrame(data={'Close': close_prices}, index=dates)
            hist_data_df.index.name = 'DateTime'
            print(f"Generated {len(hist_data_df)} bars of dummy data for {symbol}.")


        if hist_data_df.empty:
            print(f"Could not fetch or load historical data for {symbol} for backtesting period.")
            return None, None

        signals_df = self.calculate_signals(hist_data_df)
        if signals_df.empty or 'signal' not in signals_df.columns:
            print("Failed to calculate signals for backtesting.")
            return None, None

        # Backtesting loop
        position = 0  # 0: flat, 1: long (shorting not implemented in this version)
        equity = initial_capital
        entry_price = 0
        trades = []
        portfolio_history = [{'date': signals_df.index[0] - pd.Timedelta(days=1), 'equity': initial_capital}] # Start with initial capital

        for i, row in signals_df.iterrows():
            current_price = row['Close'] 
            signal = row['signal']

            # If price is NaN, skip this bar (can happen with real data)
            if pd.isna(current_price):
                # Update portfolio history with previous equity if price is NaN
                portfolio_history.append({'date': i, 'equity': portfolio_history[-1]['equity']})
                continue

            buy_price_with_slippage = current_price + self.slippage_per_trade
            sell_price_with_slippage = current_price - self.slippage_per_trade

            # Decision making based on signal
            if position == 0: # Currently flat
                if signal == 1: # Buy signal
                    position = 1
                    entry_price = buy_price_with_slippage
                    cost_of_trade = shares_per_trade * entry_price
                    equity -= cost_of_trade 
                    trades.append({'date': i, 'type': 'BUY', 'price': entry_price, 'shares': shares_per_trade, 'pnl': 0, 'cost': cost_of_trade})
                    print(f"{i}: BUY {shares_per_trade} {symbol} at {entry_price:.2f}")
            
            elif position == 1: # Currently long
                # Exit conditions:
                # 1. Opposite signal (-1)
                # 2. (Optional) End of data holding period if still in position (handled after loop)
                if signal == -1: # Sell signal to exit long
                    position = 0
                    exit_price = sell_price_with_slippage
                    proceeds_from_trade = shares_per_trade * exit_price
                    pnl = (exit_price - entry_price) * shares_per_trade 
                    equity += proceeds_from_trade
                    # Find the corresponding BUY trade to update its PnL
                    for trade in reversed(trades):
                        if trade['type'] == 'BUY' and trade['pnl'] == 0: # Naive link, assumes one open trade
                            trade['pnl'] = pnl # This PnL is for this specific buy-sell pair
                            break
                    trades.append({'date': i, 'type': 'SELL', 'price': exit_price, 'shares': shares_per_trade, 'pnl': pnl, 'proceeds': proceeds_from_trade}) # This PnL is redundant here if also in BUY
                    print(f"{i}: SELL {shares_per_trade} {symbol} at {exit_price:.2f}, P&L: {pnl:.2f}")
                    entry_price = 0 
            
            # Mark-to-market portfolio value for this bar
            current_holding_value = 0
            if position == 1: # If long
                current_holding_value = shares_per_trade * current_price # Value of shares held
            
            current_total_equity = equity + current_holding_value
            portfolio_history.append({'date': i, 'equity': current_total_equity })

        # If still in position at the end of the backtest, mark-to-market the final P&L
        if position == 1:
            final_price_eod = signals_df['Close'].iloc[-1]
            # Apply slippage for this hypothetical EOD sell
            final_exit_price_with_slippage = final_price_eod - self.slippage_per_trade
            
            pnl_eod = (final_exit_price_with_slippage - entry_price) * shares_per_trade
            equity += shares_per_trade * final_exit_price_with_slippage # Add value of selling shares
            
            # Update PnL for the last open BUY trade
            for trade in reversed(trades):
                if trade['type'] == 'BUY' and trade['pnl'] == 0:
                     trade['pnl'] = pnl_eod
                     break
            trades.append({'date': signals_df.index[-1], 'type': 'SELL_EOD', 'price': final_exit_price_with_slippage, 'shares': shares_per_trade, 'pnl': pnl_eod})
            print(f"End of backtest ({signals_df.index[-1]}): Still LONG. Liquidating position at {final_exit_price_with_slippage:.2f}. P&L for final trade: {pnl_eod:.2f}")
            # Update the last portfolio equity point
            portfolio_history[-1]['equity'] = equity # Final equity after closing all positions

        # --- Performance Metrics ---
        trades_df = pd.DataFrame(trades)
        portfolio_df = pd.DataFrame(portfolio_history).set_index('date')

        print("\n--- Backtest Results ---")
        final_equity = portfolio_df['equity'].iloc[-1] if not portfolio_df.empty else initial_capital
        total_net_pnl = final_equity - initial_capital
        
        print(f"Initial Portfolio Value: ${initial_capital:.2f}")
        print(f"Final Portfolio Value: ${final_equity:.2f}")
        print(f"Total Net P&L: ${total_net_pnl:.2f}")

        if not trades_df.empty and 'pnl' in trades_df.columns:
            # Consider only P&L from closed trades (SELL or SELL_EOD)
            closed_trades_pnl = trades_df[trades_df['type'].str.contains('SELL')]['pnl']
            num_round_trip_trades = len(closed_trades_pnl)

            if num_round_trip_trades > 0:
                winning_trades_count = closed_trades_pnl[closed_trades_pnl > 0].count()
                losing_trades_count = closed_trades_pnl[closed_trades_pnl < 0].count()
                
                win_rate = (winning_trades_count / num_round_trip_trades) * 100 if num_round_trip_trades > 0 else 0
                avg_win = closed_trades_pnl[closed_trades_pnl > 0].mean() if winning_trades_count > 0 else 0
                avg_loss_val = closed_trades_pnl[closed_trades_pnl < 0].mean() if losing_trades_count > 0 else 0
                avg_loss_abs = abs(avg_loss_val)
                
                gross_profit = closed_trades_pnl[closed_trades_pnl > 0].sum()
                gross_loss = abs(closed_trades_pnl[closed_trades_pnl < 0].sum())
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

                print(f"Number of Trades (Round Trips): {num_round_trip_trades}")
                print(f"Win Rate: {win_rate:.2f}%")
                print(f"Average Winning Trade: ${avg_win:.2f}")
                print(f"Average Losing Trade: ${avg_loss_abs:.2f}")
                print(f"Profit Factor (Gross Profit / Gross Loss): {profit_factor:.2f}")
                if avg_loss_abs > 0:
                    print(f"Average Win/Loss Ratio: {avg_win / avg_loss_abs:.2f}")
                else:
                    print("Average Win/Loss Ratio: N/A (no losses)")

                # Sharpe Ratio calculation based on portfolio daily returns
                if not portfolio_df.empty and len(portfolio_df) > 1:
                    portfolio_df['daily_return'] = portfolio_df['equity'].pct_change().fillna(0)
                    # Exclude initial zero return if it's the first entry from pct_change on first value
                    valid_returns = portfolio_df['daily_return'][1:] if portfolio_df['daily_return'].iloc[0] == 0 and len(portfolio_df['daily_return']) > 1 else portfolio_df['daily_return']
                    
                    if len(valid_returns) > 1 and valid_returns.std() != 0 :
                        # Annualization factor depends on data frequency (bar_size_setting)
                        annualization_factor = 252 # Default for daily data
                        if 'min' in bar_size_setting:
                            minutes_per_bar = int(bar_size_setting.split()[0]) if bar_size_setting.split()[0].isdigit() else 1
                            annualization_factor = 252 * (6.5 * 60 / minutes_per_bar) # Assuming 6.5 trading hours/day
                        elif 'hour' in bar_size_setting:
                            hours_per_bar = int(bar_size_setting.split()[0]) if bar_size_setting.split()[0].isdigit() else 1
                            annualization_factor = 252 * (6.5 / hours_per_bar)
                        
                        sharpe_ratio = (valid_returns.mean() / valid_returns.std()) * np.sqrt(annualization_factor)
                        print(f"Sharpe Ratio (annualized, risk-free rate=0): {sharpe_ratio:.2f}")
                    else:
                        print("Sharpe Ratio: N/A (Std Dev of returns is 0 or not enough data points for returns)")
                else:
                    print("Sharpe Ratio: N/A (Not enough portfolio history for calculation)")
            else:
                print("No round-trip trades were made during the backtest period.")
        else:
            print("No trades were executed, or P&L data is missing.")
        
        print("--- Backtest End ---")
        return trades_df, portfolio_df


# Main execution block
if __name__ == "__main__":
    # --- Configuration ---
    run_live_trading = False # Set to True to run live trading example
    run_backtest = True    # Set to True to run backtesting example
    
    ibkr_connection_details = {'host': '127.0.0.1', 'port': 7497, 'clientId': 131} # Ensure unique clientId
    
    # --- Setup IBKR Connection (if needed for live or data fetching for backtest) ---
    app = None # IBapi instance
    api_thread = None

    # Import trade_utils here, as it's used by the strategy instance
    import trade_utils 

    if run_live_trading or (run_backtest and True): # True: means backtest might use IB for data
        app = IBapi()
        try:
            app.connect(ibkr_connection_details['host'], ibkr_connection_details['port'], clientId=ibkr_connection_details['clientId'])
            print("Attempting to connect to IBKR...")
            # Start the API processing thread
            api_thread = threading.Thread(target=lambda: app.run(), daemon=True)
            api_thread.start()
            
            # Wait for connection to establish and nextValidId to be processed
            # A more robust check would be app.isConnected() and hasattr(app, 'nextorderId')
            # This requires EWrapper.nextValidId to set a flag or event.
            # For simplicity, using a time delay and then checking attributes.
            print("Waiting for IBKR connection and nextValidId...")
            connection_timeout = 10 # seconds
            start_wait_time = time.time()
            while not (hasattr(app, 'nextorderId') and app.nextorderId is not None):
                time.sleep(0.5)
                if time.time() - start_wait_time > connection_timeout:
                    print("IBKR connection timeout or nextValidId not received.")
                    break
            
            if hasattr(app, 'nextorderId') and app.nextorderId is not None:
                print(f"Successfully connected to IBKR. Next Order ID: {app.nextorderId}")
            else:
                print("Failed to connect to IBKR or obtain nextValidId. Live trading and IB-data-dependent backtesting might fail.")
                if app and hasattr(app, 'disconnect') and callable(app.disconnect):
                    app.disconnect() # Ensure disconnect is called if connection failed partially
                app = None # Set app to None if connection failed

        except Exception as e:
            print(f"Exception during IBKR connection: {e}")
            app = None # Ensure app is None if any exception occurs

    # --- Strategy Instantiation ---
    # Pass 'app' if available and connected, else None. Pass trade_utils module.
    strategy = IntradayStrategy(ib_client=app, trade_utils_module=trade_utils)

    # --- Execute Live Strategy Example ---
    if run_live_trading and app and hasattr(app, 'nextorderId') and app.nextorderId is not None:
        print("\n--- Running Live Strategy Example ---")
        try:
            strategy.run_strategy(symbol="AAPL") # Example: run for Apple
        except Exception as e:
            print(f"An error occurred during live strategy execution: {e}")
    elif run_live_trading:
        print("\n--- Live Trading skipped (IBKR connection issue or run_live_trading=False) ---")

    # --- Execute Backtesting Example ---
    if run_backtest:
        print("\n--- Running Backtesting Example ---")
        try:
            trades_log, portfolio_over_time = strategy.backtest_strategy(
                symbol="MSFT", 
                start_date_str="20230101", # YYYYMMDD
                end_date_str="20231231",   # YYYYMMDD
                bar_size_setting="1 day",  # e.g., "1 day", "1 hour", "30 mins", "5 mins", "1 min"
                initial_capital=100000,
                shares_per_trade=50
            )
            if trades_log is not None and not trades_log.empty:
                print("\nBacktest Trades Log (first 5 and last 5):")
                print(trades_log.head())
                print("...")
                print(trades_log.tail())

            if portfolio_over_time is not None and not portfolio_over_time.empty:
                 print("\nPortfolio Value Over Time (first 5 and last 5):")
                 print(portfolio_over_time.head())
                 print("...")
                 print(portfolio_over_time.tail())
                # Optionally plot: portfolio_over_time['equity'].plot(title=f"Portfolio Equity for {strategy.symbol}")
                # import matplotlib.pyplot as plt
                # plt.show()

        except Exception as e:
            print(f"An error occurred during backtesting: {e}")
            import traceback
            traceback.print_exc()

    # --- Disconnection (if app instance exists and was connected) ---
    if app and hasattr(app, 'disconnect') and callable(app.disconnect):
        # Check if isConnected exists and is True, or if no isConnected but thread is alive.
        is_connected_flag = hasattr(app, 'isConnected') and app.isConnected()
        if is_connected_flag or (api_thread and api_thread.is_alive()):
            print("\nDisconnecting from IBKR.")
            app.disconnect()
            # Give some time for disconnection to complete
            if api_thread and api_thread.is_alive():
                time.sleep(1) # wait for disconnect to process
            print("Disconnected from IBKR.")
        else:
            print("\nIBKR client was not connected or already disconnected.")
    else:
        print("\nIBKR client instance not created or does not support disconnect.")
```
