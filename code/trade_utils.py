import pandas as pd
from ibapi.contract import Contract
from ibapi.order import Order
# Assuming IBapi is in ibkr_client.py in the same directory
# from ibkr_client import IBapi 
import time

def analysis(app_data):
    """
    Performs a simple moving average analysis on the provided data.
    Expects app_data to be a list of lists, e.g., [[timestamp, close_price], ...]
    """
    if not app_data:
        print("No data to analyze.")
        return None
    
    df = pd.DataFrame(app_data, columns=['DateTime', 'Close'])
    if 'Close' not in df.columns:
        print("DataFrame does not contain 'Close' column for analysis.")
        return df

    # Ensure 'Close' is numeric
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df.dropna(subset=['Close'], inplace=True)

    if len(df['Close']) >= 20:
        df['20SMA'] = df['Close'].rolling(20).mean()
    else:
        print("Not enough data points to calculate 20SMA.")
        df['20SMA'] = None # Or handle as appropriate
    return df


def stock_order(symbol, sec_type='STK', exchange='SMART', currency='USD'):
    contract = Contract()
    contract.symbol = symbol
    contract.secType = sec_type
    contract.exchange = exchange
    contract.currency = currency
    return contract


def options_order(symbol, expiry, strike, right_type, exchange='SMART', currency='USD', multiplier='100'):
    contract = Contract()
    contract.symbol = symbol
    contract.secType = 'OPT'
    contract.exchange = exchange
    contract.currency = currency
    contract.lastTradeDateOrContractMonth = expiry
    contract.strike = strike
    contract.right = right_type # 'P' for Put, 'C' for Call
    contract.multiplier = multiplier
    return contract


def get_order(action, quantity, order_type, lmt_price=None, aux_price=None):
    order = Order()
    order.action = action # "BUY" or "SELL"
    order.totalQuantity = quantity
    order.orderType = order_type # "LMT", "MKT", "STP", etc.
    
    if order_type == "LMT" and lmt_price is not None:
        order.lmtPrice = lmt_price
    elif order_type == "STP" and aux_price is not None:
        order.auxPrice = aux_price
    # Add other order types and their specific price fields as needed
    
    return order


def execute_stop_loss_trade(app, symbol='SPY', quantity=1, lmt_price=523.6, stop_loss_price=522.0, take_profit_price=525.0):
    if not hasattr(app, 'nextorderId') or app.nextorderId is None:
        print("IB API not connected or nextorderId not available.")
        return app

    # Create contract
    contract = stock_order(symbol)

    # Entry Order
    entry_order = get_order(action="BUY", quantity=quantity, order_type="LMT", lmt_price=lmt_price)
    entry_order.orderId = app.nextorderId
    app.nextorderId += 1
    entry_order.transmit = False # Transmit False for parent order in a bracket order

    # Stop Loss Order
    sl_order = get_order(action="SELL", quantity=quantity, order_type="STP", aux_price=stop_loss_price)
    sl_order.orderId = app.nextorderId
    app.nextorderId += 1
    sl_order.parentId = entry_order.orderId
    sl_order.transmit = False # Transmit False for child order if part of a larger group

    # Take Profit Order
    tp_order = get_order(action="SELL", quantity=quantity, order_type="LMT", lmt_price=take_profit_price)
    tp_order.orderId = app.nextorderId
    app.nextorderId += 1
    tp_order.parentId = entry_order.orderId
    tp_order.transmit = True # Transmit True for the last order in the bracket or if submitting individually after setup

    print(f"Placing orders for {symbol}: Entry LMT @ {lmt_price}, SL STP @ {stop_loss_price}, TP LMT @ {take_profit_price}")
    
    app.placeOrder(entry_order.orderId, contract, entry_order)
    app.placeOrder(sl_order.orderId, contract, sl_order)
    app.placeOrder(tp_order.orderId, contract, tp_order)
    
    print("Bracket order placed.")
    return app


def execute_options_trade(app, symbol, expiry, strike, right_type, quantity=1, action="BUY", order_type="MKT", lmt_price=None):
    if not hasattr(app, 'nextorderId') or app.nextorderId is None:
        print("IB API not connected or nextorderId not available.")
        return app

    contract = options_order(symbol=symbol, expiry=expiry, strike=strike, right_type=right_type)
    
    order = get_order(action=action, quantity=quantity, order_type=order_type, lmt_price=lmt_price)
    order.orderId = app.nextorderId
    app.nextorderId += 1
    order.transmit = True

    print(f"Placing options order: {action} {quantity} {symbol} {expiry} {strike}{right_type} @ {order_type} {lmt_price if lmt_price else ''}")
    app.placeOrder(order.orderId, contract, order)
    
    print("Options order placed.")
    return app

if __name__ == '__main__':
    # This section is for example usage and testing.
    # It requires a running IBapi instance (app).
    # For direct testing, you might need to mock the 'app' object or connect to TWS/Gateway.

    print("trade_utils.py executed. Contains utility functions for trading.")
    print("To test functions like execute_stop_loss_trade, an IBapi app instance is needed.")

    # Example of how you might set up a mock app for basic testing (won't actually place orders)
    class MockIBapi:
        def __init__(self):
            self.nextorderId = 1
            self.placed_orders = []

        def placeOrder(self, orderId, contract, order):
            print(f"Mock placing order: {orderId}, {contract.symbol}, {order.action}, {order.orderType}")
            self.placed_orders.append({'orderId': orderId, 'symbol': contract.symbol, 'action': order.action})

    mock_app = MockIBapi()

    print("\n--- Testing execute_stop_loss_trade ---")
    execute_stop_loss_trade(mock_app, symbol='AAPL', quantity=10, lmt_price=170.0, stop_loss_price=169.0, take_profit_price=172.0)
    
    print("\n--- Testing execute_options_trade ---")
    execute_options_trade(mock_app, symbol='MSFT', expiry='20241220', strike=400, right_type='C', quantity=2, action='BUY', order_type='LMT', lmt_price=5.50)

    print("\n--- Testing analysis function ---")
    sample_data = [['2023-01-01 10:00:00', 150.0], ['2023-01-01 10:05:00', 150.2]]
    # Add more data points for SMA calculation
    for i in range(20):
        sample_data.append([f'2023-01-01 10:{10+i*5:02d}:00', 150.0 + i * 0.1])
    
    analysis_df = analysis(sample_data)
    if analysis_df is not None:
        print(analysis_df.head())
        print(analysis_df.tail())
