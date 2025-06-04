from ibapi.wrapper import EWrapper
from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.order import Order
import threading
import time
import pandas as pd

class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = []
        self.contract_details = {}
        self.bardata = {}

    def tickPrice(self, reqId, tickType, price, attrib):
        if tickType == 2 and reqId == 1:
            print('The current ask price is: ', price)

    def historicalData(self, reqId, bar):
        print(f'Time: {bar.date} Close: {bar.close}')
        self.data.append([bar.date, bar.close])

    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.nextorderId = orderId
        print('The next valid order id is: ', self.nextorderId)

    def orderStatus(self, orderId, status, filled, remaining, avgFullPrice, permId, parentId, lastFillPrice, clientId,
                    whyHeld, mktCapPrice):
        print('orderStatus - orderid:', orderId, 'status:', status, 'filled', filled, 'remaining', remaining,
              'lastFillPrice', lastFillPrice)

    def openOrder(self, orderId, contract, order, orderState):
        print('openOrder id:', orderId, contract.symbol, contract.secType, '@', contract.exchange, ':', order.action,
              order.orderType, order.totalQuantity, orderState.status)

    def execDetails(self, reqId, contract, execution):
        print('Order Executed: ', reqId, contract.symbol, contract.secType, contract.currency, execution.execId,
              execution.orderId, execution.shares, execution.lastLiquidity)

    def contractDetails(self, reqId: int, contractDetails):
        self.contract_details[reqId] = contractDetails

    def get_contract_details(self, reqId, contract):
        self.contract_details[reqId] = None
        self.reqContractDetails(reqId, contract)

        # Wait for contract details
        for _ in range(50): # 5 seconds timeout
            if self.contract_details[reqId] is None:
                time.sleep(0.1)
            else:
                break
        else:
            raise Exception('Error getting contract details: Timeout')
        
        # Check if contract_details[reqId] is still None or if .contract is not available
        if self.contract_details[reqId] is None or not hasattr(self.contract_details[reqId], 'contract'):
             raise Exception('Error getting contract details: Details not found or contract attribute missing')

        return self.contract_details[reqId].contract

    def tickByTickAllLast(self, reqId, tickType, time_val, price, size, tickAttribLast, exchange, specialConditions):
        if tickType == 1: # TickType 1 indicates 'Last' tick
            # Convert Unix timestamp to datetime
            dt_object = pd.to_datetime(time_val, unit='s')
            if reqId not in self.bardata:
                 self.bardata[reqId] = pd.DataFrame(columns=['price']).set_index(pd.DatetimeIndex([]))
            self.bardata[reqId].loc[dt_object] = price


    def tick_df(self, reqId, contract):
        ''' custom function to init DataFrame and request Tick Data '''
        self.bardata[reqId] = pd.DataFrame(columns=['price'])
        self.bardata[reqId].set_index(pd.to_datetime([]), inplace=True) # Initialize with an empty DatetimeIndex
        self.reqTickByTickData(reqId, contract, "Last", 0, True)
        return self.bardata[reqId]


def run_loop(app_instance):
    app_instance.run()

# Example usage (optional, can be removed or commented out if not needed for library use)
if __name__ == "__main__":
    app = IBapi()
    app.connect('127.0.0.1', 7497, 123) # Connect to TWS or Gateway
    
    # Start the socket in a thread
    api_thread = threading.Thread(target=run_loop, args=(app,), daemon=True)
    api_thread.start()

    # Check if connection is established
    time.sleep(3) # Allow time for connection to establish and nextValidId to be processed

    if hasattr(app, 'nextorderId') and app.nextorderId is not None:
        print('Successfully connected to IBKR')
        print('Next valid order ID: ', app.nextorderId)
        
        # Example: Request contract details for SPY
        spy_contract = Contract()
        spy_contract.symbol = "SPY"
        spy_contract.secType = "STK"
        spy_contract.exchange = "SMART"
        spy_contract.currency = "USD"
        
        try:
            resolved_contract = app.get_contract_details(reqId=1, contract=spy_contract)
            print(f"Successfully retrieved contract details for {resolved_contract.symbol}")
            
            # Example: Request tick data
            # tick_data_df = app.tick_df(reqId=2, contract=resolved_contract)
            # print("Streaming tick data. Press Ctrl+C to stop.")
            # while True:
            #     time.sleep(1) # Keep main thread alive to stream data
            #     print(tick_data_df.tail())


        except Exception as e:
            print(f"An error occurred: {e}")

    else:
        print('Failed to connect to IBKR or obtain next valid order ID.')

    app.disconnect()
