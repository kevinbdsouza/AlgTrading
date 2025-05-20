# AlgTrading
Trying to find short term signals for intraday/HFT

## Github Promising
1. https://github.com/quantopian
2. https://github.com/quantopian/zipline

## Web 
1. https://www.quantconnect.com/
2. https://epchan.blogspot.com/
3. https://pyquantnews.com/code-for-a-trading-strategy-with-a-0-75-sharpe/
4. https://in.tradingview.com/

## Videos
1. https://www.youtube.com/watch?v=xfzGZB4HhEE
2. https://www.youtube.com/watch?v=UU4ZQF-X9jE
3. https://www.youtube.com/watch?v=qJv2Ii_l6JQ
4. https://www.youtube.com/watch?v=SEQbb8w7VTw
5. https://www.youtube.com/watch?v=QIUxPv5PJOY
6. https://www.youtube.com/watch?v=J_kzoZOxsZ0
7. https://www.youtube.com/watch?v=SEQbb8w7VTw
8. https://www.youtube.com/watch?v=c9OjEThuJjY
9. https://www.youtube.com/watch?v=9Y3yaoi9rUQ

## Note
* Look into Momentum strategies, Mean reverting strategies 
* Hard to account for natural disasters, market shocks, economic news. Diversification of strategies helps minimize different types of risks. 

## Quantitative Analysis with Toraniko

This repository now includes the [toraniko](https://github.com/0xfdf/toraniko) library, a multi-factor equity risk model for quantitative trading.

### Dependencies

Toraniko and its dependencies (numpy, polars) have been added to `requirements.txt`. You can install them using:

```bash
pip install -r requirements.txt
```

### Example Usage

An example script demonstrating basic usage of toraniko for momentum factor calculation can be found in `luchkata_training/toraniko_example.py`. This script uses simulated data and placeholder values for market capitalization and sector information to illustrate how to structure data and call toraniko's functions.

To run the example:
```bash
python luchkata_training/toraniko_example.py
```

You can adapt this example to incorporate more sophisticated risk modeling and factor analysis into your trading strategies developed with the `tradingbot.py` framework.
