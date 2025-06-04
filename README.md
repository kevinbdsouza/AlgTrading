# AlgTrading
Trying to find short term signals for intraday

## Note
* Look into Momentum strategies, Mean reverting strategies 
* Diversification of strategies helps minimize different types of risks. 

## Quantitative Analysis with Toraniko

This repository now includes the [toraniko](https://github.com/0xfdf/toraniko) library, a multi-factor equity risk model for quantitative trading.

### Dependencies

Toraniko and its dependencies (numpy, polars) have been added to `requirements.txt`. You can install them using:

```bash
pip install -r requirements.txt
```

### Example Usage

An example script demonstrating basic usage of toraniko for momentum factor calculation can be found in `docs/examples/toraniko_example.py`. This script uses simulated data and placeholder values for market capitalization and sector information to illustrate how to structure data and call toraniko's functions.

To run the example:
```bash
python docs/examples/toraniko_example.py
```

You can adapt this example to incorporate more sophisticated risk modeling and factor analysis into your trading strategies developed with the `tradingbot.py` framework.

The repository includes a lightweight wrapper `FactorModel` in `code/factor_model.py` that exposes toraniko's momentum scoring and factor return estimation functions for use within your own strategies or analysis scripts.
