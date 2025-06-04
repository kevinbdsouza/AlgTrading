# Import necessary libraries
import polars as pl
import numpy as np
from toraniko.styles import factor_mom
from toraniko.model import estimate_factor_returns

def main():
    # --- 1. Load Data ---
    print("Loading data...")
    try:
        # Load the simulated daily data
        # The CSV has a trailing comma in the header, so read_csv might infer an extra null column.
        # We'll select specific columns to avoid this.
        df = pl.read_csv("luchkata_training/simulated_daily_data.csv", columns=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- 2. Preprocess Data ---
    print("Preprocessing data...")
    # Rename columns
    df = df.rename({"Date": "date", "Close": "close_price"})

    # Add a placeholder symbol
    df = df.with_columns(pl.lit("SIMULATED_ASSET").alias("symbol"))

    # Convert 'date' column to polars Date type
    try:
        df = df.with_columns(pl.col("date").str.to_date(format="%Y-%m-%d")) # Adjust format if needed
    except Exception as e: # Handle cases where date parsing might fail for some rows
        print(f"Warning: Could not parse all dates. Error: {e}. Dropping null dates.")
        df = df.with_columns(pl.col("date").str.to_date(format="%Y-%m-%d", strict=False)).drop_nulls("date")

    # Calculate daily returns from 'close_price'
    # asset_returns = (current_price - previous_price) / previous_price
    # Sort by date to ensure correct pct_change calculation
    df = df.sort("date")
    df = df.with_columns(
        asset_returns = pl.col("close_price").pct_change().over("symbol") # Calculate pct_change per symbol
    )

    # Select only necessary columns and drop rows with null returns (typically the first row)
    returns_df = df.select(["date", "symbol", "asset_returns"]).drop_nulls("asset_returns")
    
    if returns_df.is_empty():
        print("Error: No data left after preprocessing returns. Check date parsing and returns calculation.")
        return

    print("Returns data preprocessed:")
    print(returns_df.head())

    # --- 3. Create Placeholder Data for Other Inputs ---
    print("Creating placeholder data...")

    # Get unique dates and symbol from returns_df for consistency
    unique_dates = returns_df.select("date").unique().sort("date")
    symbol_value = returns_df.select("symbol").unique().item() # Assuming single symbol

    # Placeholder Market Cap Data
    # Using a consistent number of rows based on unique_dates
    n_rows = len(unique_dates)
    mkt_cap_data = {
        "date": unique_dates["date"],
        "symbol": pl.Series("symbol", [symbol_value] * n_rows),
        "market_cap": pl.Series("market_cap", np.random.uniform(1e9, 100e9, size=n_rows))
    }
    mkt_cap_df = pl.DataFrame(mkt_cap_data)
    
    print("Placeholder Market Cap data:")
    print(mkt_cap_df.head())

    # Placeholder Sector Data
    # For a single asset, we can assign it to a general 'Market' sector or a specific one like 'Technology'.
    # The structure toraniko expects is 'date', 'symbol', and then one-hot encoded sector columns.
    # For a single asset and a single sector, it's a column of 1s for that sector.
    sector_data = {
        "date": unique_dates["date"],
        "symbol": pl.Series("symbol", [symbol_value] * n_rows),
        "Market": pl.Series("Market", [1.0] * n_rows), # Asset belongs to 'Market' sector
        # "Technology": pl.Series("Technology", [1.0] * n_rows) # Example if using 'Technology'
    }
    sector_df = pl.DataFrame(sector_data)
    
    print("Placeholder Sector data:")
    print(sector_df.head())
    
    # Ensure returns_df also only contains dates present in unique_dates (especially if there were date parsing issues)
    returns_df = returns_df.filter(pl.col("date").is_in(unique_dates["date"]))

    # --- 4. Calculate Momentum Factor Scores ---
    print("Calculating momentum factor scores...")
    try:
        # factor_mom expects columns: 'date', 'symbol', 'asset_returns'
        # It returns a DataFrame with 'date', 'symbol', and momentum factor scores.
        style_df = factor_mom(returns_df, trailing_days=60) # Using 60 days for momentum, common is ~252 for 1y
        style_df = style_df.rename({"factor_mom": "Momentum"}) # Renaming for clarity if needed
    except Exception as e:
        print(f"Error calculating momentum factor: {e}")
        # This can happen if there are not enough data points for the trailing_days window for some assets.
        # For this example, we'll create an empty placeholder if it fails
        print("Creating empty style_df due to error.")
        style_df = pl.DataFrame({
            "date": unique_dates["date"],
            "symbol": pl.Series("symbol", [symbol_value] * n_rows),
            "Momentum": pl.Series("Momentum", [0.0] * n_rows) # Placeholder momentum
        })


    print("Momentum Factor (Style) data:")
    print(style_df.head())

    # --- 5. Prepare Inputs for estimate_factor_returns ---
    print("Preparing inputs for estimate_factor_returns...")

    # Ensure all DataFrames are sorted by date and symbol
    # returns_df is already sorted by date, and has one symbol.
    # mkt_cap_df, sector_df, style_df were created with sorted unique_dates and single symbol.
    
    # Align dates: Use an inner join to ensure all dataframes have the same dates
    # Start with returns_df as the base for dates
    aligned_df = returns_df.join(mkt_cap_df, on=["date", "symbol"], how="inner")
    aligned_df = aligned_df.join(sector_df, on=["date", "symbol"], how="inner")
    aligned_df = aligned_df.join(style_df, on=["date", "symbol"], how="inner")

    if aligned_df.is_empty():
        print("Error: Data alignment resulted in an empty DataFrame. Check date consistency and join keys.")
        return
        
    final_returns_df = aligned_df.select(["date", "symbol", "asset_returns"])
    final_mkt_cap_df = aligned_df.select(["date", "symbol", "market_cap"])
    final_sector_df = aligned_df.select(["date", "symbol"] + [col for col in sector_df.columns if col not in ["date", "symbol"]])
    final_style_df = aligned_df.select(["date", "symbol"] + [col for col in style_df.columns if col not in ["date", "symbol"]])


    print("Aligned returns_df head:")
    print(final_returns_df.head())
    print("Aligned mkt_cap_df head:")
    print(final_mkt_cap_df.head())
    print("Aligned sector_df head:")
    print(final_sector_df.head())
    print("Aligned style_df head:")
    print(final_style_df.head())
    
    # --- 6. Run estimate_factor_returns ---
    print("Running estimate_factor_returns...")
    try:
        # The function expects specific column names and structures.
        # returns_df: ['date', 'symbol', 'asset_returns']
        # mkt_cap_df: ['date', 'symbol', 'market_cap']
        # sector_df: ['date', 'symbol', 'Sector1', 'Sector2', ...] (one-hot encoded)
        # style_df: ['date', 'symbol', 'StyleFactor1', 'StyleFactor2', ...]
        fac_df = estimate_factor_returns(
            returns_df=final_returns_df,
            mkt_cap_df=final_mkt_cap_df,
            sector_df=final_sector_df, # Must contain 'date', 'symbol', and then sector columns
            style_df=final_style_df    # Must contain 'date', 'symbol', and then style factor columns
        )
        
        # --- 7. Print Results ---
        print("Factor Returns (fac_df) head:")
        print(fac_df.head())

    except Exception as e:
        print(f"Error running estimate_factor_returns: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
