"""Factor model utilities using toraniko."""
from __future__ import annotations

import polars as pl
from toraniko.styles import factor_mom
from toraniko.model import estimate_factor_returns

class FactorModel:
    """Wrapper for toraniko factor modeling functions."""

    @staticmethod
    def momentum_scores(returns_df: pl.DataFrame, trailing_days: int = 60) -> pl.DataFrame:
        """Calculate momentum factor scores using toraniko."""
        return factor_mom(returns_df, trailing_days=trailing_days)

    @staticmethod
    def factor_returns(
        returns_df: pl.DataFrame,
        mkt_cap_df: pl.DataFrame,
        sector_df: pl.DataFrame,
        style_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Estimate factor returns using toraniko."""
        return estimate_factor_returns(
            returns_df=returns_df,
            mkt_cap_df=mkt_cap_df,
            sector_df=sector_df,
            style_df=style_df,
        )
