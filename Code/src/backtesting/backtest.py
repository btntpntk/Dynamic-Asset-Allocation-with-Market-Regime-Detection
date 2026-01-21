import pandas as pd
import numpy as np

def compute_portfolio_returns(
    returns: pd.DataFrame,
    weights: pd.DataFrame
) -> pd.Series:
    """
    Compute portfolio returns from asset returns and weights
    """
    aligned = returns.loc[weights.index]
    return (aligned * weights).sum(axis=1)


def cumulative_returns(portfolio_returns: pd.Series) -> pd.Series:
    """
    Compute cumulative returns
    """
    return (1 + portfolio_returns).cumprod()
