import numpy as np
import pandas as pd
from scipy.stats import norm

def annualized_volatility(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(252)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, trading_days_per_year: int = 252) -> float:
    """
    Calculates the annualized Sharpe ratio of a returns stream.
    """
    annualized_excess_return = (returns.mean() * trading_days_per_year) - risk_free_rate
    annualized_volatility = returns.std() * np.sqrt(trading_days_per_year)
    
    if annualized_volatility == 0:
        return 0.0 # Avoid division by zero
        
    annualized_sharpe = annualized_excess_return / annualized_volatility
    return annualized_sharpe


def max_drawdown(cumulative_returns: pd.Series) -> float:
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()


def value_at_risk(returns: pd.Series, alpha: float = 0.05) -> float:
    return np.percentile(returns, alpha * 100)


def expected_shortfall(returns: pd.Series, alpha: float = 0.05) -> float:
    var = value_at_risk(returns, alpha)
    return returns[returns <= var].mean()
