import pandas as pd
import numpy as np

def compute_log_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns from price series
    """
    returns = np.log(price_df / price_df.shift(1))
    return returns.dropna()


def save_processed_data(df: pd.DataFrame, path: str) -> None:
    """
    Save processed dataset
    """
    df.to_csv(path)
