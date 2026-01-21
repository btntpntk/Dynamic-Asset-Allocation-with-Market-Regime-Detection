import pandas as pd
from sklearn.preprocessing import StandardScaler

def compute_rolling_volatility(
    returns: pd.DataFrame,
    window: int = 21
) -> pd.DataFrame:
    """
    Compute rolling volatility for each asset
    """
    return returns.rolling(window).std()


def build_hmm_features(
    returns: pd.DataFrame,
    volatility: pd.DataFrame
) -> pd.DataFrame:
    """
    Construct feature matrix for HMM
    """
    features = pd.concat(
        [returns.mean(axis=1), volatility.mean(axis=1)],
        axis=1
    )
    features.columns = ["return", "volatility"]
    return features.dropna()


def scale_features(features: pd.DataFrame):
    """
    Standardize features for Gaussian HMM
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return scaled, scaler
