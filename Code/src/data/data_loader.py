import yfinance as yf
import pandas as pd


def download_price_data(
    tickers: list,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Download close prices using yfinance.
    """
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        progress=False,
        interval="1wk"
    )

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]

    prices = prices.dropna()

    return prices

