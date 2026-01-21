import pandas as pd
import numpy as np
import sys
import os

# -----------------------------
# Path setup
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# -----------------------------
# Core pipeline imports
# -----------------------------
from src.data.data_loader import download_price_data
from src.data.preprocessing import compute_log_returns
from src.features.feature_engineering import (
    compute_rolling_volatility,
    build_hmm_features,
    scale_features
)
from src.models.hmm_regime import MarketRegimeHMM
from src.models.portfolio_optimization import mean_variance_optimization
from src.risk.risk_metrics import (
    sharpe_ratio,
    max_drawdown,
    value_at_risk,
    expected_shortfall
)

def run_pipeline(
    tickers: list,
    start_date: str,
    end_date: str,
    initial_capital: float = 1000.0,
    n_regimes: int = 2,
    risk_aversion: float = 1.0,
    prompt_template: str | None = None,
    enable_agent: bool = False,
    regime_descriptions: dict | None = None
) -> dict:
    """
    End-to-end regime-switching portfolio pipeline using an event-driven,
    buy-and-hold backtesting strategy.
    """

    # --------------------------------------------------
    # 1️⃣ Load price data (no change)
    # --------------------------------------------------
    prices = download_price_data(tickers, start_date, end_date)
    if prices.empty:
        raise ValueError("No price data downloaded. Check tickers or date range.")

    # --------------------------------------------------
    # 2️⃣ Compute log returns (no change)
    # --------------------------------------------------
    returns = compute_log_returns(prices)

    # --------------------------------------------------
    # 3️⃣ Feature engineering (no change)
    # --------------------------------------------------
    asset_returns = returns.copy()
    volatility = compute_rolling_volatility(asset_returns)
    features = build_hmm_features(asset_returns, volatility)
    X, scaler = scale_features(features)
    
    # Align prices with features and returns for the backtest loop
    aligned_prices = prices.loc[features.index]
    aligned_returns = returns.loc[features.index]

    # --------------------------------------------------
    # 4️⃣ NEW: Event-Driven Backtest Initialization
    # --------------------------------------------------
    cash = initial_capital
    holdings = {ticker: 0.0 for ticker in tickers}  # Stores number of shares
    
    # History tracking
    portfolio_history = []
    last_regime = -1
    min_obs_for_train = 252
    
    # To be populated by the loop
    regime_series = pd.Series(index=aligned_prices.index, dtype=int)

    # --------------------------------------------------
    # 5️⃣ NEW: Stateful, Event-Driven Backtesting Loop
    # --------------------------------------------------
    for t_idx, t in enumerate(aligned_prices.index):
        current_prices = aligned_prices.loc[t]

        # --- A. Calculate current portfolio value ---
        current_holdings_value = sum(holdings[asset] * current_prices[asset] for asset in tickers)
        total_value = current_holdings_value + cash
        
        # Record daily history
        daily_snapshot = {'total_value': total_value, 'cash': cash}
        daily_snapshot.update({f'{asset}_shares': holdings[asset] for asset in tickers})
        portfolio_history.append(daily_snapshot)

        # --- B. Train HMM & Predict Regime ---
        if t_idx < min_obs_for_train:
            regime_series.loc[t] = -1
            last_regime = -1
            continue

        start_idx = 0
        X_train = X[start_idx:t_idx]
        train_returns = aligned_returns.iloc[start_idx:t_idx]
        
        hmm = MarketRegimeHMM(max_states=n_regimes)
        fit_successful = hmm.fit(X_train)

        if not fit_successful:
            current_regime = -1
        else:
            current_regime = hmm.predict_regimes(X[t_idx:t_idx+1])[0]
        
        regime_series.loc[t] = current_regime
        
        # --- C. The Event: Act on Regime Change ---
        if current_regime != last_regime and current_regime != -1:
            # --- i. Get Target Portfolio for the new regime ---
            train_regimes = hmm.predict_regimes(X_train)
            regime_mask = train_regimes == current_regime
            r_returns = train_returns.iloc[regime_mask]

            target_weights = pd.Series(0.0, index=tickers) # Default to 0
            if len(r_returns) >= 30:
                mu = r_returns.mean() * 252
                cov = r_returns.cov() * 252
                
                # Note: The optimizer needs returns without CASH, but we pass full tickers list
                weights_array = mean_variance_optimization(mu.values, cov.values, risk_aversion)
                target_weights = pd.Series(weights_array, index=asset_returns.columns)

            # --- ii. Liquidate / Rebalance ---
            for asset in tickers:
                target_weight = target_weights.get(asset, 0)
                target_dollar_value = total_value * target_weight
                current_dollar_value = holdings[asset] * current_prices[asset]
                
                # We sell if target value is less than current, buy if more
                dollar_change = target_dollar_value - current_dollar_value
                
                if dollar_change != 0:
                    shares_to_trade = dollar_change / current_prices[asset]
                    
                    # Update holdings and cash
                    holdings[asset] += shares_to_trade
                    cash -= dollar_change # Add cash if selling, subtract if buying

        # --- D. Update State for Next Iteration ---
        last_regime = current_regime

    # --------------------------------------------------
    # 6️⃣ NEW: Post-Loop Processing
    # --------------------------------------------------
    history_df = pd.DataFrame(portfolio_history, index=aligned_prices.index)
    portfolio_value = history_df['total_value']
    portfolio_returns = portfolio_value.pct_change().fillna(0)
    
    # Create historical weights df from history
    weights_history_list = []
    for t in history_df.index:
        current_prices = aligned_prices.loc[t]
        total_val = history_df.loc[t, 'total_value']
        weights = {asset: (history_df.loc[t, f'{asset}_shares'] * current_prices[asset]) / total_val for asset in tickers}
        weights['CASH'] = history_df.loc[t, 'cash'] / total_val
        weights_history_list.append(pd.Series(weights, name=t))
        
    historical_weights_df = pd.DataFrame(weights_history_list).T
    historical_weights_df.index.name = "Asset"
    historical_weights_df.columns.name = "Date"

    # --------------------------------------------------
    # 7️⃣ Compute Final Risk Metrics (no change)
    # --------------------------------------------------
    metrics = {
        "sharpe_ratio": sharpe_ratio(portfolio_returns),
        "max_drawdown": max_drawdown(portfolio_value / initial_capital),
        "var_95": value_at_risk(portfolio_returns, 0.05),
        "expected_shortfall": expected_shortfall(portfolio_returns, 0.05),
        "annualized_volatility": portfolio_returns.std() * np.sqrt(252)
    }

    # --------------------------------------------------
    # 8️⃣ Optional LLM-based interpretation (Adapted)
    # --------------------------------------------------
    commentary = "Agent disabled. Quantitative results only."
    latest_regime = int(regime_series.iloc[-1]) if not regime_series.empty else -1
    
    final_weights_series = historical_weights_df.iloc[:, -1]

    if enable_agent and prompt_template is not None:
        from src.agents.regime_interpretation_agent import generate_risk_commentary
        
        current_regime_name = "Undefined"
        if latest_regime != -1 and regime_descriptions is not None:
            current_regime_name = regime_descriptions.get(latest_regime, f"Regime {latest_regime}")

        final_value = portfolio_value.iloc[-1]
        var_95_dollars = abs(metrics['var_95'] * final_value)

        llm_payload = {
            "current_regime_name": current_regime_name,
            "sharpe_ratio": f"{metrics['sharpe_ratio']:.2f}",
            "max_drawdown": f"{metrics['max_drawdown']:.2%}",
            "var_95_dollars": f"${var_95_dollars:,.2f}",
            "recommended_weights": final_weights_series.to_dict(),
            "regime_probability": "N/A" # Regime probability is not easily available in this new structure
        }
        commentary = generate_risk_commentary(llm_payload, prompt_template)

    # --------------------------------------------------
    # 9️⃣ Return results (Adapted)
    # --------------------------------------------------
    return {
        "prices": prices,
        "returns": aligned_returns,
        "portfolio_returns": portfolio_returns,
        "portfolio_value": portfolio_value,
        "features": features,
        "weights": final_weights_series,
        "historical_weights": historical_weights_df,
        "risk_metrics": metrics,
        "regime_series": regime_series,
        "llm_commentary": commentary
    }