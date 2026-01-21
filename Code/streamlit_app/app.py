import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Compare Equity Bond Gold

# ----------------------------------------------------
# Fix Python path so `src` can be imported
# ----------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.run_pipeline import run_pipeline

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Regime-Aware Portfolio Optimization",
    layout="wide"
)

st.title("📊 Regime-Aware Portfolio Optimization Dashboard")

# =========================
# Sidebar Inputs
# =========================
st.sidebar.header("Configuration")

tickers_input = st.sidebar.text_input(
    label="Assets (comma separated)",
    # value="AAPL,MSFT,GOOGL,AMZN,TSLA"
    value="SPY,TLT,IEF,GLD,DBC"
)

initial_capital = st.sidebar.number_input(
    label="Initial Capital ($)",
    min_value=100.0,
    value=1000.0,
    step=100.0,
    format="%.2f"
)

start_date = st.sidebar.date_input(
    label="Start Date",
    value=pd.to_datetime("2011-01-01")
)

end_date = st.sidebar.date_input(
    label="End Date",
    value=pd.to_datetime("2025-12-31")
)

enable_llm_agent = st.sidebar.checkbox(
    label="Enable AI Portfolio Analyst (LLM)",
    value=True # Enable by default for demonstration
)

run_button = st.sidebar.button("Run Analysis")

# =========================
# Run Pipeline
# =========================
if run_button:
    with st.spinner("Running regime analysis & portfolio optimization..."):
        llm_prompt_template = None
        if enable_llm_agent:
            prompt_file_path = os.path.join(PROJECT_ROOT, "prompts", "enhanced_portfolio_analyst_prompt.txt")
            try:
                with open(prompt_file_path, "r") as f:
                    llm_prompt_template = f.read()
            except FileNotFoundError:
                st.error(f"Prompt file not found: {prompt_file_path}. Please ensure it exists.")
                llm_prompt_template = None
                enable_llm_agent = False # Disable agent if prompt is missing

        # Define REGIME_DESCRIPTIONS here so it's available for run_pipeline
        REGIME_DESCRIPTIONS_FOR_PIPELINE = {
            -1: "Undefined: Insufficient historical data for the model.",
            0: "Strong Bull / Low Volatility Regime.",
            1: "Moderate Growth / Transitional Regime.",
            2: "Correction / High Volatility Regime.",
            3: "Crisis / Bear Market Regime."
        }

        user_tickers = [t.strip() for t in tickers_input.split(",")]
        benchmark_ticker = 'SPY'
        
        # Ensure benchmark ticker is in the download list, without duplicates
        tickers_to_download = user_tickers.copy()
        if benchmark_ticker not in tickers_to_download:
            tickers_to_download.append(benchmark_ticker)
            
        results = run_pipeline(
            tickers=tickers_to_download, # Pass the extended list
            start_date=str(start_date),
            end_date=str(end_date),
            initial_capital=initial_capital,
            n_regimes=4,
            enable_agent=enable_llm_agent,
            prompt_template=llm_prompt_template,
            regime_descriptions=REGIME_DESCRIPTIONS_FOR_PIPELINE # Pass the descriptions to the pipeline
        )

    prices = results["prices"]
    historical_weights_df = results["historical_weights"]
    portfolio_value = results["portfolio_value"]
    weights = results["weights"]
    risk_metrics = results["risk_metrics"]
    regime_series = results["regime_series"]

    # =========================
    # KPI Metrics
    # =========================
    final_value = portfolio_value.iloc[-1]
    total_return_pct = (final_value / initial_capital) - 1
    total_pnl = final_value - initial_capital

    st.subheader("📈 Portfolio Performance")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric("Final Portfolio Value", f"${final_value:,.2f}")
    col2.metric("Total PnL", f"${total_pnl:,.2f}", delta=f"{total_return_pct:.2%}")
    col3.metric("Sharpe Ratio", f"{risk_metrics['sharpe_ratio']:.2f}")
    col4.metric("Max Drawdown", f"{risk_metrics['max_drawdown']:.2%}")
    col5.metric("VaR (95%)", f"{risk_metrics['var_95']:.2%}")
    col6.metric("Annualized Volatility", f"{risk_metrics['annualized_volatility']:.2%}")
    

    # =========================
    # Portfolio Weights Over Time (Stacked Bar Chart) & Portfolio Value Chart
    # =========================
    st.subheader("📊 Portfolio Weights & Value Over Time")

    col_a, col_b = st.columns(2)

    # Reshape historical_weights_df for Plotly Express
    # From wide format (Assets as rows, Dates as columns) to long format
    historical_weights_long = historical_weights_df.T.reset_index()
    historical_weights_long = historical_weights_long.melt(id_vars=['Date'], var_name='Asset', value_name='Weight')

    # Color mapping, ensuring CASH is consistently grey
    asset_list_for_colors = historical_weights_df.index.tolist()
    color_map = px.colors.qualitative.Plotly
    asset_colors = {asset: color_map[i % len(color_map)] for i, asset in enumerate(asset_list_for_colors)}
    asset_colors["CASH"] = "#CCCCCC" # Assign a neutral grey for cash

    fig_historical_weights = px.bar(
        historical_weights_long,
        x="Date",
        y="Weight",
        color="Asset",
        title="Asset Allocation Over Time",
        color_discrete_map=asset_colors,
    )
    fig_historical_weights.update_layout(yaxis_title="Portfolio Weight", xaxis_title="Date")
    fig_historical_weights.update_layout(yaxis_range=[0, 1]) # Weights sum to 1
    col_a.plotly_chart(fig_historical_weights, width='stretch')

    # --- Create combined DataFrame for plotting portfolio value and benchmark ---
    # 1. Normalize portfolio value
    normalized_portfolio = portfolio_value / initial_capital
    
    # Create a DataFrame for our strategy's performance
    plot_df = pd.DataFrame({"Strategy": normalized_portfolio})

    # 2. Add S&P 500 benchmark if available
    benchmark_ticker = 'SPY'
    if benchmark_ticker in prices.columns:
        # Align benchmark series to the portfolio's date index and drop any non-overlapping dates
        benchmark_prices = prices[benchmark_ticker].reindex(portfolio_value.index).dropna()
        if not benchmark_prices.empty:
            # Normalize the benchmark
            normalized_benchmark = benchmark_prices / benchmark_prices.iloc[0]
            plot_df[f'{benchmark_ticker} (Benchmark)'] = normalized_benchmark

    # 3. Plot both lines using the combined DataFrame
    fig_value = px.line(
        plot_df,
        title="Strategy Performance vs. Benchmark"
    )
    fig_value.update_layout(yaxis_title="Return", xaxis_title="Date")
    col_b.plotly_chart(fig_value, width='stretch')

    # =========================
    # Asset Prices
    # =========================
    st.subheader("💹 Asset Prices (Regime-Shaded)")

    # Base price chart
    fig_prices = px.line(
        prices,
        title="Asset Price History with Market Regimes",
        color_discrete_map=asset_colors
    )

    # Ensure alignment and correct data types
    regime_aligned = regime_series.reindex(prices.index).fillna(-1).astype(int)

    # Identify regime change points
    regime_changes = regime_aligned.ne(regime_aligned.shift()).cumsum()

    regime_blocks = (
        regime_aligned
        .groupby(regime_changes)
        .agg(
            start_date=lambda x: x.index.min(),
            end_date=lambda x: x.index.max(),
            regime=lambda x: x.iloc[0]
        )
    )

    # Regime colors
    REGIME_COLORS = {
        0: "rgba(0, 200, 0, 1)",
        1: "rgba(255, 165, 0, 1)",
        2: "rgba(255, 0, 0, 1)",
        3: "rgba(120, 0, 120, 1)",
        -1: "rgba(150, 150, 150, 1)" # Color for undefined regime
    }

    # Efficiently create a list of shapes to draw all at once
    shapes = []
    for _, row in regime_blocks.iterrows():
        shape = dict(
            type="rect",
            xref="x",
            yref="paper",  # Use 'paper' to span the entire y-axis
            x0=row["start_date"],
            x1=row["end_date"],
            y0=0,
            y1=1,
            fillcolor=REGIME_COLORS.get(row["regime"], "rgba(200,200,200,0.05)"),
            opacity=0.2,
            layer="below",
            line_width=0,
        )
        shapes.append(shape)

    # Update the figure layout with all shapes in a single call
    fig_prices.update_layout(shapes=shapes)

    st.plotly_chart(fig_prices, width='stretch')

    # =========================
    # Dynamic Regime Descriptions
    # =========================

    REGIME_DESCRIPTIONS = {
        -1: "Undefined: Insufficient historical data for the model.",
        0: "Strong Bull / Low Volatility",
        1: "Moderate Growth / Transitional",
        2: "Correction / High Volatility",
        3: "Crisis / Bear Market"
    }

    unique_regimes = sorted([r for r in regime_series.unique() if r != -1])

    if unique_regimes:
        st.subheader("🔍 Regime Descriptions")
        # Create one column per regime
        cols = st.columns(len(unique_regimes))
        for col, regime_id in zip(cols, unique_regimes):
            description = REGIME_DESCRIPTIONS.get(
                regime_id,
                "A distinct market regime was detected."
            )
            bg_color = REGIME_COLORS.get(regime_id, "rgba(200,200,200,0.15)")
            border_color = bg_color.replace("0.15", "0.8")

            with col:
                st.markdown(
                    f"""
                    <div style="
                        background-color: {bg_color};
                        padding: 8px;
                        border-left: 6px solid {border_color};
                        border-radius: 10px;
                        min-height: 40px;
                    ">
                        <strong>Regime {regime_id}</strong>
                        : {description}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # =========================
    # AI Portfolio Analyst Commentary
    # =========================
    if enable_llm_agent and results["llm_commentary"]:
        st.subheader("🤖 AI Portfolio Analyst Commentary")
        st.markdown(results["llm_commentary"])

else:
    st.info("👈 Configure parameters and click **Run Analysis** to begin.")