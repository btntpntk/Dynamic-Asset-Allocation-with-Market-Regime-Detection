# 📊 Regime-Aware Portfolio Optimization Dashboard

This project is a Streamlit-based web application that implements and visualizes a regime-aware portfolio optimization strategy. The application identifies different market regimes (e.g., bull, bear, high/low volatility) using a Hidden Markov Model (HMM) and then constructs an optimal portfolio for each regime.

## ✨ Features

-   **Dynamic Market Regime Detection**: Utilizes a Hidden Markov Model (HMM) to classify market conditions into distinct regimes based on historical data.
-   **Mean-Variance Optimization**: Constructs an optimal portfolio for each regime using mean-variance optimization.
-   **Event-Driven Backtesting**: A stateful, event-driven backtesting engine that simulates the strategy's performance over time.
-   **Interactive Dashboard**: A Streamlit application to configure and visualize the backtest results.
-   **AI-Powered Insights**: An optional "AI Portfolio Analyst" (powered by a Large Language Model) to provide qualitative commentary on the portfolio's risk and performance.
-   **Performance and Risk Analysis**: Comprehensive performance and risk metrics, including Sharpe ratio, max drawdown, and Value at Risk (VaR).

## ⚙️ How It Works

The core of the project is the `run_pipeline.py` script, which executes the following steps:

1.  **Data Loading**: Downloads historical price data for a given set of tickers from Yahoo Finance.
2.  **Feature Engineering**: Calculates log returns and rolling volatility to be used as features for the HMM.
3.  **HMM Training**: Trains a Hidden Markov Model on the engineered features to identify market regimes.
4.  **Event-Driven Backtest**: Iterates through the historical data, and at each step:
    -   Predicts the current market regime.
    -   If the regime changes, it rebalances the portfolio by calculating the optimal asset allocation for the new regime using mean-variance optimization.
5.  **Performance Analysis**: Calculates various performance and risk metrics for the backtest period.
6.  **AI Commentary**: (Optional) Generates a qualitative analysis of the portfolio's risk and performance using a large language model.

The Streamlit application (`streamlit_app/app.py`) provides a user-friendly interface to configure the pipeline and visualize the results.

## 🚀 Getting Started

### Prerequisites

-   Python 3.8 or higher
-   An API key from OpenAI (if you want to use the AI Portfolio Analyst feature).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repository.git
    cd your-repository
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables:**
    -   Create a file named `.env` in the root of the project.
    -   Add your OpenAI API key to the `.env` file:
        ```
        OPENAI_API_KEY="your-api-key"
        ```

### Usage

To run the Streamlit application, execute the following command in your terminal:

```bash
streamlit run streamlit_app/app.py
```

This will open the application in your web browser. You can then configure the backtest parameters in the sidebar and click "Run Analysis" to see the results.

## 📁 Project Structure

```
├── .env                  # Environment variables (e.g., API keys)
├── requirements.txt      # Python dependencies
├── todo.txt              # TODO list
├── prompts/              # Prompts for the LLM agent
├── reports/              # Output reports
├── src/                  # Core application logic
│   ├── agents/           # LLM-based agents
│   ├── backtesting/      # Backtesting engine
│   ├── data/             # Data loading and preprocessing
│   ├── features/         # Feature engineering
│   ├── models/           # HMM and portfolio optimization models
│   ├── risk/             # Risk metrics
│   └── run_pipeline.py   # Main pipeline script
└── streamlit_app/
    └── app.py            # Streamlit application
```

## 📦 Dependencies

The main dependencies of the project are:

-   **`numpy`** and **`pandas`**: For numerical computing and data manipulation.
-   **`yfinance`**: For downloading financial data.
-   **`scikit-learn`** and **`hmmlearn`**: For machine learning and the HMM model.
-   **`cvxpy`**: For portfolio optimization.
-   **`streamlit`**: For the web application.
-   **`plotly`**: For interactive visualizations.
-   **`openai`**: For the AI Portfolio Analyst.
-   **`backtesting.py`**: For the backtesting engine.

A full list of dependencies can be found in the `requirements.txt` file.
