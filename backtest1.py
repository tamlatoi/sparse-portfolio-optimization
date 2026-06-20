import concurrent.futures
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# Pure link import from your local core optimization engine
from main import solve_large_sparse_portfolio


def run_comprehensive_backtest(
    df_returns,
    lookback_window=252,
    rebalance_freq=21,
    lambda_val=2.0,       
    gamma_val=0.0,       
    tau_val=0.005,
):
    """Simulates both the turnover-constrained and baseline portfolios simultaneously."""
    n_timesteps, n_assets = df_returns.shape

    # Data collection arrays for Strategy 1 (Turnover Constrained)
    strat_turnover_returns = []
    active_turnover_count = []
    weights_turnover = None

    # Data collection arrays for Strategy 2 (Baseline No-Turnover)
    strat_baseline_returns = []
    active_baseline_count = []
    weights_baseline = None

    backtest_dates = []

    for t in range(lookback_window, n_timesteps):
        current_date = df_returns.index[t]
        daily_returns = df_returns.iloc[t].values

        # --- 1. DAILY DRIFT TRACKING ---
        if weights_turnover is not None:
            drifted_turnover = weights_turnover * (1 + daily_returns)
            weights_turnover = drifted_turnover / np.sum(drifted_turnover)

        if weights_baseline is not None:
            drifted_baseline = weights_baseline * (1 + daily_returns)
            weights_baseline = drifted_baseline / np.sum(drifted_baseline)

        # --- 2. REBALANCING WINDOW ---
        if (t - lookback_window) % rebalance_freq == 0:
            window_data = df_returns.iloc[t - lookback_window : t]
            mu_window = window_data.mean().values * 252
            Q_window = window_data.cov().values * 252

            # A. Optimize Strategy 1 (Passes genuine drifted weights baseline and tau)
            w_drift_target = (
                weights_turnover
                if weights_turnover is not None
                else np.zeros(n_assets)
            )
            new_w_turnover = solve_large_sparse_portfolio(
                mu_window,
                Q_window,
                lambda_val,
                gamma_val,
                w_drift=w_drift_target,
                tau=tau_val,
            )

            # B. Optimize Strategy 2 (Wipes slate clean: w_drift=None, tau=0.0)
            new_w_baseline = solve_large_sparse_portfolio(
                mu_window, Q_window, lambda_val, gamma_val, w_drift=None, tau=0.0
            )

            # Save allocations safely if optimization steps succeeded
            if new_w_turnover is not None:
                new_w_turnover[new_w_turnover < 0.005] = 0.0
                weights_turnover = new_w_turnover / np.sum(new_w_turnover)

            if new_w_baseline is not None:
                new_w_baseline[new_w_baseline < 0.005] = 0.0
                weights_baseline = new_w_baseline / np.sum(new_w_baseline)

        # --- 3. RECORD REALIZED RETURNS & ASSET COUNTS ---
        if weights_turnover is not None and weights_baseline is not None:
            ret_turnover = np.dot(weights_turnover, daily_returns)
            ret_baseline = np.dot(weights_baseline, daily_returns)

            strat_turnover_returns.append(ret_turnover)
            strat_baseline_returns.append(ret_baseline)
            backtest_dates.append(current_date)

            active_turnover_count.append(np.sum(weights_turnover > 0))
            active_baseline_count.append(np.sum(weights_baseline > 0))

    results_df = pd.DataFrame(
        {
            "Strategy_Turnover": strat_turnover_returns,
            "Strategy_Baseline": strat_baseline_returns,
            "Active_Assets_Turnover": active_turnover_count,
            "Active_Assets_Baseline": active_baseline_count,
        },
        index=backtest_dates,
    )
    return results_df


def evaluate_metrics(results_df, benchmark_returns):
    """Computes and compares performance metrics with an external aligned benchmark series."""
    summary_table = {}
    
    # Force alignment across index vectors
    results_df.index = pd.to_datetime(results_df.index).tz_localize(None)
    benchmark_returns.index = pd.to_datetime(benchmark_returns.index).tz_localize(None)
    
    # Intersect to handle matching date shapes seamlessly
    common_dates = results_df.index.intersection(benchmark_returns.index)
    results_df = results_df.loc[common_dates]
    
    results_df["SPY_Benchmark"] = benchmark_returns.loc[common_dates]

    strategies = {
        "Turnover Constrained (My Strategy)": "Strategy_Turnover",
        "No Turnover Penalty (Baseline)": "Strategy_Baseline",
        "S&P 500 Index (SPY Benchmark)": "SPY_Benchmark",
    }

    for name, column in strategies.items():
        returns = results_df[column]
        ann_return = (1 + returns.mean()) ** 252 - 1
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
        cum_return = (1 + returns).cumprod().iloc[-1] - 1

        summary_table[name] = {
            "Cumulative Return": f"{cum_return * 100:.2f}%",
            "Annualized Return": f"{ann_return * 100:.2f}%",
            "Annualized Volatility": f"{ann_vol * 100:.2f}%",
            "Sharpe Ratio": f"{sharpe:.2f}",
        }

    return pd.DataFrame(summary_table).T


def plot_comparative_results(results_df):
    """Generates a comparative plot tracking strategies against the SPY Index."""
    results_df.index = pd.to_datetime(results_df.index).tz_localize(None)
    fig, ax1 = plt.subplots(figsize=(12, 6))

    cum_turnover = (1 + results_df["Strategy_Turnover"]).cumprod() - 1
    cum_baseline = (1 + results_df["Strategy_Baseline"]).cumprod() - 1
    cum_benchmark = (1 + results_df["SPY_Benchmark"]).cumprod() - 1

    # --- AXIS 1: LEFT SIDE (EQUITY TRAJECTORIES) ---
    ax1.plot(
        cum_turnover.index,
        cum_turnover * 100,
        label="Turnover Constrained Strategy (With w_drift)",
        color="#1f77b4",
        linewidth=2.5,
        zorder=5,
    )
    ax1.plot(
        cum_baseline.index,
        cum_baseline * 100,
        label="No Turnover Penalty Baseline (No w_drift)",
        color="#d62728",
        linestyle="-.",
        linewidth=1.5,
        zorder=4,
    )
    ax1.plot(
        cum_benchmark.index,
        cum_benchmark * 100,
        label="S&P 500 Market Benchmark (SPY)",
        color="#555555",  
        linestyle="--",
        linewidth=1.8,
        zorder=3,
    )

    ax1.set_xlabel("Date", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Cumulative Return (%)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5, zorder=0)

    # --- AXIS 2: RIGHT SIDE (ASSET COUNTS) ---
    ax2 = ax1.twinx()
    ax2.fill_between(
        results_df.index,
        results_df["Active_Assets_Turnover"],
        step="post",       
        color="#2ca02c",
        alpha=0.06,       
        label="Turnover Strategy Asset Count",
        zorder=1
    )

    ax2.set_ylabel(
        "Number of Active Tickers Held", fontsize=11, fontweight="bold", color="#2ca02c"
    )
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax2.set_ylim(0, max(results_df["Active_Assets_Turnover"]) + 5)

    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9)

    plt.title(
        "Strategy vs Actual S&P 500 Index (SPY) Comparison Matrix",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    fig.tight_layout()

    plt.savefig("spy_market_strategy_comparison.png", dpi=300)
    print("\nVisual plot generated vs SPY Market Index.")
    plt.show()


def download_single_ticker(ticker):
    """Downloads a single asset and handles formatting to eliminate multi-index corruption."""
    try:
        df = yf.download(ticker, start="2015-01-01", end="2026-01-01", progress=False)
        if df.empty:
            return ticker, None
        
        # Unpack MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            col_field = "Adj Close" if "Adj Close" in df.columns.levels[0] else "Close"
            series = df[col_field][ticker]
        else:
            col_field = "Adj Close" if "Adj Close" in df.columns else "Close"
            series = df[col_field]
            
        # Standardize index timeline
        series.index = pd.to_datetime(series.index).tz_localize(None)
        return ticker, series
    except Exception:
        return ticker, None


if __name__ == "__main__":
    print("Scraping comprehensive list of current S&P 500 tickers from Wikipedia...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            html_content = response.read()
        payload = pd.read_html(html_content, attrs={"id": "constituents"})[0]
        raw_tickers = [str(t).replace(".", "-").strip() for t in payload["Symbol"].tolist()]
        print(f"Successfully scraped {len(raw_tickers)} ticker components from Wikipedia.")
    except Exception as e:
        print(f"Scraping encountered an error ({e}). Falling back to manual fallback list...")
        raw_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "JPM", "UNH", "V"]

    # 1. Isolate market benchmark data
    print("\nDownloading SPY Benchmark Index...")
    _, spy_series = download_single_ticker("SPY")
    if spy_series is None:
        raise RuntimeError("Critical Error: SPY Benchmark failed to download. Check network connection.")
    spy_returns = spy_series.pct_change().dropna()

    # 2. Process data downloads using an isolated parallel loop
    print(f"Launching parallel execution threads to download {len(raw_tickers)} assets...")
    price_dictionary = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_ticker = {executor.submit(download_single_ticker, tick): tick for tick in raw_tickers}
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker, series = future.result()
            # Verify data is returned as a valid Series before adding it to the tracking matrix
            if series is not None and isinstance(series, pd.Series) and len(series) > 0:
                price_dictionary[ticker] = series

    # Secure asset matrix construction
    component_prices = pd.DataFrame(price_dictionary)
    component_prices.index = pd.to_datetime(component_prices.index).tz_localize(None)

    print("\nFiltering asset timeline maturities to secure a clean 2016 engine startup...")
    
    # Filter out assets missing more than 10% of their historical data points
    total_expected_days = len(component_prices.index)
    valid_counts = component_prices.notna().sum(axis=0)
    min_required_days = int(total_expected_days * 0.90)
    
    mature_tickers = valid_counts[valid_counts >= min_required_days].index.tolist()
    component_prices = component_prices[mature_tickers]

    # Forward-fill gaps, then clean drop lingering structural initialization records
    component_prices = component_prices.ffill()
    cleaned_components = component_prices.dropna(axis=1)

    print(f"Data matrix stabilized. Dropped {len(raw_tickers) - cleaned_components.shape[1]} young assets.")
    print(f"Simulating across {cleaned_components.shape[1]} long-history S&P components.")
    
    daily_returns = cleaned_components.pct_change().dropna()

    # --- Hyperparameters ---
    lambda_val = 2.0
    gamma_val = 0.0
    tau_val = 0.05

    print("\nInitiating massive 500-asset twin-engine backtest loop simulation...\n")
    results = run_comprehensive_backtest(
        daily_returns,
        lookback_window=252,
        rebalance_freq=21,
        lambda_val=lambda_val,
        gamma_val=gamma_val,
        tau_val=tau_val,
    )

    comparison_metrics = evaluate_metrics(results, spy_returns)
    print("\n========================= PERFORMANCE SUMMARY TABLE =========================")
    print(comparison_metrics.to_string())
    print("=============================================================================")

    plot_comparative_results(results)