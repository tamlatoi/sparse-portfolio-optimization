import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# Import my core optimization engine cleanly from main.py
from main import solve_large_sparse_portfolio


def run_comprehensive_backtest(
    df_returns,
    lookback_window=252,
    rebalance_freq=21,
    lambda_val=0.1,
    gamma_val=0.03,
    tau_val=0.005,
):
    """Simulates both the turnover-constrained and the baseline no-turnover portfolios simultaneously."""
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
                new_w_turnover[np.abs(new_w_turnover) < 1e-4] = 0.0
                weights_turnover = new_w_turnover / np.sum(new_w_turnover)

            if new_w_baseline is not None:
                new_w_baseline[np.abs(new_w_baseline) < 1e-4] = 0.0
                weights_baseline = new_w_baseline / np.sum(new_w_baseline)

        # --- 3. RECORD REALIZED RETURNS & ASSET COUNTS ---
        if weights_turnover is not None and weights_baseline is not None:
            ret_turnover = np.dot(weights_turnover, daily_returns)
            ret_baseline = np.dot(weights_baseline, daily_returns)

            strat_turnover_returns.append(ret_turnover)
            strat_baseline_returns.append(ret_baseline)
            backtest_dates.append(current_date)

            # Track structural sparse portfolio density changes
            active_turnover_count.append(np.sum(weights_turnover > 0))
            active_baseline_count.append(np.sum(weights_baseline > 0))

    # Combine historical streams into a structured summary dataframe
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


def evaluate_metrics(results_df, df_returns):
    """Computes and compares metrics across all three operational investment structures."""
    summary_table = {}
    benchmark_returns = df_returns.loc[results_df.index].mean(axis=1)
    results_df["Benchmark"] = benchmark_returns

    strategies = {
        "Turnover Constrained (My Strategy)": "Strategy_Turnover",
        "No Turnover Penalty (Baseline)": "Strategy_Baseline",
        "Equally Weighted S&P 100 Index": "Benchmark",
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
    """Generates a clean comparative plot charting equity curves on the left 
    and a highly faded green bar series for asset counts resting safely in the background.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Calculate compounded wealth lines
    cum_turnover = (1 + results_df["Strategy_Turnover"]).cumprod() - 1
    cum_baseline = (1 + results_df["Strategy_Baseline"]).cumprod() - 1
    cum_benchmark = (1 + results_df["Benchmark"]).cumprod() - 1

    # --- AXIS 1: LEFT SIDE (EQUITY TRAJECTORIES) ---
    # High zorder ensures these lines are physically rendered on top of everything else
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
        label="Equally Weighted S&P 100 Benchmark",
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.5,
        zorder=3,
    )

    ax1.set_xlabel("Date", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Cumulative Return (%)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5, zorder=0)

    # --- AXIS 2: RIGHT SIDE (PERFECTLY FLAT FADED BACKGROUND SHAPE) ---
    ax2 = ax1.twinx()

    # Using 'where=pre' or 'where=post' creates a clean, solid block structure 
    # without overlapping daily artifacts
    ax2.fill_between(
        results_df.index,
        results_df["Active_Assets_Turnover"],
        step="pre",       # Crucial: forces a clean geometric block layout
        color="#2ca02c",
        alpha=0.06,       # Keeps it beautifully faded and soft
        label="Turnover Strategy Asset Count",
        zorder=1
    )

    ax2.set_ylabel(
        "Number of Active Tickers Held", fontsize=11, fontweight="bold", color="#2ca02c"
    )
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    
    # Set rigid vertical limits for asset count so it looks balanced
    ax2.set_ylim(0, max(results_df["Active_Assets_Turnover"]) + 2)

    # Ensure right axis doesn't draw lines over the left axis
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    # Cleanly integrate legends together
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9)

    plt.title(
        "Comprehensive Backtest Matrix & Dynamic Sparsity Densities",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    fig.tight_layout()

    plt.savefig("comprehensive_strategy_comparison.png", dpi=300)
    print("\nVisual plot generated with a faded green background asset density block.")
    plt.show()

if __name__ == "__main__":
    sp100_string = "AAPL MSFT GOOGL AMZN NVDA META TSLA BRK-B LLY V UNH JPM XOM WMT MA PG GE GILD PLTR T"
    raw_tickers = sp100_string.split()

    print("Downloading global multi-asset historical returns matrix (Asset-by-Asset mode)...")
    
    # 1. Download asset-by-asset to isolate timeline failures
    downloaded_series = {}
    for ticker in raw_tickers:
        try:
            # Fetching data individually keeps one young stock (like PLTR) from breaking old ones (like AAPL)
            df_single = yf.download(ticker, start="2015-01-01", end="2026-01-01", progress=False)
            
            # Extract closing price safely regardless of yfinance multi-index nesting
            if isinstance(df_single.columns, pd.MultiIndex):
                col = "Adj Close" if "Adj Close" in df_single.columns.levels[0] else "Close"
                series = df_single[col][ticker]
            else:
                col = "Adj Close" if "Adj Close" in df_single.columns else "Close"
                series = df_single[col]
                
            if not series.dropna().empty:
                downloaded_series[ticker] = series
        except Exception:
            print(f"Skipping {ticker}: Insufficient data footprint for this timeline.")

    # 2. Reconstruct dataframe from verified downloads
    df_raw = pd.DataFrame(downloaded_series)
    
    # 3. Handle assets with shorter histories gracefully
    # Drop rows at the beginning where newer assets didn't exist yet, 
    # or drop assets that are too young to give you a clean backtest matrix
    cleaned_data = df_raw.dropna(axis=1, how="any")  # Keeps assets alive across the entire period
    
    if cleaned_data.empty or cleaned_data.shape[1] < 3:
        print("\n[Warning]: Dropna(how='any') left too few assets. Falling back to clearing younger assets...")
        # Alternate strategy: Keep assets that have at least 80% data coverage over the 26-year horizon
        threshold = int(len(df_raw) * 0.8)
        df_filtered = df_raw.dropna(thresh=threshold, axis=1)
        cleaned_data = df_filtered.fillna(method="ffill").dropna()

    print(f"Data matrix stabilized. Simulating across {cleaned_data.shape[1]} long-history assets.")
    daily_returns = cleaned_data.pct_change().dropna()

    # --- Hyperparameters ---
    lambda_val = 0.1
    gamma_val = 0.03
    tau_val = 0.005

    print("Initiating twin-engine backtest loop simulation...\n")
    results = run_comprehensive_backtest(
        daily_returns,
        lookback_window=252,
        rebalance_freq=21,
        lambda_val=lambda_val,
        gamma_val=gamma_val,
        tau_val=tau_val,
    )

    comparison_metrics = evaluate_metrics(results, daily_returns)
    print("\n========================= PERFORMANCE SUMMARY TABLE =========================")
    print(comparison_metrics.to_string())
    print("=============================================================================")

    plot_comparative_results(results)