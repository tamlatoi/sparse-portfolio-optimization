import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# Import your core optimization engine cleanly from main.py
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

    # Data collection arrays
    strat_turnover_returns = []
    active_turnover_count = []
    weights_turnover = None

    strat_baseline_returns = []
    active_baseline_count = []
    weights_baseline = None

    backtest_dates = []

    for t in range(lookback_window, n_timesteps):
        current_date = df_returns.index[t]
        daily_returns = df_returns.iloc[t].values

        # --- 1. REBALANCING WINDOW (Executed at the START of the day/period) ---
        if (t - lookback_window) % rebalance_freq == 0:
            window_data = df_returns.iloc[t - lookback_window : t]

            n_days = window_data.shape[0]
            terminal_wealth = (1.0 + window_data).prod(axis=0)
            terminal_wealth = np.maximum(1e-5, terminal_wealth.values)

            mu_window = (terminal_wealth) ** (252.0 / n_days) - 1.0
            Q_window = window_data.cov().values * 252

            # A. Optimize Strategy 1 (Uses weights drifted from the PREVIOUS day's close)
            w_drift_target = (
                weights_turnover if weights_turnover is not None else np.zeros(n_assets)
            )
            new_w_turnover = solve_large_sparse_portfolio(
                mu_window,
                Q_window,
                lambda_val,
                gamma_val,
                w_drift=w_drift_target,
                tau=tau_val,
            )

            # B. Optimize Strategy 2
            new_w_baseline = solve_large_sparse_portfolio(
                mu_window, Q_window, lambda_val, gamma_val, w_drift=None, tau=0.0
            )

            if new_w_turnover is not None:
                new_w_turnover[np.abs(new_w_turnover) < 1e-4] = 0.0
                weights_turnover = new_w_turnover / np.sum(new_w_turnover)
            elif weights_turnover is None:
                weights_turnover = np.ones(n_assets) / n_assets  # Fallback allocation

            if new_w_baseline is not None:
                new_w_baseline[np.abs(new_w_baseline) < 1e-4] = 0.0
                weights_baseline = new_w_baseline / np.sum(new_w_baseline)
            elif weights_baseline is None:
                weights_baseline = np.ones(n_assets) / n_assets

        # --- 2. RECORD REALIZED RETURNS ---
        # Calculate returns using the weights that ENTERED the trading day
        ret_turnover = np.dot(weights_turnover, daily_returns)
        ret_baseline = np.dot(weights_baseline, daily_returns)

        strat_turnover_returns.append(ret_turnover)
        strat_baseline_returns.append(ret_baseline)
        backtest_dates.append(current_date)

        active_turnover_count.append(np.sum(weights_turnover > 0))
        active_baseline_count.append(np.sum(weights_baseline > 0))

        # --- 3. END-OF-DAY DRIFT TRACKING ---
        # Drift allocations based on today's market movements to prepare for tomorrow
        drifted_turnover = weights_turnover * (1 + daily_returns)
        weights_turnover = drifted_turnover / np.sum(drifted_turnover)

        drifted_baseline = weights_baseline * (1 + daily_returns)
        weights_baseline = drifted_baseline / np.sum(drifted_baseline)

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
        n_days = len(returns)

        total_growth = (1 + returns).prod()
        ann_return = (total_growth) ** (252.0 / n_days) - 1.0

        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
        cum_return = total_growth - 1.0

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

    cum_turnover = (1 + results_df["Strategy_Turnover"]).cumprod() - 1
    cum_baseline = (1 + results_df["Strategy_Baseline"]).cumprod() - 1
    cum_benchmark = (1 + results_df["Benchmark"]).cumprod() - 1

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

    ax2 = ax1.twinx()
    ax2.fill_between(
        results_df.index,
        results_df["Active_Assets_Turnover"],
        step="pre",
        color="#2ca02c",
        alpha=0.06,
        label="Turnover Strategy Asset Count",
        zorder=1,
    )

    ax2.set_ylabel(
        "Number of Active Tickers Held",
        fontsize=11,
        fontweight="bold",
        color="#2ca02c",
    )
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax2.set_ylim(0, max(results_df["Active_Assets_Turnover"]) + 2)

    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

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
    plt.show()


if __name__ == "__main__":
    sp100_string = """
    AAPL MSFT GOOGL AMZN NVDA META TSLA BRK-B LLY V
    UNH JPM XOM WMT MA PG AVGO ORCL HD CVX
    COST MRK BAC ABBV KO PEP AMD ADBE CRM QCOM
    CMCSA NFLX DIS TMUS CSCO VZ INTC TXN AMGN IBM
    HON GE CAT LMT RTX AXP GS BLK C MS
    SCHW LOW NKE SBUX TJX TGT MCD SRE DUK SO
    NEE LIN APD FCX COP EOG SLB JNJ PFE BMY
    GILD MDT ISRG SYK REGN UPS FDX DE MMM EMR
    AEP EXC WM PLTR PANW ANET DELL MU LRCX AMAT
    T MDLZ MO PM CL EL COF BK WFC
    """
    raw_tickers = sp100_string.split()

    print(
        "Downloading global multi-asset historical returns matrix (Asset-by-Asset mode)..."
    )

    downloaded_series = {}
    for ticker in raw_tickers:
        try:
            df_single = yf.download(
                ticker, start="2015-01-01", end="2026-01-01", progress=False
            )

            if isinstance(df_single.columns, pd.MultiIndex):
                col = (
                    "Adj Close"
                    if "Adj Close" in df_single.columns.levels[0]
                    else "Close"
                )
                series = df_single[col][ticker]
            else:
                col = "Adj Close" if "Adj Close" in df_single.columns else "Close"
                series = df_single[col]

            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]

            series = series.squeeze()
            series.name = ticker

            if not series.dropna().empty:
                downloaded_series[ticker] = series
        except Exception:
            print(
                f"Skipping {ticker}: Insufficient data footprint for this timeline."
            )

    df_raw = pd.DataFrame(downloaded_series)
    cleaned_data = df_raw.dropna(axis=1, how="any")

    if cleaned_data.empty or cleaned_data.shape[1] < 3:
        print(
            "\n[Warning]: Dropna(how='any') left too few assets. Falling back to clearing younger assets..."
        )
        threshold = int(len(df_raw) * 0.8)
        df_filtered = df_raw.dropna(thresh=threshold, axis=1)
        # FIXED: .fillna(method="ffill") replaced with modern .ffill() syntax
        cleaned_data = df_filtered.ffill().dropna()

    print(
        f"Data matrix stabilized. Simulating across {cleaned_data.shape[1]} long-history assets."
    )
    daily_returns = cleaned_data.pct_change().dropna()

    lambda_val = 0.1
    gamma_val = 0.03
    tau_val = 0.05

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
    print(
        "\n========================= PERFORMANCE SUMMARY TABLE ========================="
    )
    print(comparison_metrics.to_string())
    print(
        "============================================================================="
    )

    plot_comparative_results(results)