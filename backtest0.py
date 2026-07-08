import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cp

def solve_markowitz_portfolio(mu, Q, lambda_param, w_drift=None, tau=0.0):
    """
    Solves a standard Markowitz Mean-Variance optimization problem
    with an optional dynamic turnover constraint relative to drifted weights.
    """
    n = len(mu)
    w = cp.Variable(n)
    
    if w_drift is None:
        w_drift = np.zeros(n)
        
    # Standard Markowitz Objective: Maximize Expected Return - Risk Penalty
    portfolio_risk = 0.5 * cp.quad_form(w, Q)
    expected_return = lambda_param * (mu @ w)
    turnover_penalty = tau * cp.norm(w - w_drift, 1)
    
    objective = cp.Minimize(portfolio_risk - expected_return + turnover_penalty)
    constraints = [cp.sum(w) == 1, w >= 0]  # Fully invested, long-only
    
    prob = cp.Problem(objective, constraints)
    
    try:
        prob.solve(solver=cp.OSQP)
    except Exception:
        try:
            prob.solve(solver=cp.ECOS)
        except Exception:
            return None
            
    if prob.status not in ["optimal", "optimal_inaccurate"] or w.value is None:
        return None
        
    return w.value


def run_comprehensive_backtest(
    df_returns,
    lookback_window=252,
    rebalance_freq=21,
    lambda_val=0.1,
    tau_val=0.005,
):
    """
    Simulates turnover-constrained Markowitz vs baseline Markowitz portfolios.
    Starts with a true 0% holding (cash) position and safely enforces 
    out-of-sample rebalancing execution without lookahead bias.
    """
    n_timesteps, n_assets = df_returns.shape

    strat_turnover_returns = []
    active_turnover_count = []
    weights_turnover = np.zeros(n_assets)  # Explicitly initialized to 0% at launch

    strat_baseline_returns = []
    active_baseline_count = []
    weights_baseline = np.zeros(n_assets)  # Explicitly initialized to 0% at launch

    backtest_dates = []

    for t in range(lookback_window, n_timesteps):
        current_date = df_returns.index[t]
        daily_returns = df_returns.iloc[t].values

        # --- 1. REBALANCING WINDOW (Using strictly historical data up to t-1) ---
        if (t - lookback_window) % rebalance_freq == 0:
            window_data = df_returns.iloc[t - lookback_window : t]

            # Annualized arithmetic parameters
            mu_window = window_data.mean().values * 252
            Q_window = window_data.cov().values * 252

            # Check if this is the absolute first deployment from cash (all weights are 0)
            is_first_deployment = np.all(weights_turnover == 0.0)

            # A. Optimize Strategy 1 (Turnover Penalized Markowitz)
            # We bypass tau on deployment day to avoid solver precision flattening
            new_w_turnover = solve_markowitz_portfolio(
                mu_window, 
                Q_window, 
                lambda_val, 
                w_drift=weights_turnover, 
                tau=0.0 if is_first_deployment else tau_val
            )
            if new_w_turnover is not None:
                weights_turnover = new_w_turnover

            # B. Optimize Strategy 2 (Pure Markowitz baseline, always starts unconstrained)
            new_w_baseline = solve_markowitz_portfolio(
                mu_window, Q_window, lambda_val, w_drift=None, tau=0.0
            )
            if new_w_baseline is not None:
                weights_baseline = new_w_baseline

        # --- 2. RECORD OUT-OF-SAMPLE REALIZED RETURNS ON DAY T ---
        ret_turnover = np.dot(weights_turnover, daily_returns)
        ret_baseline = np.dot(weights_baseline, daily_returns)

        strat_turnover_returns.append(ret_turnover)
        strat_baseline_returns.append(ret_baseline)
        backtest_dates.append(current_date)

        active_turnover_count.append(np.sum(weights_turnover > 0.001))
        active_baseline_count.append(np.sum(weights_baseline > 0.001))

        # --- 3. END-OF-DAY DRIFT TRACKING FOR DAY t+1 ---
        drifted_turnover = weights_turnover * (1.0 + daily_returns)
        denom_turnover = np.sum(drifted_turnover)
        weights_turnover = drifted_turnover / denom_turnover if denom_turnover > 0 else weights_turnover

        drifted_baseline = weights_baseline * (1.0 + daily_returns)
        denom_baseline = np.sum(drifted_baseline)
        weights_baseline = drifted_baseline / denom_baseline if denom_baseline > 0 else weights_baseline

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
    """Computes basic annualized statistics and benchmarks against equal weights."""
    summary_table = {}
    benchmark_returns = df_returns.loc[results_df.index].mean(axis=1)
    results_df["Benchmark"] = benchmark_returns

    strategies = {
        "Turnover Constrained Markowitz": "Strategy_Turnover",
        "Pure Markowitz (Baseline)": "Strategy_Baseline",
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
    """Generates visual graphics tracking performance curves and asset counts."""
    fig, ax1 = plt.subplots(figsize=(12, 6))

    cum_turnover = (1 + results_df["Strategy_Turnover"]).cumprod() - 1
    cum_baseline = (1 + results_df["Strategy_Baseline"]).cumprod() - 1
    cum_benchmark = (1 + results_df["Benchmark"]).cumprod() - 1

    ax1.plot(cum_turnover.index, cum_turnover * 100, label="Turnover Constrained Markowitz", color="#1f77b4", linewidth=2.5, zorder=5)
    ax1.plot(cum_baseline.index, cum_baseline * 100, label="Pure Markowitz Baseline", color="#d62728", linestyle="-.", linewidth=1.5, zorder=4)
    ax1.plot(cum_benchmark.index, cum_benchmark * 100, label="Equally Weighted Benchmark", color="#ff7f0e", linestyle="--", linewidth=1.5, zorder=3)

    ax1.set_xlabel("Date", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Cumulative Return (%)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5, zorder=0)

    ax2 = ax1.twinx()
    ax2.fill_between(results_df.index, results_df["Active_Assets_Turnover"], step="pre", color="#2ca02c", alpha=0.06, label="Turnover Asset Count", zorder=1)

    ax2.set_ylabel("Number of Tickers Held", fontsize=11, fontweight="bold", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax2.set_ylim(0, max(results_df["Active_Assets_Turnover"]) + 5)

    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9)

    plt.title("Markowitz Backtest Comparison (Initial Deployment from Zero)", fontsize=13, fontweight="bold", pad=15)
    fig.tight_layout()
    plt.savefig("SP100_backtest.png", dpi=300)
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

    print("Downloading historical asset price matrix from yfinance...")
    # Single robust multi-ticker download string to circumvent rate limiting
    df_download = yf.download(raw_tickers, start="2015-01-01", end="2026-01-01", progress=False)
    
    if isinstance(df_download.columns, pd.MultiIndex):
        if "Adj Close" in df_download.columns.levels[0]:
            df_prices = df_download["Adj Close"]
        else:
            df_prices = df_download["Close"]
    else:
        df_prices = df_download

    # Keep assets that contain at least 85% full lifespan rows
    cleaned_prices = df_prices.dropna(axis=1, thresh=int(len(df_prices) * 0.85))
    cleaned_prices = cleaned_prices.ffill().bfill().dropna()

    if cleaned_prices.empty or cleaned_prices.shape[1] < 2:
        print("\n[CRITICAL]: yfinance data down or blocked. Using synthetic data matrix...")
        dates = pd.date_range(start="2015-01-01", end="2026-01-01", freq="B")
        synthetic_data = np.random.normal(loc=0.0005, scale=0.015, size=(len(dates), 5))
        daily_returns = pd.DataFrame(synthetic_data, index=dates, columns=["A", "B", "C", "D", "E"])
    else:
        print(f"Data stabilized. Simulating across {cleaned_prices.shape[1]} unique assets.")
        daily_returns = cleaned_prices.pct_change().dropna()

    # Parameters
    lambda_val = 0.1  # Risk-aversion index
    tau_val = 0.05

    print("Running out-of-sample parallel historical simulation loop...\n")
    results = run_comprehensive_backtest(
        daily_returns,
        lookback_window=252,
        rebalance_freq=21,
        lambda_val=lambda_val,
        tau_val=tau_val,
    )

    comparison_metrics = evaluate_metrics(results, daily_returns)
    print("\n========================= PERFORMANCE SUMMARY TABLE =========================")
    print(comparison_metrics.to_string())
    print("=============================================================================")

    plot_comparative_results(results)