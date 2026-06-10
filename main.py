import sys
import cvxpy as cp
import numpy as np
import yfinance as yf


def solve_large_sparse_portfolio(mu, Q, lambda_param, gamma, w_drift=None, tau=0.0):
    """Solves the sparse portfolio optimization problem with a dynamic turnover penalty."""
    n = len(mu)
    w = cp.Variable(n)

    # 1. Core Objective Elements
    portfolio_risk = 0.5 * cp.quad_form(w, Q)
    expected_return = lambda_param * (mu @ w)
    sparsity_penalty = gamma * cp.norm(w, 1)

    # 2. Dynamic Turnover Penalty (Handles trading friction relative to drifted weights)
    if w_drift is None:
        w_drift = np.zeros(n)
    turnover_penalty = tau * cp.norm(w - w_drift, 1)

    # Combine all elements into a unified objective
    objective = cp.Minimize(
        portfolio_risk - expected_return + sparsity_penalty + turnover_penalty
    )
    constraints = [cp.sum(w) == 1, w >= 0]  # Fully invested, long-only

    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver=cp.OSQP)
    except Exception:
        prob.solve(solver=cp.ECOS)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        return None  # Let backtester fall back safely rather than crashing

    return w.value


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
    sp100_tickers = [ticker for ticker in sp100_string.split()]

    print(f"Successfully initialized {len(sp100_tickers)} components of the S&P 100.")
    print("Downloading 3 years of historical market data from Yahoo Finance...")

    df_raw = yf.download(sp100_tickers, start="2023-01-01", end="2026-01-01")

    # --- ARMORED COLUMN PARSING ---
    if df_raw.columns.ndim > 1:
        available_fields = df_raw.columns.levels[0]
        raw_data = (
            df_raw["Adj Close"]
            if "Adj Close" in available_fields
            else df_raw["Close"]
        )
    else:
        raw_data = (
            df_raw["Adj Close"]
            if "Adj Close" in df_raw.columns
            else df_raw["Close"]
        )

    raw_data = raw_data.dropna(axis=1, how="all")
    cleaned_data = raw_data.dropna(axis=1, how="any")
    active_tickers = list(cleaned_data.columns)

    print(f"Data prep complete. Optimizing across {len(active_tickers)} stocks.")

    daily_returns = cleaned_data.pct_change().dropna()
    mu_real = daily_returns.mean().values * 252
    Q_real = daily_returns.cov().values * 252

    # Standalone execution uses base hyperparameters (no drift constraint on Day 0)
    lambda_val = 0.1
    gamma_val = 0.5

    try:
        weights = solve_large_sparse_portfolio(
            mu_real, Q_real, lambda_val, gamma_val
        )

        print("\n=============================================")
        print(f"       STATIC PORTFOLIO WEIGHT ALLOCATIONS    ")
        print(f"      (Regularization Penalty γ = {gamma_val})  ")
        print("=============================================")

        allocated_count = 0
        for i, ticker in enumerate(active_tickers):
            w_i = weights[i]
            clean_weight = 0.0 if np.isclose(w_i, 0, atol=1e-4) else w_i

            if clean_weight > 0:
                print(f"  {ticker:<6} : {clean_weight:.2%}")
                allocated_count += 1

        print("---------------------------------------------")
        print(f"Result: Concentrated basket of ONLY {allocated_count} core picks.")
        print("=============================================")

    except Exception as e:
        print(f"Optimization pipeline hit an error: {e}")