import os
# Force thread pool allocation limits for processing large ticker universes safely.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import cvxpy as cp
from curl_cffi.requests import Session

# ==============================================================================
# 1. ROBUST COVARIANCE ESTIMATORS & MATRIX CLEANING
# ==============================================================================

def nearest_positive_definite(matrix, eps=1e-4):
    """Computes the nearest Symmetric Positive Definite (SPD) matrix."""
    symmetric_mat = (matrix + matrix.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric_mat)
    eigenvalues = np.maximum(eigenvalues, eps)
    spd_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return (spd_matrix + spd_matrix.T) / 2


def robust_gerber_covariance_mad(returns, c=0.5):
    """Computes the Gerber Covariance Matrix using Median Absolute Deviation (MAD)."""
    T, N = returns.shape
    returns_matrix = np.asarray(returns)
    
    med = np.median(returns_matrix, axis=0)
    mad = np.median(np.abs(returns_matrix - med), axis=0)
    mad = np.where(mad < 1e-5, 1e-5, mad)  
    thresholds = c * mad
    
    U = (returns_matrix >= thresholds).astype(float)
    D = (returns_matrix <= -thresholds).astype(float)
    
    N_CONC = (U.T @ U) + (D.T @ D)
    N_DISC = (U.T @ D) + (D.T @ U)
    
    denominator = T - ((1.0 - U) * (1.0 - D)).T @ ((1.0 - U) * (1.0 - D))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        G = np.where(denominator > 0, (N_CONC - N_DISC) / denominator, 0.0)
    np.fill_diagonal(G, 1.0)
    
    # FIX: Make the core relation matrix G positive definite BEFORE variance scaling
    G_cleaned = nearest_positive_definite(G, eps=1e-4)
    
    sample_std = np.std(returns_matrix, axis=0, ddof=1)
    gerber_cov = np.diag(sample_std) @ G_cleaned @ np.diag(sample_std)
    return gerber_cov


def de_noise_covariance(cov_matrix, T, N):
    """Marchenko-Pastur De-noising Framework tuned for high-dimensions."""
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sigma_sq = np.mean(eigenvalues[eigenvalues > 1e-5]) if np.any(eigenvalues > 1e-5) else 1.0
    
    # FIX: Guarding Marchenko-Pastur when N > T
    aspect_ratio = min(N / T, 1.0)
    lambda_plus = sigma_sq * (1.0 + np.sqrt(aspect_ratio)) ** 2
    
    is_noise = eigenvalues <= lambda_plus
    if np.any(is_noise):
        avg_noise_eigenvalue = np.mean(eigenvalues[is_noise])
        eigenvalues[is_noise] = avg_noise_eigenvalue
        
    de_noised_cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return nearest_positive_definite(de_noised_cov)

# ==============================================================================
# 2. HIGH-DIMENSIONAL CVAR PORTFOLIO OPTIMIZER 
# ==============================================================================

def optimize_portfolio_hd_cvar(historical_scenarios, cov_matrix, mu, w_initial=None, lamb=1.0, tau=0.02, use_cvar=True):
    """Convex Portfolio Optimizer scaling CVaR with dynamic turnover constraints."""
    S, N = historical_scenarios.shape
    scenarios_matrix = np.asarray(historical_scenarios)
    stabilized_cov = nearest_positive_definite(cov_matrix, eps=1e-5)
    
    w = cp.Variable(N)
    
    # FIX: Both risk metrics and returns are now properly annualized inside the objective function
    portfolio_risk = 0.5 * cp.quad_form(w, cp.psd_wrap(stabilized_cov * 252))
    scaled_return = lamb * (mu @ w)
    
    # FIX: Explicitly ignore turnover regularizers if the portfolio is starting from scratch (weights all 0)
    if w_initial is not None and np.sum(w_initial) > 1e-5:
        norm_regularization = tau * cp.norm(w - w_initial, 1)
        objective = cp.Minimize(portfolio_risk - scaled_return + norm_regularization)
    else:
        objective = cp.Minimize(portfolio_risk - scaled_return)
        
    constraints = [
        cp.sum(w) == 1.0,
        w >= 0.0
    ]

    # FIX: Conditional CVaR Block (Allows true baseline separation for Strategy 2 Markowitz)
    if use_cvar:
        zeta_95 = cp.Variable()
        zeta_99 = cp.Variable()
        z_95 = cp.Variable(S)
        z_99 = cp.Variable(S)
        
        eq_losses = -scenarios_matrix @ (np.ones(N) / N)
        var_95_est = np.percentile(eq_losses, 95)
        var_99_est = np.percentile(eq_losses, 99)
        cvar_95_limit = max(0.015, np.mean(eq_losses[eq_losses >= var_95_est]))
        cvar_99_limit = max(0.025, np.mean(eq_losses[eq_losses >= var_99_est]))
        
        constraints += [
            z_95 >= 0.0,
            (-scenarios_matrix @ w) - zeta_95 <= z_95,
            zeta_95 + (1.0 / (1.0 - 0.95)) * cp.mean(z_95) <= cvar_95_limit, 
            
            z_99 >= 0.0,
            (-scenarios_matrix @ w) - zeta_99 <= z_99,
            zeta_99 + (1.0 / (1.0 - 0.99)) * cp.mean(z_99) <= cvar_99_limit  
        ]
    
    for solver_candidate, opts in [(cp.CLARABEL, {'tol_gap_abs': 1e-4, 'tol_gap_rel': 1e-4}), (cp.SCS, {'max_iters': 2500})]:
        try:
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=solver_candidate, **opts)
            if prob.status in ["optimal", "optimal_inaccurate"] and w.value is not None:
                return w.value
        except Exception:
            continue
        
    fallback_obj = cp.Minimize(portfolio_risk)
    prob_fallback = cp.Problem(fallback_obj, [cp.sum(w) == 1.0, w >= 0.0])
    try:
        prob_fallback.solve(solver=cp.CLARABEL)
        if w.value is not None:
            return w.value
    except Exception:
        pass
        
    return np.ones(N) / N

# ==============================================================================
# 3. BACKTEST SIMULATION ENGINE
# ==============================================================================

def run_hd_backtest(df_returns, benchmark_returns, lookback_window=252, rebalance_freq=21, lamb=1.0, tau=0.02):
    """Simulates performance vectors over scrolling lookback windows."""
    n_timesteps, n_assets = df_returns.shape
    
    strat_robust_returns = []
    strat_markowitz_returns = []
    strat_eq_returns = []          
    strat_spy_returns = []
    backtest_dates = []
    
    active_robust_counts = []
    active_markowitz_counts = []
    
    # FIX: Initialize all tracking weights to absolute zero as requested
    w_robust = np.zeros(n_assets)
    w_markowitz = np.zeros(n_assets)

    for t in range(lookback_window, n_timesteps):
        current_date = df_returns.index[t]
        daily_returns = df_returns.iloc[t].values
        
        # --- End of Day Asset Drift Adjustments ---
        if t > lookback_window:
            prev_day_returns = df_returns.iloc[t-1].values
            
            # Guard against drift operations if it is still empty (Pre-Rebalance)
            if w_robust.sum() > 1e-5:
                w_robust = w_robust * (1.0 + prev_day_returns)
                w_robust = w_robust / w_robust.sum()
            
            if w_markowitz.sum() > 1e-5:
                w_markowitz = w_markowitz * (1.0 + prev_day_returns)
                w_markowitz = w_markowitz / w_markowitz.sum()

        # --- Rebalancing Matrix Intersections ---
        if (t - lookback_window) % rebalance_freq == 0:
            window_returns = df_returns.iloc[t - lookback_window:t]
            scenarios = window_returns.values
            
            raw_gerber = robust_gerber_covariance_mad(window_returns, c=0.5)
            cleaned_cov = de_noise_covariance(raw_gerber, scenarios.shape[0], scenarios.shape[1])
            
            # FIX: Swapped out log formulas for proper linear annualized arithmetic returns
            mu_ann = window_returns.mean(axis=0).values * 252
            
            # Strategy 1: Robust Turnover Constrained
            try:
                w_opt_r = optimize_portfolio_hd_cvar(
                    historical_scenarios=scenarios, cov_matrix=cleaned_cov, mu=mu_ann,
                    w_initial=w_robust, lamb=lamb, tau=tau, use_cvar=True
                )
                w_robust = np.array(w_opt_r).flatten()
                w_robust[w_robust < 1e-3] = 0.0
                w_robust /= w_robust.sum()
            except Exception:
                if w_robust.sum() < 1e-5: # Fallback configuration if day 1 optimization fails
                    w_robust = np.ones(n_assets) / n_assets

            # Strategy 2: Pure Classical Markowitz (FIX: Removed CVaR Constraint Layer)
            try:
                w_opt_m = optimize_portfolio_hd_cvar(
                    historical_scenarios=scenarios, cov_matrix=cleaned_cov, mu=mu_ann,
                    w_initial=None, lamb=lamb, tau=tau, use_cvar=False
                )
                w_markowitz = np.array(w_opt_m).flatten()
                w_markowitz[w_markowitz < 1e-3] = 0.0
                w_markowitz /= w_markowitz.sum()
            except Exception:
                if w_markowitz.sum() < 1e-5:
                    w_markowitz = np.ones(n_assets) / n_assets

        # Handle early tracking if rebalance has not generated allocations yet
        rec_robust = np.dot(w_robust, daily_returns) if w_robust.sum() > 1e-5 else 0.0
        rec_markowitz = np.dot(w_markowitz, daily_returns) if w_markowitz.sum() > 1e-5 else 0.0

        strat_robust_returns.append(rec_robust)
        strat_markowitz_returns.append(rec_markowitz)
        strat_eq_returns.append(df_returns.iloc[t].mean())
        strat_spy_returns.append(benchmark_returns.loc[current_date])
        
        backtest_dates.append(current_date)
        active_robust_counts.append(np.sum(w_robust > 0.005))
        active_markowitz_counts.append(np.sum(w_markowitz > 0.005))

    res_df = pd.DataFrame({
        "Strategy_Robust": strat_robust_returns,
        "Strategy_Markowitz": strat_markowitz_returns,
        "Strategy_EqualWeight": strat_eq_returns,
        "Benchmark_SPY": strat_spy_returns,
        "Active_Assets_Robust": active_robust_counts,
        "Active_Assets_Markowitz": active_markowitz_counts
    }, index=backtest_dates)
    
    return res_df

# ==============================================================================
# 4. DRIVER EXECUTION AND DATA COMPILATION
# ==============================================================================

if __name__ == "__main__":
    START_DATE = "2016-01-01" 
    END_DATE = "2026-01-01"
    CHOSEN_LAMBDA = 0.05  # Balanced perfectly now due to covariance matrix annualization fixes
    CHOSEN_TAU = 0.005    # Scaled down to realistically allow execution away from starting coordinates

    ticker_string = '''
        SPY QQQ DIA IWM VTI VO IVV IJH IJR
        MGK VUG SCHG IWF QTEC ONEQ SCHA VBK IWO
        XLK XLY XLV XLF XLI XLC SMH XBI ITA
        IGV SKYY CLOU CIBR HACK HERO GAMR IPAY FINX
        BOTZ ROBO ARKK ARKW ARKG ARKF MOO PBW TAN
        FAN LIT COPX GDX GDXJ REM VNQ IYR KRE
        QQQE RSP SPLV USMV QUAL MTUM VIG NOBL SDY
        EFA EEM VGK EWJ FXI MCHI ASHR INDA EPI PIN
        EWC EWA EWG EWQ EWT EWY EWW TUR THD DXJ
        BRK-B MSFT AAPL NVDA GOOGL AMZN META AVGO LLY V
        FAN LIT COPX GDX GDXJ GLD SLV REM VNQ IYR
    '''

    etfs_100 = list(sorted(list(set(ticker_string.split()))))
    print(f"Loaded sandbox matrix with {len(etfs_100)} multi-asset components.")
    
    scraper_session = Session(impersonate="chrome")
    
    print("Downloading 100 ETF Data Matrix from Yahoo Finance...")
    df_raw = yf.download(etfs_100, start=START_DATE, end=END_DATE, auto_adjust=True, session=scraper_session)
    df_close = df_raw['Close'] if 'Close' in df_raw.columns else df_raw.xs('Close', axis=1, level=0)
    
    initial_cols = df_close.shape[1]
    df_close = df_close.ffill().bfill()
    df_close = df_close.dropna(axis=1, how='all')
    final_cols = df_close.shape[1]
    
    print(f"Data Matrix parsed. {final_cols} out of {initial_cols} assets retained successfully.")
    
    if "SPY" not in df_close.columns:
        raise KeyError("Data Integrity Loss: 'SPY' benchmark column was dropped from the metrics matrix.")

    df_returns = df_close.pct_change().dropna()
    spy_benchmark = df_returns["SPY"].copy()

    print("\nRunning High-Dimensional Multi-Asset Optimizations...")
    res = run_hd_backtest(df_returns, spy_benchmark, lamb=CHOSEN_LAMBDA, tau=CHOSEN_TAU)

    # Performance Evaluation Summary
    metrics = {}
    target_strategies = ["Strategy_Robust", "Strategy_Markowitz", "Strategy_EqualWeight", "Benchmark_SPY"]
    for column in target_strategies:
        rets = res[column].values
        tot_ret = (1 + rets).prod() - 1
        ann_ret = ((1 + tot_ret) ** (252 / len(rets))) - 1
        ann_vol = np.std(rets, ddof=1) * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        
        cum_rets = (1 + rets).cumprod()
        running_max = np.maximum.accumulate(cum_rets)
        max_dd = np.min((cum_rets - running_max) / running_max)
        
        metrics[column] = {
            "Total Return (%)": tot_ret * 100,
            "Annualized Return (%)": ann_ret * 100,
            "Annualized Volatility (%)": ann_vol * 100,
            "Sharpe Ratio": sharpe,
            "Max Drawdown (%)": max_dd * 100
        }
        
    print("\n" + "="*85)
    print("                 COMPREHENSIVE BACKTEST RUNTIME SUMMARY")
    print("="*85)
    print(pd.DataFrame(metrics).round(2).to_string())
    print("="*85 + "\n")

    # Visualization Generation Pipeline
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(res.index, (1 + res["Strategy_Robust"]).cumprod() - 1, label="Strategy 1: Robust Turnover Constrained", color="#1f77b4", linewidth=2.5)
    ax1.plot(res.index, (1 + res["Strategy_Markowitz"]).cumprod() - 1, label="Strategy 2: True Classic Markowitz Baseline", color="#d62728", linestyle=":", linewidth=2.2)
    ax1.plot(res.index, (1 + res["Strategy_EqualWeight"]).cumprod() - 1, label="Strategy 3: Equal-Weighted 1/N Portfolio", color="grey", linestyle="-.", alpha=0.7)
    ax1.plot(res.index, (1 + res["Benchmark_SPY"]).cumprod() - 1, label="Strategy 4: S&P 500 Index Benchmark (SPY)", color="black", linestyle="--", linewidth=1.5)
    
    ax1.set_title(f"High-Dimensional Portfolio Optimization Timeline (100 ETFs Matrix | Lambda={CHOSEN_LAMBDA}, Tau={CHOSEN_TAU})", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Historical Timeline", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Cumulative Growth Return (%)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    # Secondary Asset Sparsity Overlay
    ax2 = ax1.twinx()
    ax2.plot(res.index, res["Active_Assets_Robust"], color="#1f77b4", alpha=0.15, linestyle="-")
    ax2.plot(res.index, res["Active_Assets_Markowitz"], color="#d62728", alpha=0.15, linestyle="--")
    ax2.set_ylabel("Number of Underlying Active Assets Selected", color="darkgreen", fontsize=11, fontweight="bold")
    ax2.tick_params(axis='y', labelcolor="darkgreen")

    plt.tight_layout()
    plt.savefig("ETF_backtest.png", dpi=300)
    print("Simulation complete. Performance visualization saved to ETF_backtest.png")
    plt.show()