import os
# Force single-threaded context for thread pooling over small chunks.
# This eliminates the KMeans memory leak warning on Windows architectures.
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import cvxpy as cp
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from curl_cffi.requests import Session

# ==============================================================================
# 1. ROBUST COVARIANCE ESTIMATORS & MATRIX CLEANING
# ==============================================================================

def nearest_positive_definite(matrix, eps=1e-4):
    """Computes the nearest Special Positive Devinite (SPD) matrix."""
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
    
    sample_std = np.std(returns_matrix, axis=0, ddof=1)
    gerber_cov = np.diag(sample_std) @ G @ np.diag(sample_std)
    
    return nearest_positive_definite(gerber_cov)


def de_noise_covariance(cov_matrix, T, N):
    """Marchenko-Pastur De-noising Framework tuned for high-dimensions."""
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sigma_sq = np.mean(eigenvalues)
    lambda_plus = sigma_sq * (1.0 + np.sqrt(N / T)) ** 2
    
    is_noise = eigenvalues <= lambda_plus
    if np.any(is_noise):
        avg_noise_eigenvalue = np.mean(eigenvalues[is_noise])
        eigenvalues[is_noise] = avg_noise_eigenvalue
        
    de_noised_cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return nearest_positive_definite(de_noised_cov)

# ==============================================================================
# 2. NESTED CLUSTERED OPTIMIZATION (NCO FRAMEWORK)
# ==============================================================================

def nested_clustered_optimization(returns, base_cov):
    """Executes NCO using true distance metrics on asset correlations."""
    T, N = returns.shape
    cov_cleaned = de_noise_covariance(base_cov, T, N)
    
    std_devs = np.sqrt(np.diagonal(cov_cleaned))
    corr_matrix = cov_cleaned / np.outer(std_devs, std_devs)
    corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
    
    dist_matrix = np.sqrt(0.5 * (1.0 - corr_matrix))
    
    best_k, best_score = 3, -1.0
    max_k = min(12, N // 5)
    
    for k_test in range(3, max_k):  
        km = KMeans(n_clusters=k_test, random_state=42, n_init=3)
        labels = km.fit_predict(dist_matrix)
        score = silhouette_score(dist_matrix, labels)
        if score > best_score:
            best_score, best_k = score, k_test
            
    clf = KMeans(n_clusters=best_k, random_state=42, n_init=3)
    cluster_labels = clf.fit_predict(dist_matrix)
    
    w_intra = np.zeros((N, best_k))
    for k in range(best_k):
        cluster_idx = np.where(cluster_labels == k)[0]
        if len(cluster_idx) == 0: 
            continue
        
        sub_cov = cov_cleaned[np.ix_(cluster_idx, cluster_idx)]
        w_sub = cp.Variable(len(cluster_idx))
        
        max_intra_w = max(0.10, 3.0 / len(cluster_idx))
        prob = cp.Problem(
            cp.Minimize(cp.quad_form(w_sub, cp.psd_wrap(sub_cov))), 
            [cp.sum(w_sub) == 1.0, w_sub >= 0.0, w_sub <= max_intra_w]
        )
        prob.solve(solver=cp.CLARABEL)
        w_intra[cluster_idx, k] = w_sub.value if w_sub.value is not None else 1.0 / len(cluster_idx)
        
    v_reduced = w_intra.T @ cov_cleaned @ w_intra
    v_reduced = nearest_positive_definite(v_reduced)
    
    w_inter = cp.Variable(best_k)
    prob_inter = cp.Problem(
        cp.Minimize(cp.quad_form(w_inter, cp.psd_wrap(v_reduced))), 
        [cp.sum(w_inter) == 1.0, w_inter >= 0.0]
    )
    prob_inter.solve(solver=cp.CLARABEL)
    
    return w_intra @ (w_inter.value if w_inter.value is not None else np.ones(best_k) / best_k)

# ==============================================================================
# 3. HIGH-DIMENSIONAL CVAR PORTFOLIO OPTIMIZER WITH REGULARIZATION
# ==============================================================================

def optimize_portfolio_hd_cvar(historical_scenarios, cov_matrix, mu, w_initial, w_nco_target, lamb=0.15, tau=0.02):
    """Convex Portfolio Optimizer scaling CVaR with conditional dual-regularization routing."""
    S, N = historical_scenarios.shape
    scenarios_matrix = np.asarray(historical_scenarios)
    stabilized_cov = cov_matrix + np.eye(N) * 1e-5
    
    w = cp.Variable(N)
    zeta_95 = cp.Variable()
    zeta_99 = cp.Variable()
    z_95 = cp.Variable(S)
    z_99 = cp.Variable(S)
    
    mu_daily = mu / 252.0  
    
    portfolio_risk = 0.5 * cp.quad_form(w, cp.psd_wrap(stabilized_cov))
    scaled_return = lamb * (mu_daily @ w)
    
    # Conditional Regularization Mapping Selector
    if w_initial is not None:
        # Strategy 1 path: Turnover Constrained profile penalty
        norm_regularization = tau * cp.norm(w - w_initial, 1)
    else:
        # Strategy 2 path: Absolute Sparse L1 profile penalty
        norm_regularization = tau * cp.norm(w, 1)
        
    cluster_regularization = 0.2 * cp.norm(w - w_nco_target, 2)
    objective = cp.Minimize(portfolio_risk - scaled_return + norm_regularization + cluster_regularization)
    
    constraints = [
        cp.sum(w) == 1.0,
        w >= 0.0, 
        
        z_95 >= 0.0,
        (-scenarios_matrix @ w) - zeta_95 <= z_95,
        zeta_95 + (1.0 / (1.0 - 0.95)) * cp.mean(z_95) <= 0.045, 
        
        z_99 >= 0.0,
        (-scenarios_matrix @ w) - zeta_99 <= z_99,
        zeta_99 + (1.0 / (1.0 - 0.99)) * cp.mean(z_99) <= 0.060  
    ]
    
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-4, tol_gap_rel=1e-4)
    except Exception:
        prob.solve(solver=cp.SCS, max_iters=2000)
        
    if prob.status not in ["optimal", "optimal_inaccurate"] or w.value is None:
        fallback_obj = cp.Minimize(cp.norm(w - w_nco_target, 2) + norm_regularization)
        prob_fallback = cp.Problem(fallback_obj, [cp.sum(w) == 1.0, w >= 0.0, w <= 0.06])
        prob_fallback.solve(solver=cp.CLARABEL)
        
    return w.value if w.value is not None else w_nco_target

# ==============================================================================
# 4. BACKTEST SIMULATION ENGINE (4 STRATEGIES PARALLEL SIMULATION)
# ==============================================================================

def run_hd_backtest(df_returns, benchmark_returns, lookback_window=252, rebalance_freq=21, lamb=0.15, tau=0.02):
    """Simulates 4 independent strategic streams simultaneously over lookback frames."""
    n_timesteps, n_assets = df_returns.shape
    
    # Strategy Vector Allocations Setup
    strat_robust_returns = []
    strat_norm_returns = []
    strat_eq_returns = []          
    strat_spy_returns = []
    backtest_dates = []
    
    active_robust_counts = []
    active_norm_counts = []
    
    w_robust = np.ones(n_assets) / n_assets
    w_norm = np.ones(n_assets) / n_assets

    for t in range(lookback_window, n_timesteps):
        current_date = df_returns.index[t]
        daily_returns = df_returns.iloc[t].values
        
        # --- End of Day Asset Drift Adjustments ---
        if t > lookback_window:
            w_robust = w_robust * (1.0 + df_returns.iloc[t-1].values)
            w_robust = w_robust / w_robust.sum() if w_robust.sum() > 1e-5 else np.ones(n_assets) / n_assets
            
            w_norm = w_norm * (1.0 + df_returns.iloc[t-1].values)
            w_norm = w_norm / w_norm.sum() if w_norm.sum() > 1e-5 else np.ones(n_assets) / n_assets

        # --- Rebalancing Matrix Intersections ---
        if (t - lookback_window) % rebalance_freq == 0:
            window_returns = df_returns.iloc[t - lookback_window:t]
            scenarios = window_returns.values
            
            raw_gerber = robust_gerber_covariance_mad(window_returns, c=0.5)
            w_nco_target = nested_clustered_optimization(window_returns, raw_gerber)
            
            mu_ann = (np.prod(1 + window_returns, axis=0) ** (252 / lookback_window) - 1).values
            cleaned_cov = de_noise_covariance(raw_gerber, scenarios.shape[0], scenarios.shape[1])
            
            # Execute Strategy 1: Turnover Constrained
            w_opt_r = optimize_portfolio_hd_cvar(
                historical_scenarios=scenarios, cov_matrix=cleaned_cov, mu=mu_ann,
                w_initial=w_robust, w_nco_target=w_nco_target, lamb=lamb, tau=tau
            )
            w_robust = np.array(w_opt_r).flatten()
            w_robust[w_robust < 1e-3] = 0.0
            w_robust /= w_robust.sum()

            # Execute Strategy 2: Absolute Regularized Sparse
            w_opt_n = optimize_portfolio_hd_cvar(
                historical_scenarios=scenarios, cov_matrix=cleaned_cov, mu=mu_ann,
                w_initial=None, w_nco_target=w_nco_target, lamb=lamb, tau=tau
            )
            w_norm = np.array(w_opt_n).flatten()
            w_norm[w_norm < 1e-3] = 0.0
            w_norm /= w_norm.sum()

        strat_robust_returns.append(np.dot(w_robust, daily_returns))
        strat_norm_returns.append(np.dot(w_norm, daily_returns))
        strat_eq_returns.append(df_returns.iloc[t].mean())
        strat_spy_returns.append(benchmark_returns.loc[current_date])
        
        backtest_dates.append(current_date)
        active_robust_counts.append(np.sum(w_robust > 0.005))
        active_norm_counts.append(np.sum(w_norm > 0.005))

    res_df = pd.DataFrame({
        "Strategy_Robust": strat_robust_returns,
        "Strategy_AbsoluteNorm": strat_norm_returns,
        "Strategy_EqualWeight": strat_eq_returns,
        "Benchmark_SPY": strat_spy_returns,
        "Active_Assets_Robust": active_robust_counts,
        "Active_Assets_AbsoluteNorm": active_norm_counts
    }, index=backtest_dates)
    
    return res_df

# ==============================================================================
# 5. DRIVER EXECUTION AND DATA COMPILATION
# ==============================================================================

if __name__ == "__main__":
    START_DATE = "2016-01-01" 
    END_DATE = "2026-01-01"
    CHOSEN_LAMBDA = 1.0      
    CHOSEN_TAU = 0.2         

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

    # Performance Summaries Parsing Engine (Tracking all 4 strategic directions)
    metrics = {}
    target_strategies = ["Strategy_Robust", "Strategy_AbsoluteNorm", "Strategy_EqualWeight", "Benchmark_SPY"]
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
    print("                 COMPREHENSIVE 4-STRATEGY SANDBOX PERFORMANCE SUMMARY")
    print("="*85)
    print(pd.DataFrame(metrics).round(2).to_string())
    print("="*85 + "\n")

    # Render Visual Performance Output for all 4 profiles
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(res.index, (1 + res["Strategy_Robust"]).cumprod() - 1, label="Strategy 1: Robust Turnover Constrained (||w - w_drift||)", color="#1f77b4", linewidth=2.5)
    ax1.plot(res.index, (1 + res["Strategy_AbsoluteNorm"]).cumprod() - 1, label="Strategy 2: Absolute Regularized Baseline (||w||)", color="#d62728", linestyle=":", linewidth=2.2)
    ax1.plot(res.index, (1 + res["Strategy_EqualWeight"]).cumprod() - 1, label="Strategy 3: Equal-Weighted 1/N Portfolio", color="grey", linestyle="-.", alpha=0.7)
    ax1.plot(res.index, (1 + res["Benchmark_SPY"]).cumprod() - 1, label="Strategy 4: S&P 500 Index Benchmark (SPY)", color="black", linestyle="--", linewidth=1.5)
    
    ax1.set_title(f"High-Dimensional Portfolio Optimization Timeline (100 ETFs Matrix | Lambda={CHOSEN_LAMBDA}, Tau={CHOSEN_TAU})", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Historical Timeline", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Cumulative Growth Return (%)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    # Dual Sparsity Matrix Verification overlay
    ax2 = ax1.twinx()
    ax2.plot(res.index, res["Active_Assets_Robust"], color="#1f77b4", alpha=0.18, linestyle="-")
    ax2.plot(res.index, res["Active_Assets_AbsoluteNorm"], color="#d62728", alpha=0.18, linestyle="--")
    ax2.set_ylabel("Number of Underlying Active Assets Selected", color="darkgreen", fontsize=11, fontweight="bold")
    ax2.tick_params(axis='y', labelcolor="darkgreen")

    plt.tight_layout()
    plt.savefig("hd_portfolio_4_strategy_performance.png", dpi=300)
    print("Simulation complete. 4-Strategy performance summary saved to hd_portfolio_4_strategy_performance.png")
    plt.show()