import os
os.environ["OMP_NUM_THREADS"] = "2"

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
    symmetric_mat = (matrix + matrix.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric_mat)
    eigenvalues = np.maximum(eigenvalues, eps)
    spd_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return (spd_matrix + spd_matrix.T) / 2


def robust_gerber_covariance_mad(returns, c=0.5):
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
    """ Marchenko-Pastur De-noising Framework tuned for high-dimensions """
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
    T, N = returns.shape
    cov_cleaned = de_noise_covariance(base_cov, T, N)
    
    std_devs = np.sqrt(np.diagonal(cov_cleaned))
    corr_matrix = cov_cleaned / np.outer(std_devs, std_devs)
    corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
    
    # Scale up search limits for 100 assets to find structural multi-collinear roots
    best_k, best_score = 3, -1.0
    for k_test in range(3, min(12, N // 5)):  
        km = KMeans(n_clusters=k_test, random_state=42, n_init=3)
        labels = km.fit_predict(corr_matrix)
        score = silhouette_score(corr_matrix, labels)
        if score > best_score:
            best_score, best_k = score, k_test
            
    clf = KMeans(n_clusters=best_k, random_state=42, n_init=3)
    cluster_labels = clf.fit_predict(corr_matrix)
    
    w_intra = np.zeros((N, best_k))
    for k in range(best_k):
        cluster_idx = np.where(cluster_labels == k)[0]
        if len(cluster_idx) == 0: continue
        
        sub_cov = cov_cleaned[np.ix_(cluster_idx, cluster_idx)]
        w_sub = cp.Variable(len(cluster_idx))
        
        # Enforce intra-cluster bounds to maintain structural diversity
        max_intra_w = max(0.10, 3.0 / len(cluster_idx))
        prob = cp.Problem(cp.Minimize(cp.quad_form(w_sub, sub_cov)), [cp.sum(w_sub) == 1.0, w_sub >= 0.0, w_sub <= max_intra_w])
        prob.solve(solver=cp.CLARABEL)
        w_intra[cluster_idx, k] = w_sub.value if w_sub.value is not None else 1.0 / len(cluster_idx)
        
    v_reduced = w_intra.T @ cov_cleaned @ w_intra
    w_inter = cp.Variable(best_k)
    prob_inter = cp.Problem(cp.Minimize(cp.quad_form(w_inter, v_reduced)), [cp.sum(w_inter) == 1.0, w_inter >= 0.0])
    prob_inter.solve(solver=cp.CLARABEL)
    
    return w_intra @ (w_inter.value if w_inter.value is not None else np.ones(best_k)/best_k)

# ==============================================================================
# 3. HIGH-DIMENSIONAL CVAR PORTFOLIO OPTIMIZER WITH REGULARIZATION
# ==============================================================================

def optimize_portfolio_hd_cvar(historical_scenarios, cov_matrix, mu, w_initial, w_nco_target, lamb=0.15, tau=0.02):
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
    turnover_regularization = tau * cp.norm(w - w_initial, 1)
    cluster_regularization = 0.10 * cp.norm(w - w_nco_target, 2) # Penalize deviation from NCO
    
    objective = cp.Minimize(portfolio_risk - scaled_return + turnover_regularization + cluster_regularization)
    
    constraints = [
        cp.sum(w) == 1.0,
        w >= 0.0,
        w <= 0.06,  # Strict concentration cap (Max 6% weight per ETF)
        
        z_95 >= 0.0,
        (-scenarios_matrix @ w) - zeta_95 <= z_95,
        zeta_95 + (1.0 / (1.0 - 0.95)) * cp.mean(z_95) <= 0.035, # 3.5% Daily Tail Cap
        
        z_99 >= 0.0,
        (-scenarios_matrix @ w) - zeta_99 <= z_99,
        zeta_99 + (1.0 / (1.0 - 0.99)) * cp.mean(z_99) <= 0.060  # 6.0% Daily Extreme Tail Cap
    ]
    
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-4, tol_gap_rel=1e-4)
    except Exception:
        prob.solve(solver=cp.ECOS)
        
    if prob.status not in ["optimal", "optimal_inaccurate"] or w.value is None:
        # Fallback tracking routine if constraints are breached during extreme market turns
        fallback_obj = cp.Minimize(cp.norm(w - w_nco_target, 2) + turnover_regularization)
        prob_fallback = cp.Problem(fallback_obj, [cp.sum(w) == 1.0, w >= 0.0, w <= 0.06])
        prob_fallback.solve(solver=cp.CLARABEL)
        
    return w.value if w.value is not None else w_nco_target

# ==============================================================================
# 4. BACKTEST SIMULATION ENGINE
# ==============================================================================

def run_hd_backtest(df_returns, benchmark_returns, lookback_window=252, rebalance_freq=21, lamb=0.15, tau=0.02):
    n_timesteps = len(df_returns)
    
    strat_robust_returns = []
    strat_eq_returns = []          
    backtest_dates = []
    active_asset_counts = []
    
    w_current = np.ones(df_returns.shape[1]) / df_returns.shape[1]

    for t in range(lookback_window, n_timesteps):
        current_date = df_returns.index[t]
        daily_returns = df_returns.iloc[t].values
        
        if t > lookback_window:
            w_current = w_current * (1.0 + df_returns.iloc[t-1].values)
            if w_current.sum() > 1e-5:
                w_current /= w_current.sum()
            else:
                w_current = np.ones(df_returns.shape[1]) / df_returns.shape[1]

        # Monthly Rebalancing Node
        if (t - lookback_window) % rebalance_freq == 0:
            window_returns = df_returns.iloc[t - lookback_window:t]
            scenarios = window_returns.values
            
            raw_gerber = robust_gerber_covariance_mad(window_returns, c=0.5)
            w_nco_target = nested_clustered_optimization(window_returns, raw_gerber)
            
            mu_ann = window_returns.mean().values * 252.0
            cleaned_cov = de_noise_covariance(raw_gerber, scenarios.shape[0], scenarios.shape[1])
            
            w_optimized = optimize_portfolio_hd_cvar(
                historical_scenarios=scenarios,
                cov_matrix=cleaned_cov,
                mu=mu_ann,
                w_initial=w_current,
                w_nco_target=w_nco_target,
                lamb=lamb,
                tau=tau
            )
            w_current = np.array(w_optimized).flatten()
            w_current[w_current < 1e-3] = 0.0
            w_current /= w_current.sum()

        strat_robust_returns.append(np.dot(w_current, daily_returns))
        strat_eq_returns.append(df_returns.iloc[t].mean())
        backtest_dates.append(current_date)
        active_asset_counts.append(np.sum(w_current > 0.005))

    res_df = pd.DataFrame({
        "Strategy_Robust": strat_robust_returns,
        "Strategy_EqualWeight": strat_eq_returns,
        "Benchmark_SPY": benchmark_returns.loc[backtest_dates].values.flatten(),
        "Active_Assets": active_asset_counts
    }, index=backtest_dates)
    
    return res_df

# ==============================================================================
# 5. DRIVER EXECUTION AND DATA COMPILATION
# ==============================================================================

if __name__ == "__main__":
    START_DATE = "2016-01-01" 
    END_DATE = "2026-01-01"
    CHOSEN_LAMBDA = 0.15      # Return optimization focus
    CHOSEN_TAU = 0.02         # Turnover friction anchor

    # Curated selection of exactly 100 highly liquid, non-overlapping institutional ETFs
    etfs_100 = [
        # Broad Market & Style Factors (20)
        "SPY", "QQQ", "DIA", "IWM", "VTV", "VUG", "IJH", "IJR", "OEF", "SPLV", 
        "MTUM", "VLUE", "DVY", "SDY", "VYM", "RSP", "MDY", "SCHX", "SCHA", "IUSG",
        # Core Economic Sectors (11)
        "XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLB", "XLRE", "XLU", "XLC",
        # Thematic Industries & Fixed Assets (24)
        "SMH", "SOXX", "IGV", "XSW", "IBB", "XBI", "KRE", "KIE", "XLF", "IYT", 
        "ITA", "XAR", "XHB", "ITB", "XRT", "XME", "GDX", "GDXJ", "XOP", "IYE", 
        "VNQ", "IYR", "REM", "IHF",
        # International & Regional Clusters (25)
        "EEM", "VWO", "IEMG", "EFA", "VEA", "IEFA", "VGK", "EZU", "EWJ", "DXJ", 
        "FXI", "MCHI", "ASHR", "INDA", "EWT", "EWY", "EWZ", "EWW", "EWC", "EWA", 
        "EWD", "EWG", "EWQ", "EWP", "EWI",
        # Fixed Income & Credit Structures (15)
        "TLT", "IEF", "SHY", "BIL", "AGG", "BND", "LQD", "HYG", "JNK", "EMB", 
        "VWOB", "TIP", "VTIP", "MUB", "SHV",
        # Alternatives & Commodities (5)
        "GLD", "SLV", "USO", "DBA", "GSG"
    ]
    
    # De-duplicate entries cleanly
    etfs_100 = list(sorted(list(set(etfs_100))))
    print(f"Verified Active Target Sandbox: {len(etfs_100)} unique structural ETFs.")
    
    scraper_session = Session(impersonate="chrome")
    
    print("Downloading 100 ETF Data Matrix from Yahoo Finance...")
    df_raw = yf.download(etfs_100, start=START_DATE, end=END_DATE, auto_adjust=True, session=scraper_session)
    df_close = df_raw['Close'] if 'Close' in df_raw.columns else df_raw.xs('Close', axis=1, level=0)
    
    # Drop columns containing missing history profiles
    df_close = df_close.dropna(axis=1, how='any')
    print(f"Data Matrix parsed. {df_close.shape[1]} ETFs match the full historical timeframe.")
    
    df_returns = df_close.pct_change().dropna()
    spy_benchmark = df_returns["SPY"].copy()

    print("\nRunning High-Dimensional Multi-Asset Optimizations...")
    res = run_hd_backtest(df_returns, spy_benchmark, lamb=CHOSEN_LAMBDA, tau=CHOSEN_TAU)

    # Compile Summary Dashboards
    metrics = {}
    for column in ["Strategy_Robust", "Strategy_EqualWeight", "Benchmark_SPY"]:
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
        
    print("\n" + "="*60 + "\n          100-ETF SANDBOX ENGINE PERFORMANCE SUMMARY\n" + "="*60)
    print(pd.DataFrame(metrics).round(2).to_string())
    print("="*60 + "\n")

    # Render Visual Performance Output
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(res.index, (1 + res["Strategy_Robust"]).cumprod() - 1, label="Robust Engine (100 ETFs Matrix)", color="#1f77b4", linewidth=2.5)
    ax1.plot(res.index, (1 + res["Strategy_EqualWeight"]).cumprod() - 1, label="Equal-Weighted Baseline (1/N)", color="grey", linestyle="-.")
    ax1.plot(res.index, (1 + res["Benchmark_SPY"]).cumprod() - 1, label="S&P 500 Index (SPY)", color="black", linestyle="--")
    ax1.set_title("High-Dimensional Portfolio Optimization Timeline (100 ETFs)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Historical Timeline", fontsize=11)
    ax1.set_ylabel("Cumulative Growth Return (%)", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.fill_between(res.index, res["Active_Assets"], color="green", alpha=0.03)
    ax2.set_ylabel("Number of Underlying Active Assets Selected", color="green", fontsize=11)
    ax2.tick_params(axis='y', labelcolor="green")

    plt.tight_layout()
    plt.savefig("hd_portfolio_performance.png", dpi=300)
    plt.show()