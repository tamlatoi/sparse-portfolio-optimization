import os
os.environ["OMP_NUM_THREADS"] = "2"

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import cvxpy as cp
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ==============================================================================
# 1. ROBUST COVARIANCE ESTIMATORS & MATRIX CLEANING
# ==============================================================================

def nearest_positive_definite(matrix, eps=1e-4):
    symmetric_mat = (matrix + matrix.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric_mat)
    eigenvalues = np.maximum(eigenvalues, eps)
    spd_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return (spd_matrix + spd_matrix.T) / 2


def robust_gerber_covariance_mad(returns, c=0.4):
    T, N = returns.shape
    returns_matrix = np.asarray(returns)
    
    med = np.median(returns_matrix, axis=0)
    mad = np.median(np.abs(returns_matrix - med), axis=0)
    thresholds = c * mad
    
    U = (returns_matrix >= thresholds).astype(float)
    D = (returns_matrix <= -thresholds).astype(float)
    
    N_UU = U.T @ U
    N_DD = D.T @ D
    N_UD = U.T @ D
    N_DU = D.T @ U
    
    N_CONC = N_UU + N_DD
    N_DISC = N_UD + N_DU
    
    denominator = T - ((1.0 - U) * (1.0 - D)).T @ ((1.0 - U) * (1.0 - D))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        G = np.where(denominator > 0, (N_CONC - N_DISC) / denominator, 0.0)
    np.fill_diagonal(G, 1.0)
    
    sample_std = np.std(returns_matrix, axis=0, ddof=1)
    gerber_cov = np.diag(sample_std) @ G @ np.diag(sample_std)
    
    return nearest_positive_definite(gerber_cov)


def de_noise_covariance(cov_matrix, T, N):
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
    
    best_k, best_score = 2, -1.0
    for k_test in range(2, min(8, N)):
        km = KMeans(n_clusters=k_test, random_state=42, n_init=5)
        labels = km.fit_predict(corr_matrix)
        score = silhouette_score(corr_matrix, labels)
        if score > best_score:
            best_score, best_k = score, k_test
            
    clf = KMeans(n_clusters=best_k, random_state=42, n_init=5)
    cluster_labels = clf.fit_predict(corr_matrix)
    
    w_intra = np.zeros((N, best_k))
    for k in range(best_k):
        cluster_idx = np.where(cluster_labels == k)[0]
        if len(cluster_idx) == 0: continue
        
        sub_cov = cov_cleaned[np.ix_(cluster_idx, cluster_idx)]
        w_sub = cp.Variable(len(cluster_idx))
        max_intra_w = max(0.15, 2.0 / len(cluster_idx))
        prob = cp.Problem(cp.Minimize(cp.quad_form(w_sub, sub_cov)), [cp.sum(w_sub) == 1.0, w_sub >= 0.0, w_sub <= max_intra_w])
        prob.solve(solver=cp.CLARABEL)
        w_intra[cluster_idx, k] = w_sub.value
        
    v_reduced = w_intra.T @ cov_cleaned @ w_intra
    w_inter = cp.Variable(best_k)
    prob_inter = cp.Problem(cp.Minimize(cp.quad_form(w_inter, v_reduced)), [cp.sum(w_inter) == 1.0, w_inter >= 0.0])
    prob_inter.solve(solver=cp.CLARABEL)
    
    return w_intra @ w_inter.value

# ==============================================================================
# 3. PURE TURNOVER CONSTRAINED CVAR OPTIMIZER
# ==============================================================================

def optimize_portfolio_cvar(historical_scenarios, cov_matrix, mu, w_initial, lamb=1.0, tau=0.0010):
    S, N = historical_scenarios.shape
    scenarios_matrix = np.asarray(historical_scenarios)
    
    cov_matrix = np.nan_to_num(cov_matrix)
    mu = np.nan_to_num(mu)
    stabilized_cov = cov_matrix + np.eye(N) * 1e-6
    
    w = cp.Variable(N)
    zeta_95 = cp.Variable()
    zeta_99 = cp.Variable()
    z_95 = cp.Variable(S)
    z_99 = cp.Variable(S)
    
    mu_daily = mu / 252.0  
    
    portfolio_risk = 0.5 * cp.quad_form(w, cp.psd_wrap(stabilized_cov))
    scaled_return = lamb * (mu_daily @ w)
    turnover_regularization = tau * cp.norm(w - w_initial, 1)
    
    objective = cp.Minimize(portfolio_risk - scaled_return + turnover_regularization)
    
    constraints = [
        cp.sum(w) == 1.0,
        w >= 0.0,
        
        z_95 >= 0.0,
        (-scenarios_matrix @ w) - zeta_95 <= z_95,
        zeta_95 + (1.0 / (1.0 - 0.95)) * cp.mean(z_95) <= 0.12, 
        
        z_99 >= 0.0,
        (-scenarios_matrix @ w) - zeta_99 <= z_99,
        zeta_99 + (1.0 / (1.0 - 0.99)) * cp.mean(z_99) <= 0.20
    ]
    
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-5, tol_gap_rel=1e-5)
    except Exception:
        try:
            prob.solve(solver=cp.ECOS)
        except Exception:
            pass
            
    if prob.status not in ["optimal", "optimal_inaccurate"] or w.value is None:
        fallback_objective = cp.Minimize(portfolio_risk + turnover_regularization)
        prob_fallback = cp.Problem(fallback_objective, [cp.sum(w) == 1.0, w >= 0.0])
        prob_fallback.solve(solver=cp.CLARABEL)
        
    return w.value

# ==============================================================================
# 4. SIMULATION BACKTEST ENGINE
# ==============================================================================

def run_sp500_backtest(df_returns, sp500_returns, lookback_window=252, rebalance_freq=21, lamb=1.0, tau=0.0010):
    n_timesteps, n_assets = df_returns.shape

    strat_robust_returns = []
    strat_eq_returns = []          
    active_robust_count = []
    w_current = np.ones(n_assets) / n_assets
    backtest_dates = []

    active_scenarios = np.zeros((lookback_window, n_assets))
    gerber_cov = np.eye(n_assets)
    active_mu = np.zeros(n_assets)

    for t in range(lookback_window, n_timesteps):
        current_date = df_returns.index[t]
        daily_returns = df_returns.iloc[t].values

        if t > lookback_window:
            w_current = w_current * (1.0 + df_returns.iloc[t-1].values)
            sum_drift = np.sum(w_current)
            w_current = w_current / sum_drift if sum_drift > 1e-5 else np.ones(n_assets) / n_assets

        if (t - lookback_window) % rebalance_freq == 0:
            window_data = df_returns.iloc[t - lookback_window:t]
            active_mask = (window_data.std() > 1e-7) & (~window_data.iloc[-1].isna())
            active_indices = np.where(active_mask)[0]
            
            if len(active_indices) > 20:  
                active_window_returns = window_data.iloc[:, active_indices]
                active_scenarios = active_window_returns.values
                
                raw_gerber = robust_gerber_covariance_mad(active_window_returns, c=0.4)
                w_nco = nested_clustered_optimization(active_window_returns, raw_gerber)
                
                active_mu = active_window_returns.mean().values * 252.0 * w_nco
                gerber_cov = de_noise_covariance(raw_gerber, active_scenarios.shape[0], active_scenarios.shape[1])
                
                if t == lookback_window:
                    w_initial_active = np.zeros(len(active_indices))
                else:
                    w_initial_active = w_current[active_indices]
                    if np.sum(w_initial_active) > 1e-5:
                        w_initial_active = w_initial_active / np.sum(w_initial_active)
                    else:
                        w_initial_active = np.ones(len(active_indices)) / len(active_indices)
                
                try:
                    optimized_w_active = optimize_portfolio_cvar(
                        historical_scenarios=active_scenarios,
                        cov_matrix=gerber_cov,
                        mu=active_mu,
                        w_initial=w_initial_active,
                        lamb=lamb,
                        tau=tau
                    )
                    
                    if optimized_w_active is not None:
                        optimized_w_active = np.array(optimized_w_active).flatten()
                        optimized_w_active[np.abs(optimized_w_active) < 1e-3] = 0.0
                        
                        w_new_master = np.zeros(n_assets)
                        w_new_master[active_indices] = optimized_w_active
                        
                        if np.sum(w_new_master) > 1e-4:
                            w_current = w_new_master / np.sum(w_new_master)
                except Exception:
                    pass

        ret_robust = np.dot(w_current, daily_returns)
        
        today_active_mask = ~df_returns.iloc[t].isna()
        w_eq = np.zeros(n_assets)
        w_eq[today_active_mask] = 1.0 / np.sum(today_active_mask)
        ret_eq = np.dot(w_eq, daily_returns)

        strat_robust_returns.append(ret_robust)
        strat_eq_returns.append(ret_eq)
        backtest_dates.append(current_date)
        active_robust_count.append(np.sum(w_current > 0.001))

    results_df = pd.DataFrame({
        "Strategy_Robust": strat_robust_returns,
        "Strategy_EqualWeight": strat_eq_returns,
        "Active_Assets_Robust": active_robust_count
    }, index=backtest_dates)

    results_df["SP500"] = sp500_returns.loc[results_df.index]
    return results_df

# ==============================================================================
# 5. EXECUTION DRIVER WITH TARGETED CO-ORDINATES & PLOTTING
# ==============================================================================

if __name__ == "__main__":
    # CHOOSE YOUR FIXED TUNING VALUES HERE
    CHOSEN_LAMBDA = 0.1
    CHOSEN_TAU = 0.5

    sp500_string = """
    AAPL MSFT NVDA AMZN GOOGL GOOG AVGO TSLA META MU BRK-B LLY WMT AMD JPM INTC V XOM JNJ ORCL
    AMAT LRCX CSCO CAT MA COST BAC ABBV GE UNH MS CVX PG KLAC KO HD GS NFLX PLTR GEV TXN MRK
    PM MRVL WDC DELL WFC STX RTX C QCOM LIN PANW IBM AXP ANET ADI APH MCD TMUS PEP VZ AMGN
    NEE TJX DIS BA CRWD GLW LOW NKE SBUX BK HWM GEHC CMCSA SYK INTU BLDR FIS CRM DHR ELV ISRG
    CI CB LMT UPS ABT BLK ACN NOW MDT FI REGN HON PLD LULU DE BSX SCHW ADSK MO COR ETN ECL TT
    WELL ITW VRTX FTNT FDX COF HCA NVR CTAS AJG AON BMY TRV FICO EMR PGR MCO NOC GD FCX MET NSC
    CEG GWW RMD NXPI TFC ORLY SRE MCK CME FSLR STLD WM MAR AMP MPC GILD GIS LHX JCI NUE ADM WBD
    AEE EOG PCG MSI CNC STT DXCM PSX CPS PAYX TRGP SYY DFS HLT PPW KMI KDP PCAR NEM ALGN CINF
    HEI PEG FTV AEP CDW PRU ALL WST FITB DUK GDDY EXR VLO VTRS FAST KR ED FE BAX O PAYC
    SO ODFL DLR LNT DLTR SBAC VMC IDXX KEYS A CMI DOW CTRA AWK HIG EQT LUV DG KSB OTIS OKE
    WEC WTW GRMN TSCO AVB K URI GPN HRL CHD BG KEY CTSH GL INVH EBAY HPQ DOV TSN RJF BRO SWKS
    CAH KMX APA BBY WBA JKHY HAS HOLX BEN IP CLX ESS MGM WRB TYL ALK NI ATO TECH TFX NDAQ
    AKAM IPG REG CRL NTAP GEN DRI POOL PODD MAS EVRG LKQ JBHT FRT DPZ CNP CPRT AES AAP SWK
    MHK RE NWL SEE XRAY LUMN CZR GNRC NXP PENN FLS SLG VNT XRX PTC MRNA BBWI FDS WU MOH
    CPT VICI ON IPGP UA UAA CSGP ACGL VLTO DXC HUBB JBL UBER OGN DAY DOC DECK SMCI WHR
    ZION SOLV VFC VST KKR CMA ILMN RHI SW WSM ERIE TPL MRO APO LII WDAY DASH EXE TKO BWA
    CE FMC COIN DDOG JNPR TTD ANSS HES PSKY IBKR APP EME HOOD SOLS Q EMN FISV
    """
    
    raw_tickers = list(set([t.strip() for t in sp500_string.split() if t.strip()]))
    print(f"Ingesting data target profiles for {len(raw_tickers)} equity tickers...")

    df_bench_raw = yf.download("^GSPC", start="2016-01-01", end="2026-01-01", auto_adjust=True, progress=False)
    if isinstance(df_bench_raw.columns, pd.MultiIndex):
        sp500_series = df_bench_raw.xs('Close', axis=1, level=0).squeeze()
    else:
        sp500_series = df_bench_raw['Close'].squeeze()
    sp500_series = sp500_series.ffill()

    chunk_size = 40
    valid_stock_series = {}
    for i in range(0, len(raw_tickers), chunk_size):
        chunk = raw_tickers[i:i + chunk_size]
        df_chunk_raw = yf.download(chunk, start="2016-01-01", end="2026-01-01", auto_adjust=True, progress=False)
        if isinstance(df_chunk_raw.columns, pd.MultiIndex):
            df_close = df_chunk_raw.xs('Close', axis=1, level=0)
        else:
            df_close = df_chunk_raw['Close'] if 'Close' in df_chunk_raw.columns else df_chunk_raw
            
        for ticker in chunk:
            if ticker in df_close.columns:
                valid_stock_series[ticker] = df_close[ticker].squeeze()

    df_stocks = pd.DataFrame(valid_stock_series)

    common_idx = df_stocks.index.intersection(sp500_series.index)
    df_stocks = df_stocks.loc[common_idx]
    sp500_series = sp500_series.loc[common_idx]

    stock_returns = df_stocks.ffill().pct_change().fillna(0.0)
    sp500_returns = sp500_series.pct_change().dropna()

    print(f"\nRunning target profile using Lambda={CHOSEN_LAMBDA}, Tau={CHOSEN_TAU}...")
    res = run_sp500_backtest(stock_returns, sp500_returns, lamb=CHOSEN_LAMBDA, tau=CHOSEN_TAU)

    # Calculate Cumulative Compounded Growth Matrices
    cum_robust = (1 + res["Strategy_Robust"]).cumprod() - 1
    cum_eq = (1 + res["Strategy_EqualWeight"]).cumprod() - 1
    cum_sp500 = (1 + res["SP500"]).cumprod() - 1

    # GENERATE DUAL-AXIS PERFORMANCE PLOT
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Left Axis: Cumulative Performance (Percentages)
    ax1.plot(res.index, cum_robust * 100, label="Robust Paper Portfolio", color="#1f77b4", linewidth=2.5)
    ax1.plot(res.index, cum_eq * 100, label="Equal-Weighted 1/N Portfolio", color="grey", linestyle="-.", alpha=0.8)
    ax1.plot(res.index, cum_sp500 * 100, label="S&P 500 Index Market (^GSPC)", color="black", linestyle="--", linewidth=2)
    ax1.set_xlabel("Historical Timeline", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Cumulative Performance Return (%)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    # Right Axis: Active Breadth Count Profile
    ax2 = ax1.twinx()
    ax2.fill_between(res.index, res["Active_Assets_Robust"], color="green", alpha=0.04, label="Robust Sparsity Selection Profile")
    ax2.set_ylabel("Number of Active Assets Selected", color="green", fontsize=11, fontweight="bold")
    ax2.tick_params(axis='y', labelcolor="green")

    plt.title(f"Integrated Strategy Vector Performance vs S&P 500 Benchmark Universe (Lambda={CHOSEN_LAMBDA}, Tau={CHOSEN_TAU})", fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    
    # Save chart locally
    plt.savefig("integrated_strategy_performance.png", dpi=300)
    print("\nSimulation complete. Performance graph saved to integrated_strategy_performance.png")
    plt.show()