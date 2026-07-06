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
    
    # Avoid zero division if MAD is perfectly flat
    thresholds = np.where(thresholds < 1e-6, 1e-6, thresholds)
    
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
    G = np.clip(G, -1.0, 1.0)
    
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
    
    # Map correlation space into true Euclidean distance metric space
    distance_matrix = np.sqrt(np.clip((1.0 - corr_matrix) / 2.0, 0.0, 1.0))
    
    best_k, best_score = 2, -1.0
    max_k = min(8, N - 1)
    if max_k <= 2:
        best_k = 2
    else:
        for k_test in range(2, max_k + 1):
            km = KMeans(n_clusters=k_test, random_state=42, n_init=5)
            labels = km.fit_predict(distance_matrix)
            score = silhouette_score(distance_matrix, labels)
            if score > best_score:
                best_score, best_k = score, k_test
            
    clf = KMeans(n_clusters=best_k, random_state=42, n_init=5)
    cluster_labels = clf.fit_predict(distance_matrix)
    
    w_intra = np.zeros((N, best_k))
    for k in range(best_k):
        cluster_idx = np.where(cluster_labels == k)[0]
        if len(cluster_idx) == 0: continue
        
        sub_cov = cov_cleaned[np.ix_(cluster_idx, cluster_idx)]
        w_sub = cp.Variable(len(cluster_idx))
        max_intra_w = max(0.15, 2.0 / len(cluster_idx))
        prob = cp.Problem(cp.Minimize(cp.quad_form(w_sub, cp.psd_wrap(sub_cov))), 
                          [cp.sum(w_sub) == 1.0, w_sub >= 0.0, w_sub <= max_intra_w])
        prob.solve(solver=cp.CLARABEL)
        w_intra[cluster_idx, k] = w_sub.value if w_sub.value is not None else 1.0 / len(cluster_idx)
        
    v_reduced = w_intra.T @ cov_cleaned @ w_intra
    v_reduced = nearest_positive_definite(v_reduced)
    
    w_inter = cp.Variable(best_k)
    prob_inter = cp.Problem(cp.Minimize(cp.quad_form(w_inter, cp.psd_wrap(v_reduced))), 
                               [cp.sum(w_inter) == 1.0, w_inter >= 0.0])
    prob_inter.solve(solver=cp.CLARABEL)
    
    inter_val = w_inter.value if w_inter.value is not None else np.ones(best_k) / best_k
    return w_intra @ inter_val

# ==============================================================================
# 3. BALANCED SCALED CVAR OPTIMIZER WITH NCO TRACKING
# ==============================================================================

def optimize_portfolio_cvar(historical_scenarios, cov_matrix, mu, w_initial, w_nco, lamb=1.0, tau=0.01):
    S, N = historical_scenarios.shape
    scenarios_matrix = np.asarray(historical_scenarios)
    
    cov_matrix = np.nan_to_num(cov_matrix)
    mu = np.nan_to_num(mu)
    stabilized_cov = cp.psd_wrap(cov_matrix + np.eye(N) * 1e-6)
    
    # Re-scale inputs entirely to Annual Space to prevent numerical ill-conditioning
    cov_annual = stabilized_cov * 252.0
    mu_annual = mu
    scenarios_annual = scenarios_matrix * np.sqrt(252.0)
    
    eq_weights = np.ones(N) / N
    equal_weight_scenarios = -scenarios_annual @ eq_weights
    base_95_cvar = np.percentile(equal_weight_scenarios, 95)
    base_99_cvar = np.percentile(equal_weight_scenarios, 99)
    
    max_allowable_95_cvar = max(0.12, base_95_cvar * 0.98)
    max_allowable_99_cvar = max(0.18, base_99_cvar * 0.98)
    
    w = cp.Variable(N)
    zeta_95 = cp.Variable()
    zeta_99 = cp.Variable()
    z_95 = cp.Variable(S)
    z_99 = cp.Variable(S)
    
    # Balanced Objective Terms
    portfolio_risk = 0.5 * cp.quad_form(w, cov_annual)
    scaled_return = lamb * (mu_annual @ w)
    
    # MODIFIED: Dynamically switch penalty structure based on configuration profile
    if w_initial is not None:
        # Strategy 1: Dynamic Turnover Penalty Matrix
        regularization = tau * cp.norm(w - w_initial, 1)
    else:
        # Strategy 2: Absolute ||w|| Standard Norm Penalty Strategy
        regularization = tau * cp.norm(w, 1)
        
    # Combined Framework Optimization. We use NCO as a quadratic tracking anchor.
    nco_tracking_penalty = 0.05 * cp.quad_form(w - w_nco, np.eye(N))
    
    objective = cp.Minimize(portfolio_risk - scaled_return + regularization + nco_tracking_penalty)
    
    constraints = [
        cp.sum(w) == 1.0,
        w >= 0.0,
        w <= 0.08, # Core diversification boundary condition across multi-collinear tokens
        
        z_95 >= 0.0,
        (-scenarios_annual @ w) - zeta_95 <= z_95,
        zeta_95 + (1.0 / (1.0 - 0.95)) * cp.mean(z_95) <= max_allowable_95_cvar, 
        
        z_99 >= 0.0,
        (-scenarios_annual @ w) - zeta_99 <= z_99,
        zeta_99 + (1.0 / (1.0 - 0.99)) * cp.mean(z_99) <= max_allowable_99_cvar
    ]
    
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-4, tol_gap_rel=1e-4)
    except Exception:
        try:
            prob.solve(solver=cp.SCS, max_iters=2500)
        except Exception:
            pass
            
    if prob.status not in ["optimal", "optimal_inaccurate"] or w.value is None:
        fallback_objective = cp.Minimize(portfolio_risk + regularization)
        prob_fallback = cp.Problem(fallback_objective, [cp.sum(w) == 1.0, w >= 0.0, w <= 0.08])
        prob_fallback.solve(solver=cp.CLARABEL)
        
    return w.value

# ==============================================================================
# 4. SIMULATION BACKTEST ENGINE
# ==============================================================================

def run_sp500_backtest(df_returns, sp500_returns, lookback_window=252, rebalance_freq=21, lamb=1.0, tau=0.01):
    n_timesteps, n_assets = df_returns.shape

    # Data Streams: Strategy 1 (Turnover Architecture)
    strat_robust_returns = []
    active_robust_count = []
    w_robust = np.ones(n_assets) / n_assets

    # Data Streams: Strategy 2 (Absolute Norm Architecture)
    strat_norm_returns = []
    active_norm_count = []
    w_norm = np.ones(n_assets) / n_assets

    # Benchmarks
    strat_eq_returns = []          
    backtest_dates = []

    for t in range(lookback_window, n_timesteps):
        current_date = df_returns.index[t]
        daily_returns = df_returns.iloc[t].values

        # --- 1. END-OF-DAY DRIFT ACCOUNTING ---
        if t > lookback_window:
            # Drift Strategy 1
            w_robust = w_robust * (1.0 + df_returns.iloc[t-1].values)
            sum_drift_r = np.sum(w_robust)
            w_robust = w_robust / sum_drift_r if sum_drift_r > 1e-5 else np.ones(n_assets) / n_assets
            
            # Drift Strategy 2
            w_norm = w_norm * (1.0 + df_returns.iloc[t-1].values)
            sum_drift_n = np.sum(w_norm)
            w_norm = w_norm / sum_drift_n if sum_drift_n > 1e-5 else np.ones(n_assets) / n_assets

        # --- 2. PERIODIC REBALANCING ---
        if (t - lookback_window) % rebalance_freq == 0:
            window_data = df_returns.iloc[t - lookback_window:t]
            active_mask = (window_data.std() > 1e-7) & (~window_data.iloc[-1].isna())
            active_indices = np.where(active_mask)[0]
            
            if len(active_indices) > 20:  
                active_window_returns = window_data.iloc[:, active_indices]
                active_scenarios = active_window_returns.values
                
                raw_gerber = robust_gerber_covariance_mad(active_window_returns, c=0.4)
                w_nco_active = nested_clustered_optimization(active_window_returns, raw_gerber)
                
                # Exact Geometric Compounding
                n_days = active_window_returns.shape[0]
                terminal_wealth = (1.0 + active_window_returns).prod(axis=0)
                terminal_wealth = np.maximum(1e-5, terminal_wealth.values)
                active_mu = (terminal_wealth) ** (252.0 / n_days) - 1.0
                
                gerber_cov = de_noise_covariance(raw_gerber, active_scenarios.shape[0], active_scenarios.shape[1])
                
                # Setup localized weight representations
                if t == lookback_window:
                    w_init_active_r = np.zeros(len(active_indices))
                else:
                    w_init_active_r = w_robust[active_indices]
                    w_init_active_r = w_init_active_r / np.sum(w_init_active_r) if np.sum(w_init_active_r) > 1e-5 else np.ones(len(active_indices)) / len(active_indices)
                
                # A. Run Optimization for Strategy 1 (Turnover Constraint: passes w_initial)
                try:
                    opt_w_robust = optimize_portfolio_cvar(
                        historical_scenarios=active_scenarios, cov_matrix=gerber_cov, mu=active_mu,
                        w_initial=w_init_active_r, w_nco=w_nco_active, lamb=lamb, tau=tau
                    )
                    if opt_w_robust is not None:
                        opt_w_robust = np.array(opt_w_robust).flatten()
                        opt_w_robust[np.abs(opt_w_robust) < 1e-3] = 0.0
                        w_new_m = np.zeros(n_assets)
                        w_new_m[active_indices] = opt_w_robust
                        if np.sum(w_new_m) > 1e-4:
                            w_robust = w_new_m / np.sum(w_new_m)
                except Exception:
                    pass

                # B. Run Optimization for Strategy 2 (Absolute Norm Constraint: w_initial=None)
                try:
                    opt_w_norm = optimize_portfolio_cvar(
                        historical_scenarios=active_scenarios, cov_matrix=gerber_cov, mu=active_mu,
                        w_initial=None, w_nco=w_nco_active, lamb=lamb, tau=tau
                    )
                    if opt_w_norm is not None:
                        opt_w_norm = np.array(opt_w_norm).flatten()
                        opt_w_norm[np.abs(opt_w_norm) < 1e-3] = 0.0
                        w_new_m = np.zeros(n_assets)
                        w_new_m[active_indices] = opt_w_norm
                        if np.sum(w_new_m) > 1e-4:
                            w_norm = w_new_m / np.sum(w_new_m)
                except Exception:
                    pass

        # --- 3. HARVEST SYSTEM RETURNS ---
        ret_robust = np.dot(w_robust, daily_returns)
        ret_norm = np.dot(w_norm, daily_returns)
        
        today_active_mask = ~df_returns.iloc[t].isna()
        w_eq = np.zeros(n_assets)
        w_eq[today_active_mask] = 1.0 / np.sum(today_active_mask)
        ret_eq = np.dot(w_eq, daily_returns)

        strat_robust_returns.append(ret_robust)
        strat_norm_returns.append(ret_norm)
        strat_eq_returns.append(ret_eq)
        backtest_dates.append(current_date)
        
        active_robust_count.append(np.sum(w_robust > 0.001))
        active_norm_count.append(np.sum(w_norm > 0.001))

    results_df = pd.DataFrame({
        "Strategy_Robust": strat_robust_returns,
        "Strategy_AbsoluteNorm": strat_norm_returns,
        "Strategy_EqualWeight": strat_eq_returns,
        "Active_Assets_Robust": active_robust_count,
        "Active_Assets_AbsoluteNorm": active_norm_count
    }, index=backtest_dates)

    results_df["SP500"] = sp500_returns.loc[results_df.index]
    return results_df

# ==============================================================================
# 5. EXECUTION DRIVER WITH TARGETED CO-ORDINATES & PLOTTING
# ==============================================================================

if __name__ == "__main__":
    CHOSEN_LAMBDA = 0.1
    CHOSEN_TAU = 0.05

    from curl_cffi.requests import Session
    scraper_session = Session(impersonate="chrome")

    sp500_string = """
    AAPL MSFT NVDA AMZN GOOGL GOOG AVGO TSLA META MU
    BRK-B LLY WMT AMD JPM INTC V XOM JNJ ORCL
    AMAT LRCX CSCO CAT MA COST BAC ABBV GE UNH
    MS CVX PG KLAC KO SNDK HD GS NFLX PLTR
    GEV TXN MRK PM MRVL WDC DELL WFC STX RTX
    C QCOM LIN PANW IBM AXP ANET ADI APH MCD
    TMUS PEP VZ AMGN NEE TJX DIS BA CRWD GLW
    LOW NKE SBUX BK HWM GEHC PM AXON CMCSA SYK
    INTU BLDR FIS AMD CRM DHR ELV ISRG CI CB
    LMT UPS ABT BLK ACN NOW MDT CAT FI REGN
    HON PLD LULU DE BSX SCHW ADSK QCOM MO COR
    ETN MU DE MDT ECL SYK ADI PANW BKR EW
    LRCX FIS APD PH SNPS ZTS TT WELL KLAC ITW
    VRTX FTNT FDX COF LRCX HCA NVR CTAS APD AJG
    TT AON BMY TRV FICO EMR PGR MCO NOC GD
    FCX MET NSC CEG GWW RMD NXPI TFC ORLY
    SRE SPLK MCK CME FSLR STLD WM MAR AMP MPC
    GILD GIS LHX JCI NUE ADSK ADM WBD FICO AEE
    EOG PCG MSI COF CNC STT DXCM PSX CPS SNPS
    PAYX TRGP PH SRE SYY DFS HLT PPW KMI KDP
    PCAR NEM ALGN CINF HEI PEG FTV AEP CDW PRU
    ALL WST FITB DUK GDDY EXR VLO VTRS FAST PCG
    TRMB ADM TRV VRSK CPAY STE KR ED FE BAX
    O PAYC SO ODFL DLR LNT ALGN DLTR SBAC WELL
    DLTR CEG AON EQR VMC IDXX KEYS A CMI DOW
    CTRA AWK DUK HIG EQT LUV DLR DG KSB OTIS
    OKE STT EXPE WEC WTW GRMN TSCO AVB K KSS
    URI GPN HRL BLK SYY CHD BG KEY CTSH GL
    INVH EBAY HPQ DOV FTV TSN FDX CE RJF BRO
    WST SWKS CAH CDW KMX APA BBY EXR VLO CTAS
    VMC EXPE WBA JKHY HAS HOLX BEN IP CLX ESS
    MGM WRB TYL ALK NI ATO TECH LNT TFX TECH
    NDAQ AKAM IPG REG CRL NTAP JKHY GEN DRI HRL
    POOL PODD MAS EVRG LKQ JBHT FRT DPZ CNP CPRT
    KMX AES AAP SWK MHK RE NWL SEE XRAY LUMN
    CZR GNRC NXP PENN FLS SLG VNT XRX PTC MRNA
    BBWI FDS SBNY SEDG HBI LEG WU WTW MOH CPT
    VICI KDP ON IPGP UA UAA ELV CSGP INVH EQT
    PCG CTXS DRE TRGP ACGL GEN FSLR STLD GEHC
    BG PODD FICO LUMN AXON RVTY FI PANW EG KVUE
    COR ABNB BX LNC NWL VLTO DXC BLDR HUBB JBL
    LULU UBER ALK ATVI OGN SEDG SEE DAY DOC DECK
    SMCI WHR ZION CPAY FLT GEV SOLV VFC XRAY VST
    PXD CRWD GDDY KKR CMA ILMN RHI SW WRK DELL
    ERIE TPL MRO APO LII WDAY DASH EXE TKO WSM
    BWA CE FMC COIN DDOG JNPR TTD ANSS HES
    PSKY IBKR WBA APP EME HOOD SOLS Q EMN FISV
    """
    
    raw_tickers = list(set([t.strip() for t in sp500_string.split() if t.strip()]))
    print(f"Ingesting data target profiles for {len(raw_tickers)} equity tickers...")

    df_bench_raw = yf.download("^GSPC", start="2016-01-01", end="2026-01-01", auto_adjust=True, progress=False, session=scraper_session)
    if isinstance(df_bench_raw.columns, pd.MultiIndex):
        sp500_series = df_bench_raw.xs('Close', axis=1, level=0).squeeze()
    else:
        sp500_series = df_bench_raw['Close'].squeeze()
    sp500_series = sp500_series.ffill()

    chunk_size = 40
    valid_stock_series = {}
    for i in range(0, len(raw_tickers), chunk_size):
        chunk = raw_tickers[i:i + chunk_size]
        try:
            df_chunk_raw = yf.download(chunk, start="2016-01-01", end="2026-01-01", auto_adjust=True, progress=False, session=scraper_session)
            if isinstance(df_chunk_raw.columns, pd.MultiIndex):
                df_close = df_chunk_raw.xs('Close', axis=1, level=0)
            else:
                df_close = df_chunk_raw['Close'] if 'Close' in df_chunk_raw.columns else df_chunk_raw
                
            for ticker in chunk:
                if ticker in df_close.columns:
                    valid_stock_series[ticker] = df_close[ticker].squeeze()
        except Exception as e:
            print(f"Skipping chunk due to structural response anomaly: {e}")

    df_stocks = pd.DataFrame(valid_stock_series)

    common_idx = df_stocks.index.intersection(sp500_series.index)
    df_stocks = df_stocks.loc[common_idx]
    sp500_series = sp500_series.loc[common_idx]

    stock_returns = df_stocks.ffill().pct_change(fill_method=None).fillna(0.0)
    sp500_returns = sp500_series.pct_change().dropna()

    print(f"\nRunning matrix simulations using Lambda={CHOSEN_LAMBDA}, Tau={CHOSEN_TAU}...")
    res = run_sp500_backtest(stock_returns, sp500_returns, lamb=CHOSEN_LAMBDA, tau=CHOSEN_TAU)

    # Calculate equity curves
    cum_robust = (1 + res["Strategy_Robust"]).cumprod() - 1
    cum_norm = (1 + res["Strategy_AbsoluteNorm"]).cumprod() - 1
    cum_eq = (1 + res["Strategy_EqualWeight"]).cumprod() - 1
    cum_sp500 = (1 + res["SP500"]).cumprod() - 1

    # ==============================================================================
    # 6. PERFORMANCE METRICS ENGINE
    # ==============================================================================
    
    def print_performance_metrics(returns_df):
        metrics = {}
        trading_days = 252
        target_columns = ["Strategy_Robust", "Strategy_AbsoluteNorm", "Strategy_EqualWeight", "SP500"]
        
        for column in target_columns:
            rets = returns_df[column].values
            
            total_ret = (1 + rets).prod()
            n_days = len(rets)
            ann_return = (total_ret) ** (trading_days / n_days) - 1
            ann_vol = np.std(rets, ddof=1) * np.sqrt(trading_days)
            sharpe = ann_return / ann_vol if ann_vol > 0 else 0
            
            cum_rets = (1 + rets).cumprod()
            running_max = np.maximum.accumulate(cum_rets)
            running_max = np.where(running_max == 0, 1.0, running_max)
            drawdowns = (cum_rets - running_max) / running_max
            max_dd = np.min(drawdowns)
            
            metrics[column] = {
                "Total Return (%)": (total_ret - 1) * 100,
                "Annualized Return (%)": ann_return * 100,
                "Annualized Volatility (%)": ann_vol * 100,
                "Sharpe Ratio": sharpe,
                "Max Drawdown (%)": max_dd * 100
            }
            
        summary_df = pd.DataFrame(metrics).round(2)
        print("\n" + "="*75)
        print("                 COMPREHENSIVE BACKTEST PERFORMANCE METRICS")
        print("="*75)
        print(summary_df.to_string())
        print("="*75 + "\n")
        return summary_df

    summary_metrics = print_performance_metrics(res)

    # ==============================================================================
    # 7. GENERATE MULTI-STRATEGY PERFORMANCE PLOT
    # ==============================================================================
    fig, ax1 = plt.subplots(figsize=(14, 7))

    ax1.plot(res.index, cum_robust * 100, label="Turnover Constrained Matrix (||w - w_drift||)", color="#1f77b4", linewidth=2.5)
    ax1.plot(res.index, cum_norm * 100, label="Absolute Regularized Baseline (||w||)", color="#d62728", linestyle=":", linewidth=2.2)
    ax1.plot(res.index, cum_eq * 100, label="Equal-Weighted 1/N Portfolio", color="grey", linestyle="-.", alpha=0.7)
    ax1.plot(res.index, cum_sp500 * 100, label="S&P 500 Index Market (^GSPC)", color="black", linestyle="--", linewidth=1.5)
    
    ax1.set_xlabel("Historical Timeline", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Cumulative Performance Return (%)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    # Background Sparsity Comparison
    ax2 = ax1.twinx()
    ax2.plot(res.index, res["Active_Assets_Robust"], color="#1f77b4", alpha=0.25, linestyle="-", label="Sparsity: w_drift")
    ax2.plot(res.index, res["Active_Assets_AbsoluteNorm"], color="#d62728", alpha=0.25, linestyle="--", label="Sparsity: Absolute Norm")
    ax2.set_ylabel("Number of Active Assets Selected", color="darkgreen", fontsize=11, fontweight="bold")
    ax2.tick_params(axis='y', labelcolor="darkgreen")

    plt.title(f"Fully Cleaned Unified Architecture Performance Engine (Lambda={CHOSEN_LAMBDA}, Tau={CHOSEN_TAU})", fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    
    plt.savefig("integrated_strategy_performance.png", dpi=300)
    print("Simulation complete. Performance graph saved to integrated_strategy_performance.png")
    plt.show()