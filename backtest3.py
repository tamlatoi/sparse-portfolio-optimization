import os
# Suppresses the known Windows multi-threading KMeans warning inside Anaconda MKL layers
os.environ["OMP_NUM_THREADS"] = "2"

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import cvxpy as cp
from sklearn.covariance import LedoitWolf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ==============================================================================
# 1. ROBUST COVARIANCE ESTIMATORS & MATRIX CLEANING
# ==============================================================================

def nearest_positive_definite(matrix, eps=1e-4):
    """
    Finds the nearest symmetric positive definite (s.p.d) matrix.
    Eliminates numerical non-invertibility or negative eigenvalue defects during T < N.
    """
    symmetric_mat = (matrix + matrix.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric_mat)
    eigenvalues = np.maximum(eigenvalues, eps)
    spd_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return (spd_matrix + spd_matrix.T) / 2


def robust_gerber_covariance_mad(returns, c=0.4):
    """
    Computes Robust Gerber Covariance using Median Absolute Deviation (MAD) thresholds
    as derived in Equation (5) of the research paper to clear out unstructured financial noise.
    """
    T, N = returns.shape
    returns_matrix = np.asarray(returns)
    
    # Compute MAD: median(|y_i - median(y)|)
    med = np.median(returns_matrix, axis=0)
    mad = np.median(np.abs(returns_matrix - med), axis=0)
    thresholds = c * mad
    
    # Structural mappings for upward and downward threshold breaches
    U = (returns_matrix >= thresholds).astype(float)
    D = (returns_matrix <= -thresholds).astype(float)
    
    N_UU = U.T @ U
    N_DD = D.T @ D
    N_UD = U.T @ D
    N_DU = D.T @ U
    
    N_CONC = N_UU + N_DD
    N_DISC = N_UD + N_DU
    
    # Paper Equation (5) Denominator: Filters zones where pairs remain static inside noise thresholds
    denominator = T - ((1.0 - U) * (1.0 - D)).T @ ((1.0 - U) * (1.0 - D))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        G = np.where(denominator > 0, (N_CONC - N_DISC) / denominator, 0.0)
    np.fill_diagonal(G, 1.0)
    
    # Translate correlation framework back to variance scale using sample standard deviations
    sample_std = np.std(returns_matrix, axis=0, ddof=1)
    gerber_cov = np.diag(sample_std) @ G @ np.diag(sample_std)
    
    return nearest_positive_definite(gerber_cov)


def de_noise_covariance(cov_matrix, T, N):
    """
    Applies the Marcenko-Pastur random matrix spectrum filter.
    Clips random noise eigenvalues below the calculated boundary lambda_plus.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sigma_sq = np.mean(eigenvalues)  # Estimator of underlying base variance
    lambda_plus = sigma_sq * (1.0 + np.sqrt(N / T)) ** 2  # Max noise boundary limit
    
    is_noise = eigenvalues <= lambda_plus
    if np.any(is_noise):
        avg_noise_eigenvalue = np.mean(eigenvalues[is_noise])
        eigenvalues[is_noise] = avg_noise_eigenvalue  # Spectral flattening injection
        
    de_noised_cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return nearest_positive_definite(de_noised_cov)

# ==============================================================================
# 2. NESTED CLUSTERED OPTIMIZATION (NCO FRAMEWORK VIA LOPEZ DE PRADO)
# ==============================================================================

def nested_clustered_optimization(returns, base_cov):
    """
    Executes Lopez de Prado's NCO Algorithm pipeline over the active universe matrix.
    Clusters asset groups to handle intra-sector collinearity cleanly.
    """
    T, N = returns.shape
    cov_cleaned = de_noise_covariance(base_cov, T, N)
    
    # Derive structural correlation matrix
    std_devs = np.sqrt(np.diagonal(cov_cleaned))
    corr_matrix = cov_cleaned / np.outer(std_devs, std_devs)
    corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
    
    # Compute optimal cluster number using unsupervised Silhouette optimization checks
    best_k, best_score = 2, -1.0
    for k_test in range(2, min(8, N)):
        km = KMeans(n_clusters=k_test, random_state=42, n_init=5)
        labels = km.fit_predict(corr_matrix)
        score = silhouette_score(corr_matrix, labels)
        if score > best_score:
            best_score, best_k = score, k_test
            
    clf = KMeans(n_clusters=best_k, random_state=42, n_init=5)
    cluster_labels = clf.fit_predict(corr_matrix)
    
    # Stage 1: Intra-cluster Optimization
    w_intra = np.zeros((N, best_k))
    for k in range(best_k):
        cluster_idx = np.where(cluster_labels == k)[0]
        if len(cluster_idx) == 0: continue
        
        sub_cov = cov_cleaned[np.ix_(cluster_idx, cluster_idx)]
        w_sub = cp.Variable(len(cluster_idx))
        prob = cp.Problem(cp.Minimize(cp.quad_form(w_sub, sub_cov)), [cp.sum(w_sub) == 1.0, w_sub >= 0.0])
        prob.solve(solver=cp.CLARABEL)
        w_intra[cluster_idx, k] = w_sub.value
        
    # Stage 2: Inter-cluster Optimization
    v_reduced = w_intra.T @ cov_cleaned @ w_intra  # Dimensionality compression
    w_inter = cp.Variable(best_k)
    prob_inter = cp.Problem(cp.Minimize(cp.quad_form(w_inter, v_reduced)), [cp.sum(w_inter) == 1.0, w_inter >= 0.0])
    prob_inter.solve(solver=cp.CLARABEL)
    
    return w_intra @ w_inter.value

# ==============================================================================
# 3. CONVEX OPTIMIZER WITH DUAL CVAR CONSTRAINTS & PORTFOLIO L1 TURNOVER PENALTIES
# ==============================================================================

def optimize_portfolio_cvar(historical_scenarios, cov_matrix, mu, w_initial, lambda_tc=0.0010, sparsity_beta=0.015):
    """
    Upgraded Optimizer: Native Alpha-Risk CVaR Optimizer.
    Combines dual-tail CVaR bounds, L1 sparsity penalties, and expected tracking returns.
    """
    S, N = historical_scenarios.shape
    scenarios_matrix = np.asarray(historical_scenarios)
    
    cov_matrix = np.nan_to_num(cov_matrix)
    mu = np.nan_to_num(mu)
    
    # Add minor ridge scaling value to guarantee strict positivity
    stabilized_cov = cov_matrix + np.eye(N) * 1e-6
    
    w = cp.Variable(N)
    zeta_95 = cp.Variable()
    zeta_99 = cp.Variable()
    z_95 = cp.Variable(S)
    z_99 = cp.Variable(S)
    
    # Scale annualized mu vector into a clean daily expectation matching scenarios
    mu_daily = mu / 252.0  
    expected_return = mu_daily @ w
    
    # Wrap covariance safely to prevent PSD matrix compilation failure flags
    portfolio_variance = cp.quad_form(w, cp.psd_wrap(stabilized_cov))
    
    # Structural regularizations
    transaction_cost_penalty = lambda_tc * cp.norm(w - w_initial, 1)
    sparsity_penalty = sparsity_beta * cp.norm(w, 1)
    
    # Objective function: Maximize daily returns net of variance risk & turnover friction
    objective = cp.Minimize(-expected_return + portfolio_variance + transaction_cost_penalty + sparsity_penalty)
    
    constraints = [
        cp.sum(w) == 1.0,
        w >= 0.0,
        
        # Dual Historical CVaR restrictions
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
            
    # Resilient fallback recovery routine to preserve weight outputs
    if prob.status not in ["optimal", "optimal_inaccurate"] or w.value is None:
        fallback_objective = cp.Minimize(portfolio_variance + sparsity_penalty)
        prob_fallback = cp.Problem(fallback_objective, [cp.sum(w) == 1.0, w >= 0.0])
        prob_fallback.solve(solver=cp.CLARABEL)
        
    return w.value

# ==============================================================================
# 4. SIMULATION BACKTEST ENGINE (ACTIVE UNIVERSE DYNAMIC ALLOCATION)
# ==============================================================================

def run_sp500_backtest(df_returns, sp500_returns, lookback_window=252, rebalance_freq=21, lambda_tc=0.0010):
    n_timesteps, n_assets = df_returns.shape
    print(f"Initializing Backtest Engine. Timesteps: {n_timesteps}, Master Universe Tickers: {n_assets}")

    strat_robust_returns = []
    strat_eq_returns = []          # <-- FIXED: Initialized missing equal weight collection array
    active_robust_count = []
    
    # Initialize Master Vector across ALL tickers
    w_current = np.ones(n_assets) / n_assets
    
    backtest_dates = []

    for t in range(lookback_window, n_timesteps):
        current_date = df_returns.index[t]
        daily_returns = df_returns.iloc[t].values

        # --- 1. DAILY INDEPENDENT WEIGHT DRIFT TRACKING ---
        if t > lookback_window:
            w_current = w_current * (1.0 + df_returns.iloc[t-1].values)
            sum_drift = np.sum(w_current)
            w_current = w_current / sum_drift if sum_drift > 1e-5 else np.ones(n_assets) / n_assets

        # --- 2. PERIODIC DYNAMIC REBALANCING EXECUTION ---
        if (t - lookback_window) % rebalance_freq == 0:
            print(f" -> Rebalancing Portfolio on: {current_date.strftime('%Y-%m-%d')}...")
            
            # Extract historical lookback data frame slice for this specific epoch
            window_data = df_returns.iloc[t - lookback_window:t]
            
            # DYNAMIC UNIVERSE LOOKUP: Filter out stocks without a valid timeline or with 0 variance
            active_mask = (window_data.std() > 1e-7) & (~window_data.iloc[-1].isna())
            active_indices = np.where(active_mask)[0]
            
            if len(active_indices) > 5:  # Require a minimum critical mass of survivors to group and optimize
                # Slice matrices cleanly to survivor-space dimension rules
                active_window_returns = window_data.iloc[:, active_indices]
                active_scenarios = active_window_returns.values
                active_mu = active_window_returns.mean().values * 252.0
                
                # Compute robust covariance estimators exclusively on survivors
                gerber_cov = robust_gerber_covariance_mad(active_window_returns, c=0.4)
                
                # NCO structuring pipeline over survivors
                try:
                    target_weights_nco = nested_clustered_optimization(active_window_returns, gerber_cov)
                except Exception:
                    # Inner fallback: Use scaled drifted profile for the active space if NCO fails
                    target_weights_nco = w_current[active_indices]
                    if np.sum(target_weights_nco) > 1e-5:
                        target_weights_nco = target_weights_nco / np.sum(target_weights_nco)
                    else:
                        target_weights_nco = np.ones(len(active_indices)) / len(active_indices)
                
                # Slice corresponding initial positions from our master drift tracker
                w_initial_active = w_current[active_indices]
                if np.sum(w_initial_active) > 1e-5:
                    w_initial_active = w_initial_active / np.sum(w_initial_active)
                else:
                    w_initial_active = np.ones(len(active_indices)) / len(active_indices)
                
                # Run the Convex Optimizer on the dynamic matrix slice
                try:
                    optimized_w_active = optimize_portfolio_cvar(
                        historical_scenarios=active_scenarios,
                        cov_matrix=gerber_cov,
                        mu=active_mu,
                        w_initial=w_initial_active,
                        lambda_tc=lambda_tc,
                        sparsity_beta=0.015
                    )
                    
                    if optimized_w_active is not None:
                        optimized_w_active = np.array(optimized_w_active).flatten()
                        optimized_w_active[np.abs(optimized_w_active) < 1e-3] = 0.0
                        
                        # RE-MAP BACK TO THE MASTER MATRIX: Un-selected/delisted companies get 0
                        w_new_master = np.zeros(n_assets)
                        w_new_master[active_indices] = optimized_w_active
                        
                        if np.sum(w_new_master) > 1e-4:
                            w_current = w_new_master / np.sum(w_new_master)
                except Exception as e:
                    print(f"    [Warning] Convex optimization exception at step {t}: {e}. Maintaining positions.")
                    pass
            else:
                print(f"    [Warning] Critical mass universe failure at step {t}. Maintaining drifted positions.")

        # --- 3. HARVEST SYSTEM PERFORMANCE DATA ---
        ret_robust = np.dot(w_current, daily_returns)
        
        # Unbiased daily Equal-Weight baseline: allocation spreads across today's active entities only
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


def evaluate_metrics(results_df):
    summary_table = {}
    strategies = {
        "Robust Paper Portfolio (Gerber MAD + NCO + CVaR)": "Strategy_Robust",
        "Equal-Weighted 1/N Baseline Strategy": "Strategy_EqualWeight",
        "S&P 500 Index Benchmark (^GSPC)": "SP500"
    }
    for name, column in strategies.items():
        returns = results_df[column].dropna()
        ann_return = (1 + returns.mean()) ** 252 - 1
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
        cum_return = (1 + returns).cumprod().iloc[-1] - 1

        summary_table[name] = {
            "Cumulative Return": f"{cum_return * 100:.2f}%",
            "Annualized Return": f"{ann_return * 100:.2f}%",
            "Annualized Volatility": f"{ann_vol * 100:.2f}%",
            "Sharpe Ratio": f"{sharpe:.2f}"
        }
    return pd.DataFrame(summary_table).T


def plot_results_vs_sp500(results_df):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    cum_robust = (1 + results_df["Strategy_Robust"]).cumprod() - 1
    cum_eq = (1 + results_df["Strategy_EqualWeight"]).cumprod() - 1
    cum_sp500 = (1 + results_df["SP500"]).cumprod() - 1

    ax1.plot(cum_robust.index, cum_robust * 100, label="Robust Paper Portfolio", color="#1f77b4", linewidth=2.5)
    ax1.plot(cum_eq.index, cum_eq * 100, label="Equal-Weighted 1/N Portfolio", color="#7f7f7f", linestyle="-.", linewidth=1.2)
    ax1.plot(cum_sp500.index, cum_sp500 * 100, label="S&P 500 Index Market (^GSPC)", color="#000000", linestyle="--", linewidth=2.0)

    ax1.set_style = 'whitegrid'
    ax1.set_xlabel("Historical Timeline", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Cumulative Performance Return (%)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.fill_between(results_df.index, results_df["Active_Assets_Robust"], step="pre", color="#2ca02c", alpha=0.04, label="Robust Sparsity Selection Profile")
    ax2.set_ylabel("Number of Active Assets Selected", color="#2ca02c", fontweight="bold")
    ax2.tick_params(axis='y', labelcolor="#2ca02c")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9)

    plt.title("Integrated Strategy Vector Performance vs S&P 500 Index Benchmark Universe", fontsize=13, fontweight="bold", pad=15)
    fig.tight_layout()
    plt.savefig("integrated_strategy_vs_sp500.png", dpi=300)
    plt.show()

# ==============================================================================
# 5. MASS DATA SCRAPER & SYSTEM PIPELINE CONTROLLER
# ==============================================================================

if __name__ == "__main__":
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
    print(f"Ingesting data target profiles for {len(raw_tickers)} equity tickers via Yahoo Finance...")

    # Download benchmark line
    df_bench_raw = yf.download("^GSPC", start="2016-01-01", end="2026-01-01", progress=False)
    sp500_series = df_bench_raw['Adj Close'] if 'Adj Close' in df_bench_raw.columns else df_bench_raw['Close']
    sp500_series = sp500_series.squeeze().ffill()

    # Batch downloading tickers in increments of 40 to protect connection limits
    chunk_size = 40
    valid_stock_series = {}
    
    for i in range(0, len(raw_tickers), chunk_size):
        chunk = raw_tickers[i:i + chunk_size]
        df_chunk_raw = yf.download(chunk, start="2016-01-01", end="2026-01-01", group_by="ticker", progress=False)
        
        for ticker in chunk:
            try:
                if ticker in df_chunk_raw.columns.levels[0]:
                    ticker_df = df_chunk_raw[ticker]
                    series = ticker_df['Adj Close'] if 'Adj Close' in ticker_df.columns else ticker_df['Close']
                    series = series.squeeze()
                    # Retain raw data context natively even if gaps are present across timestamps
                    valid_stock_series[ticker] = series
            except Exception:
                continue

    df_stocks = pd.DataFrame(valid_stock_series)

    # Align structural timestamp boundaries against index market lines
    common_idx = df_stocks.index.intersection(sp500_series.index)
    df_stocks = df_stocks.loc[common_idx]
    sp500_series = sp500_series.loc[common_idx]

    # Percentage change transformation map.
    # fillna(0.0) is applied to turn daily tracking gaps into non-movement return profiles.
    stock_returns = df_stocks.pct_change().fillna(0.0)
    sp500_returns = sp500_series.pct_change().dropna()

    print(f"\nData matrices synchronized at date boundary: {stock_returns.index[0].strftime('%Y-%m-%d')}")
    print(f"Total processed master portfolio components: {stock_returns.shape[1]} S&P 500 equities.")

    # Execution Settings: 252 observation lookback, rebalancing monthly (21 trading days)
    results = run_sp500_backtest(
        stock_returns, sp500_returns,
        lookback_window=252, rebalance_freq=21, lambda_tc=0.0010 
    )

    metrics_df = evaluate_metrics(results)
    print("\n=================== FINAL PERFORMANCE MATRIX VS S&P 500 ===================")
    print(metrics_df.to_string())
    print("===========================================================================")

    plot_results_vs_sp500(results)