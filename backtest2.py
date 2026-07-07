import os
# Explicitly constrain thread allocation to prevent CPU core thrashing during 400+ ticker simulations
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import cvxpy as cp

# ==============================================================================
# 1. ROBUST COVARIANCE ESTIMATORS & MATRIX CLEANING
# ==============================================================================

def nearest_positive_definite(matrix, eps=1e-4):
    """Finds the nearest symmetric positive definite matrix."""
    symmetric_mat = (matrix + matrix.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric_mat)
    eigenvalues = np.maximum(eigenvalues, eps)
    spd_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return (spd_matrix + spd_matrix.T) / 2


def robust_gerber_covariance_mad(returns_matrix, c=0.4):
    """Computes Gerber Covariance Matrix using Median Absolute Deviation thresholds."""
    T, N = returns_matrix.shape
    
    med = np.median(returns_matrix, axis=0)
    mad = np.median(np.abs(returns_matrix - med), axis=0)
    thresholds = c * mad
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
    return gerber_cov


def de_noise_covariance(cov_matrix, T, N):
    """Cleans noise from the covariance matrix via Marchenko-Pastur properties."""
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sigma_sq = np.mean(eigenvalues)
    lambda_plus = sigma_sq * (1.0 + np.sqrt(N / T)) ** 2
    
    is_noise = eigenvalues <= lambda_plus
    if np.any(is_noise):
        eigenvalues[is_noise] = np.mean(eigenvalues[is_noise])
        
    de_noised_cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return nearest_positive_definite(de_noised_cov)

# ==============================================================================
# 2. BALANCED SCALED CVAR OPTIMIZER
# ==============================================================================

def optimize_portfolio_cvar(scenarios_annual, cov_annual, mu_annual, w_initial=None, tau=0.01):
    """Performs portfolio optimization under parametric constraints & tail-risk (CVaR) boundaries."""
    S, N = scenarios_annual.shape
    
    eq_weights = np.ones(N) / N
    equal_weight_scenarios = -scenarios_annual @ eq_weights
    
    # Adaptive upper bound tail constraint configurations
    max_allowable_95_cvar = max(0.15, np.percentile(equal_weight_scenarios, 95) * 0.98)
    max_allowable_99_cvar = max(0.22, np.percentile(equal_weight_scenarios, 99) * 0.98)
    
    w = cp.Variable(N)
    zeta_95 = cp.Variable()
    zeta_99 = cp.Variable()
    z_95 = cp.Variable(S)
    z_99 = cp.Variable(S)
    
    portfolio_risk = 0.5 * cp.quad_form(w, cov_annual)
    scaled_return = mu_annual @ w
    
    # Structural target formulation branching
    if w_initial is not None:
        # Strategy 1: Active turnover minimization platform
        objective = cp.Minimize(portfolio_risk - scaled_return + (tau * cp.norm(w - w_initial, 1)))
    else:
        # Strategy 2: Pure Classic Markowitz (Risk-Return Optimization)
        objective = cp.Minimize(portfolio_risk - scaled_return)
        
    constraints = [
        cp.sum(w) == 1.0,
        w >= 0.0, 
        
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
        fallback_objective = cp.Minimize(portfolio_risk)
        prob_fallback = cp.Problem(fallback_objective, [cp.sum(w) == 1.0, w >= 0.0, w <= 0.08])
        prob_fallback.solve(solver=cp.CLARABEL)
        
    return w.value

# ==============================================================================
# 3. SIMULATION BACKTEST ENGINE
# ==============================================================================

def run_sp500_backtest(df_returns, sp500_returns, lookback_window=252, rebalance_freq=21, lamb=1.0, tau=0.01):
    n_timesteps, n_assets = df_returns.shape

    strat_robust_returns, active_robust_count = [], []
    w_robust = np.ones(n_assets) / n_assets

    strat_markowitz_returns, active_markowitz_count = [], []
    w_markowitz = np.ones(n_assets) / n_assets

    strat_eq_returns, backtest_dates = [], []

    for t in range(lookback_window, n_timesteps):
        current_date = df_returns.index[t]
        daily_returns = df_returns.iloc[t].values

        # --- 1. END-OF-DAY DRIFT ACCOUNTING ---
        # Account for asset price movements from yesterday to today before any new rebalancing executes
        if t > lookback_window:
            prev_returns = df_returns.iloc[t-1].values
            
            w_robust = w_robust * (1.0 + prev_returns)
            w_robust = w_robust / np.sum(w_robust) if np.sum(w_robust) > 1e-5 else np.ones(n_assets) / n_assets
            
            w_markowitz = w_markowitz * (1.0 + prev_returns)
            w_markowitz = w_markowitz / np.sum(w_markowitz) if np.sum(w_markowitz) > 1e-5 else np.ones(n_assets) / n_assets

        # --- 2. PERIODIC REBALANCING (EXECUTED AT START OF DAY t BASED ON INFORMATION UP TO t-1) ---
        if (t - lookback_window) % rebalance_freq == 0:
            # Exclude day t to guarantee no look-ahead data leakage
            window_data = df_returns.iloc[t - lookback_window:t]
            active_mask = (window_data.std() > 1e-7) & (~window_data.iloc[-1].isna())
            active_indices = np.where(active_mask)[0]
            
            if len(active_indices) > 20:  
                active_scenarios = window_data.iloc[:, active_indices].values
                
                # Execution Matrix Pipeline (Single-pass NPD conversion)
                raw_gerber = robust_gerber_covariance_mad(active_scenarios, c=0.4)
                gerber_cov = de_noise_covariance(raw_gerber, active_scenarios.shape[0], active_scenarios.shape[1])
                
                # Annualization scaling step (Linear scaling for CVaR returns matrices)
                cov_annual = cp.psd_wrap(gerber_cov + np.eye(len(active_indices)) * 1e-6) * 252.0
                mu_annual = active_scenarios.mean(axis=0) * 252.0 * lamb
                scenarios_annual = active_scenarios * 252.0
                
                if t == lookback_window:
                    w_init_active_r = np.zeros(len(active_indices))
                else:
                    w_init_active_r = w_robust[active_indices]
                    w_init_active_r = w_init_active_r / np.sum(w_init_active_r) if np.sum(w_init_active_r) > 1e-5 else np.ones(len(active_indices)) / len(active_indices)
                
                # A. Strategy 1: Optimization Execution (Turnover Constraint Platform)
                try:
                    opt_w_robust = optimize_portfolio_cvar(scenarios_annual, cov_annual, mu_annual, w_initial=w_init_active_r, tau=tau)
                    if opt_w_robust is not None:
                        opt_w_robust = np.array(opt_w_robust).flatten()
                        opt_w_robust[np.abs(opt_w_robust) < 1e-3] = 0.0
                        w_new_robust = np.zeros(n_assets)
                        w_new_robust[active_indices] = opt_w_robust
                        if np.sum(w_new_robust) > 1e-4:
                            w_robust = w_new_robust / np.sum(w_new_robust)
                except Exception:
                    pass

                # B. Strategy 2: Optimization Execution (Pure Classic Markowitz)
                try:
                    opt_w_markowitz = optimize_portfolio_cvar(scenarios_annual, cov_annual, mu_annual, w_initial=None)
                    if opt_w_markowitz is not None:
                        opt_w_markowitz = np.array(opt_w_markowitz).flatten()
                        opt_w_markowitz[np.abs(opt_w_markowitz) < 1e-3] = 0.0
                        w_new_markowitz = np.zeros(n_assets)
                        w_new_markowitz[active_indices] = opt_w_markowitz
                        if np.sum(w_new_markowitz) > 1e-4:
                            w_markowitz = w_new_markowitz / np.sum(w_new_markowitz)
                except Exception:
                    pass

        # --- 3. HARVEST SYSTEM RETURNS ---
        strat_robust_returns.append(np.dot(w_robust, daily_returns))
        strat_markowitz_returns.append(np.dot(w_markowitz, daily_returns))
        
        today_active_mask = ~df_returns.iloc[t].isna()
        w_eq = np.zeros(n_assets)
        w_eq[today_active_mask] = 1.0 / np.sum(today_active_mask)
        strat_eq_returns.append(np.dot(w_eq, daily_returns))
        
        backtest_dates.append(current_date)
        active_robust_count.append(np.sum(w_robust > 0.001))
        active_markowitz_count.append(np.sum(w_markowitz > 0.001))

    results_df = pd.DataFrame({
        "Strategy_Robust": strat_robust_returns,
        "Strategy_Markowitz": strat_markowitz_returns,
        "Strategy_EqualWeight": strat_eq_returns,
        "Active_Assets_Robust": active_robust_count,
        "Active_Assets_Markowitz": active_markowitz_count
    }, index=backtest_dates)

    results_df["SP500"] = sp500_returns.loc[results_df.index]
    return results_df

# ==============================================================================
# 4. EXECUTION DRIVER & PERFORMANCE EVALUATION
# ==============================================================================

if __name__ == "__main__":
    CHOSEN_LAMBDA, CHOSEN_TAU = 0.1, 0.05

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
    FCX MET MET NSC CEG GWW RMD NXPI TFC ORLY
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
    BWA CE FMC COIN DDOG JNPR TTD ANSS XYZ HES
    PSKY IBKR WBA APP EME HOOD SOLS Q EMN FISV
    """
    raw_tickers = list(set([t.strip() for t in sp500_string.split() if t.strip()]))
    
    print(f"Ingesting data target profiles for {len(raw_tickers)} equity tickers...")
    df_bench_raw = yf.download("^GSPC", start="2018-01-01", end="2026-01-01", auto_adjust=True, progress=False, session=scraper_session)
    sp500_series = df_bench_raw['Close'].squeeze().ffill()

    df_chunk_raw = yf.download(raw_tickers, start="2018-01-01", end="2026-01-01", auto_adjust=True, progress=False, session=scraper_session)
    df_stocks = df_chunk_raw['Close']

    common_idx = df_stocks.index.intersection(sp500_series.index)
    df_stocks, sp500_series = df_stocks.loc[common_idx], sp500_series.loc[common_idx]

    stock_returns = df_stocks.ffill().pct_change(fill_method=None).fillna(0.0)
    sp500_returns = sp500_series.pct_change().dropna()

    print(f"\nRunning matrix simulations using Lambda={CHOSEN_LAMBDA}, Tau={CHOSEN_TAU}...")
    res = run_sp500_backtest(stock_returns, sp500_returns, lamb=CHOSEN_LAMBDA, tau=CHOSEN_TAU)

    # Metrics Engine
    metrics = {}
    for column in ["Strategy_Robust", "Strategy_Markowitz", "Strategy_EqualWeight", "SP500"]:
        rets = res[column].values
        cum_rets = (1 + rets).cumprod()
        running_max = np.maximum.accumulate(cum_rets)
        running_max = np.where(running_max == 0, 1.0, running_max)
        
        ann_return = (cum_rets[-1]) ** (252.0 / len(rets)) - 1
        ann_vol = np.std(rets, ddof=1) * np.sqrt(252)
        
        metrics[column] = {
            "Total Return (%)": (cum_rets[-1] - 1) * 100,
            "Annualized Return (%)": ann_return * 100,
            "Annualized Volatility (%)": ann_vol * 100,
            "Sharpe Ratio": ann_return / ann_vol if ann_vol > 0 else 0,
            "Max Drawdown (%)": np.min((cum_rets - running_max) / running_max) * 100
        }
        
    print("\n" + "="*75 + "\n" + "                 COMPREHENSIVE BACKTEST PERFORMANCE METRICS\n" + "="*75)
    print(pd.DataFrame(metrics).round(2).to_string())
    print("="*75 + "\n")

    # Visualization Generating Sequence
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(res.index, ((1 + res["Strategy_Robust"]).cumprod() - 1) * 100, label="Turnover Constrained Matrix", color="#1f77b4", linewidth=2.5)
    ax1.plot(res.index, ((1 + res["Strategy_Markowitz"]).cumprod() - 1) * 100, label="Classic Markowitz Baseline", color="#d62728", linestyle=":", linewidth=2.2)
    ax1.plot(res.index, ((1 + res["Strategy_EqualWeight"]).cumprod() - 1) * 100, label="Equal-Weighted 1/N Portfolio", color="grey", linestyle="-.", alpha=0.7)
    ax1.plot(res.index, ((1 + res["SP500"]).cumprod() - 1) * 100, label="S&P 500 Index Market", color="black", linestyle="--", linewidth=1.5)
    
    ax1.set_xlabel("Historical Timeline", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Cumulative Performance Return (%)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(res.index, res["Active_Assets_Robust"], color="#1f77b4", alpha=0.15, linestyle="-")
    ax2.plot(res.index, res["Active_Assets_Markowitz"], color="#d62728", alpha=0.15, linestyle="--")
    ax2.set_ylabel("Number of Active Assets Selected", color="darkgreen", fontsize=11, fontweight="bold")
    ax2.tick_params(axis='y', labelcolor="darkgreen")

    plt.title("Fully Cleaned Mean-Variance & CVaR Strategy Backtest Engine", fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.show()