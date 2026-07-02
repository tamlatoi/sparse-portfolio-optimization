import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Import the core optimization engine cleanly from main.py
from main import solve_large_sparse_portfolio

def run_sp500_backtest(df_returns, sp500_returns, lookback_window=252, rebalance_freq=21, lambda_val=0.4, gamma_val=0.05, tau_val=0.02):
    """
    Simulates portfolio strategies. 
    Dimensions of w and w_drift are strictly bound to df_returns.shape[1]
    """
    n_timesteps, n_assets = df_returns.shape
    print(f"Initializing Backtest Engine. Timesteps: {n_timesteps}, Active Assets: {n_assets}")

    # Strategy 1 Data Structures (Turnover Constrained)
    strat_turnover_returns = []
    active_turnover_count = []
    weights_turnover = None  # Will be initialized as np.zeros(n_assets) upon first rebalance

    # Strategy 2 Data Structures (Baseline No-Turnover)
    strat_baseline_returns = []
    weights_baseline = None

    backtest_dates = []

    for t in range(lookback_window, n_timesteps):
        current_date = df_returns.index[t]
        daily_returns = df_returns.iloc[t].values

        # --- 1. DAILY WEIGHT DRIFT TRACKING ---
        if weights_turnover is not None:
            drifted_turnover = weights_turnover * (1 + daily_returns)
            # Handle zero-denominator edge case gracefully
            if np.sum(drifted_turnover) > 0:
                weights_turnover = drifted_turnover / np.sum(drifted_turnover)
            else:
                weights_turnover = np.ones(n_assets) / n_assets

        if weights_baseline is not None:
            drifted_baseline = weights_baseline * (1 + daily_returns)
            if np.sum(drifted_baseline) > 0:
                weights_baseline = drifted_baseline / np.sum(drifted_baseline)
            else:
                weights_baseline = np.ones(n_assets) / n_assets

        # --- 2. REBALANCING WINDOW ---
        if (t - lookback_window) % rebalance_freq == 0:
            window_data = df_returns.iloc[t - lookback_window:t]
            
            # Formulate parameters strictly bounded to the shape of the clean window data
            mu_window = window_data.mean().values * 252
            Q_window = window_data.cov().values * 252

            # A. Strategy 1 Optimization (Turnover penalty applied)
            w_drift_target = weights_turnover if weights_turnover is not None else np.zeros(n_assets)
            
            # Pass vectors that match the dimension of Q_window perfectly
            new_w_turnover = solve_large_sparse_portfolio(
                mu_window, Q_window, lambda_val, gamma_val, 
                w_drift=w_drift_target, tau=tau_val
            )

            # B. Strategy 2 Optimization (No turnover penalty)
            new_w_baseline = solve_large_sparse_portfolio(
                mu_window, Q_window, lambda_val, gamma_val, 
                w_drift=None, tau=0.0
            )

            # Clean and store allocations if solver is successful
            if new_w_turnover is not None:
                new_w_turnover = np.array(new_w_turnover).flatten()
                new_w_turnover[np.abs(new_w_turnover) < 1e-3] = 0.0
                if np.sum(new_w_turnover) > 0:
                    weights_turnover = new_w_turnover / np.sum(new_w_turnover)
            
            if new_w_baseline is not None:
                new_w_baseline = np.array(new_w_baseline).flatten()
                new_w_baseline[np.abs(new_w_baseline) < 1e-3] = 0.0
                if np.sum(new_w_baseline) > 0:
                    weights_baseline = new_w_baseline / np.sum(new_w_baseline)

        # --- 3. RECORD DAILY RETURNS ---
        if weights_turnover is not None and weights_baseline is not None:
            ret_turnover = np.dot(weights_turnover, daily_returns)
            ret_baseline = np.dot(weights_baseline, daily_returns)

            strat_turnover_returns.append(ret_turnover)
            strat_baseline_returns.append(ret_baseline)
            backtest_dates.append(current_date)
            active_turnover_count.append(np.sum(weights_turnover > 0.001))

    # Compile results aligned with your timeline
    results_df = pd.DataFrame({
        "Strategy_Turnover": strat_turnover_returns,
        "Strategy_Baseline": strat_baseline_returns,
        "Active_Assets_Turnover": active_turnover_count
    }, index=backtest_dates)

    results_df["SP500"] = sp500_returns.loc[results_df.index]
    return results_df

def evaluate_metrics(results_df):
    summary_table = {}
    strategies = {
        "Turnover Constrained Strategy": "Strategy_Turnover",
        "No Turnover Penalty Baseline": "Strategy_Baseline",
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
    cum_turnover = (1 + results_df["Strategy_Turnover"]).cumprod() - 1
    cum_baseline = (1 + results_df["Strategy_Baseline"]).cumprod() - 1
    cum_sp500 = (1 + results_df["SP500"]).cumprod() - 1

    ax1.plot(cum_turnover.index, cum_turnover * 100, label="Turnover Constrained Portfolio", color="#1f77b4", linewidth=2.5)
    ax1.plot(cum_baseline.index, cum_baseline * 100, label="No Turnover Penalty Baseline", color="#d62728", linestyle="-.", linewidth=1.5)
    ax1.plot(cum_sp500.index, cum_sp500 * 100, label="S&P 500 Index Market (^GSPC)", color="#000000", linestyle="--", linewidth=2.0)

    ax1.set_xlabel("Date", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Cumulative Return (%)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.fill_between(results_df.index, results_df["Active_Assets_Turnover"], step="pre", color="#2ca02c", alpha=0.05, label="Selected Assets Count")
    ax2.set_ylabel("Number of Assets Selected", color="#2ca02c", fontweight="bold")
    ax2.tick_params(axis='y', labelcolor="#2ca02c")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9)

    plt.title("Strategy Execution Vector vs. S&P 500 Index Market Benchmark", fontsize=13, fontweight="bold", pad=15)
    fig.tight_layout()
    plt.savefig("strategy_vs_sp500.png", dpi=300)
    plt.show()

if __name__ == "__main__":
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
    
    # Force clean tokenization and remove duplicates
    raw_tickers = [t.strip() for t in sp500_string.split() if t.strip()]
    raw_tickers = list(set(raw_tickers))

    print("Fetching data profiles from Yahoo Finance...")

    # 1. Download benchmark index
    df_bench_raw = yf.download("^GSPC", start="2015-01-01", end="2026-01-01", progress=False)
    sp500_series = df_bench_raw['Adj Close'] if 'Adj Close' in df_bench_raw.columns else df_bench_raw['Close']
    sp500_series = sp500_series.squeeze().ffill()

    # 2. Download stock universe using the robust group_by="ticker" layout
    chunk_size = 40
    valid_stock_series = {}
    
    for i in range(0, len(raw_tickers), chunk_size):
        chunk = raw_tickers[i:i + chunk_size]
        df_chunk_raw = yf.download(chunk, start="2015-01-01", end="2026-01-01", group_by="ticker", progress=False)
        
        for ticker in chunk:
            try:
                if ticker in df_chunk_raw.columns.levels[0]:
                    ticker_df = df_chunk_raw[ticker]
                    series = ticker_df['Adj Close'] if 'Adj Close' in ticker_df.columns else ticker_df['Close']
                    series = series.squeeze()
                    if not series.dropna().empty:
                        valid_stock_series[ticker] = series
            except Exception:
                continue

    # Reconstruct the structural master frame
    df_stocks = pd.DataFrame(valid_stock_series)

    # 3. Clean and filter based on a 90% completion rule over the historical timeline
    min_data_rows = int(len(df_stocks) * 0.90)
    df_stocks_clean = df_stocks.dropna(axis=1, thresh=min_data_rows)
    cleaned_stocks = df_stocks_clean.ffill().bfill().dropna()

    # 4. Sync timeline intersections perfectly
    common_idx = cleaned_stocks.index.intersection(sp500_series.index)
    cleaned_stocks = cleaned_stocks.loc[common_idx]
    sp500_series = sp500_series.loc[common_idx]

    # 5. Formulate final percent returns profiles
    stock_returns = cleaned_stocks.pct_change().dropna()
    sp500_returns = sp500_series.pct_change().dropna()

    print(f"\nData matrix stabilized at: {stock_returns.index[0].strftime('%Y-%m-%d')}")
    print(f"Simulating across {stock_returns.shape[1]} clean, high-fidelity long-history S&P 500 equities...")

    # Strategy Execution Parameters
    lambda_val = 0.1  
    gamma_val = 0.03  
    tau_val = 0.05    

    print("Initiating portfolio optimization loop...\n")
    results = run_sp500_backtest(
        stock_returns, sp500_returns,
        lookback_window=504, rebalance_freq=21,
        lambda_val=lambda_val, gamma_val=gamma_val, tau_val=tau_val
    )

    metrics_df = evaluate_metrics(results)
    print("\n=================== FINAL PERFORMANCE MATRIX VS S&P 500 ===================")
    print(metrics_df.to_string())
    print("===========================================================================")

    plot_results_vs_sp500(results)