import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.covariance import LedoitWolf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ==============================================================================
# 1. COVARIANCE ESTIMATORS & ROBUST ADJUDICATIONS
# ==============================================================================

def nearest_positive_definite(matrix, eps=1e-4):
    """
    Finds the nearest symmetric positive definite (s.p.d) matrix.
    Eliminates numerical non-invertibility issues when N > T.
    """
    symmetric_mat = (matrix + matrix.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric_mat)
    eigenvalues = np.maximum(eigenvalues, eps)
    spd_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return (spd_matrix + spd_matrix.T) / 2


def robust_gerber_covariance_mad(returns, c=0.4):
    """
    Computes Robust Gerber Covariance using Median Absolute Deviation (MAD) thresholds
    as derived in Equation (5) of the paper[cite: 323, 519].
    Filters noise and isolates true underlying asset regimes[cite: 321, 455].
    """
    T, N = returns.shape
    returns_matrix = np.asarray(returns)
    
    # Compute MAD: med_i(|y_i - med_j(y_j)|) [cite: 528]
    med = np.median(returns_matrix, axis=0)
    mad = np.median(np.abs(returns_matrix - med), axis=0)
    thresholds = c * mad
    
    # Structural indicators for Upward (U) and Downward (D) threshold breaches [cite: 457, 463]
    U = (returns_matrix >= thresholds).astype(float)
    D = (returns_matrix <= -thresholds).astype(float)
    
    # Event count allocations [cite: 461, 468, 473]
    N_UU = U.T @ U
    N_DD = D.T @ D
    N_UD = U.T @ D
    N_DU = D.T @ U
    
    N_CONC = N_UU + N_DD
    N_DISC = N_UD + N_DU
    
    # Equation (5) Denominator: Ignores zones where asset pairs don't cross thresholds 
    denominator = T - ((1.0 - U) * (1.0 - D)).T @ ((1.0 - U) * (1.0 - D))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        G = np.where(denominator > 0, (N_CONC - N_DISC) / denominator, 0.0)
    np.fill_diagonal(G, 1.0)
    
    # Project correlation framework to covariance space using sample standard deviations [cite: 82, 477]
    sample_std = np.std(returns_matrix, axis=0, ddof=1)
    gerber_cov = np.diag(sample_std) @ G @ np.diag(sample_std)
    
    return nearest_positive_definite(gerber_cov)


def ledoit_wolf_covariance(returns):
    """
    Calculates Ledoit-Wolf Shrinkage Covariance, shrinking toward 
    a constant-correlation target structure to prevent T < N singularity[cite: 47, 53].
    """
    lw = LedoitWolf()
    lw.fit(returns)
    return lw.covariance_


# ==============================================================================
# 2. NESTED CLUSTERED OPTIMIZATION (NCO) COMPONENTS
# ==============================================================================

def de_noise_covariance(cov_matrix, T, N):
    """
    Filters out noise-injected random components via Marcenko-Pastur spectral clipping[cite: 121, 135].
    Averages eigenvalues captured below the theoretical cutoff lambda_+[cite: 121, 561].
    """
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sigma_sq = np.mean(eigenvalues)  # Estimator of base data variance
    lambda_plus = sigma_sq * (1.0 + np.sqrt(N / T)) ** 2  # Max noise boundary [cite: 121, 561]
    
    is_noise = eigenvalues <= lambda_plus
    if np.any(is_noise):
        avg_noise_eigenvalue = np.mean(eigenvalues[is_noise])
        eigenvalues[is_noise] = avg_noise_eigenvalue  # Spectral flattening [cite: 137]
        
    de_noised_cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return nearest_positive_definite(de_noised_cov)


def nested_clustered_optimization(returns, base_cov):
    """
    Executes Lopez de Prado's Algorithm 1: Nested Clustered Optimization[cite: 114, 150].
    Isolates signal-induced collinearity into independent group clusters[cite: 118, 132].
    """
    T, N = returns.shape
    cov_cleaned = de_noise_covariance(base_cov, T, N)
    
    # Deduce cleaned correlation matrix [cite: 592]
    std_devs = np.sqrt(np.diagonal(cov_cleaned))
    corr_matrix = cov_cleaned / np.outer(std_devs, std_devs)
    corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
    
    # Identify the optimal cluster number (K) using Silhouette scores [cite: 139]
    best_k, best_score = 2, -1.0
    for k_test in range(2, min(11, N)):
        km = KMeans(n_clusters=k_test, random_state=42, n_init=10)
        labels = km.fit_predict(corr_matrix)
        score = silhouette_score(corr_matrix, labels)
        if score > best_score:
            best_score, best_k = score, k_test
            
    clf = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = clf.fit_predict(corr_matrix)
    
    # Intra-cluster Optimization Stage (Build cluster 'funds') [cite: 141, 142]
    w_intra = np.zeros((N, best_k))
    for k in range(best_k):
        cluster_idx = np.where(cluster_labels == k)[0]
        if len(cluster_idx) == 0: continue
        
        sub_cov = cov_cleaned[np.ix_(cluster_idx, cluster_idx)]
        w_sub = cp.Variable(len(cluster_idx))
        prob = cp.Problem(cp.Minimize(cp.quad_form(w_sub, sub_cov)), [cp.sum(w_sub) == 1.0, w_sub >= 0.0])
        prob.solve(solver=cp.ECOS)
        w_intra[cluster_idx, k] = w_sub.value
        
    # Inter-cluster Optimization Stage [cite: 144]
    v_reduced = w_intra.T @ cov_cleaned @ w_intra  # Lower dimension projection [cite: 142, 144]
    w_inter = cp.Variable(best_k)
    prob_inter = cp.Problem(cp.Minimize(cp.quad_form(w_inter, v_reduced)), [cp.sum(w_inter) == 1.0, w_inter >= 0.0])
    prob_inter.solve(solver=cp.ECOS)
    
    # Return mapping matrix multiplication [cite: 145]
    return w_intra @ w_inter.value


# ==============================================================================
# 3. CONVEX OPTIMIZER WITH TRANSACTION COSTS AND DUAL CVAR CONSTRAINTS
# ==============================================================================

def optimize_portfolio_cvar(historical_scenarios, cov_matrix, w_initial, lambda_tc=0.0050):
    """
    Solves Minimum Variance Optimization augmented with an L1-norm transaction cost 
    penalty and dual alpha-tail risk CVaR constraints using linear programming[cite: 104, 112, 157].
    
    Constraints applied:
    - 95% CVaR threshold restricted below 5.0% expected tail loss [cite: 157]
    - 99% CVaR threshold restricted below 8.0% expected tail loss [cite: 157]
    """
    S, N = historical_scenarios.shape
    scenarios_matrix = np.asarray(historical_scenarios)
    
    # Decoupled optimization variables
    w = cp.Variable(N)
    zeta_95 = cp.Variable()  # Auxiliary VaR boundary for alpha = 0.95 [cite: 159]
    zeta_99 = cp.Variable()  # Auxiliary VaR boundary for alpha = 0.99 [cite: 159]
    z_95 = cp.Variable(S)    # Scenario loss exceedances vector [cite: 159]
    z_99 = cp.Variable(S)    # Scenario loss exceedances vector [cite: 159]
    
    # Objective function components: w^T * V * w + lambda * ||w - w_0||_1 [cite: 112]
    portfolio_variance = cp.quad_form(w, cov_matrix)
    transaction_cost_penalty = lambda_tc * cp.norm(w - w_initial, 1)
    objective = cp.Minimize(portfolio_variance + transaction_cost_penalty)
    
    # Linearized constraint system tracking historical downside scenarios [cite: 159, 163]
    constraints = [
        cp.sum(w) == 1.0,
        w >= 0.0,
        
        # 95% CVaR Constraint Implementation [cite: 157, 159]
        z_95 >= 0.0,
        (-scenarios_matrix @ w) - zeta_95 <= z_95,
        zeta_95 + (1.0 / (1.0 - 0.95)) * cp.mean(z_95) <= 0.05,
        
        # 99% CVaR Constraint Implementation [cite: 157, 159]
        z_99 >= 0.0,
        (-scenarios_matrix @ w) - zeta_99 <= z_99,
        zeta_99 + (1.0 / (1.0 - 0.99)) * cp.mean(z_99) <= 0.08
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        # Fallback to pure minimum variance if dual tail barriers cannot converge on strict limits
        fallback_constraints = [cp.sum(w) == 1.0, w >= 0.0]
        prob_fallback = cp.Problem(objective, fallback_constraints)
        prob_fallback.solve(solver=cp.ECOS)
        
    return w.value


# ==============================================================================
# 4. BACKTEST EXECUTION SIMULATOR (S&P 500 SPECIFICATIONS)
# ==============================================================================

if __name__ == "__main__":
    # Create synthetic S&P 500 returns to test code paths (T=250 weeks, N=500 assets)
    # This directly triggers the singular matrix situation (T < N)[cite: 46, 47].
    np.random.seed(42)
    T_total_weeks = 250
    N_assets_sp500 = 500
    
    # Generate random matrix with heavy tail returns
    synthetic_sp500_returns = np.random.normal(0.0005, 0.025, size=(T_total_weeks, N_assets_sp500))
    
    # Generate placeholder capitalization weights for the baseline portfolio [cite: 176]
    caps = np.random.uniform(10, 1000, size=N_assets_sp500)
    market_cap_benchmark_weights = caps / np.sum(caps)
    
    # Set parameters to match the paper's framework [cite: 157, 166]
    lookback_window = 200 # Window size for covariance matrix estimation [cite: 166]
    initial_w0 = np.ones(N_assets_sp500) / N_assets_sp500  # Equal weight starting profile [cite: 113]
    
    print(f"Executing robust matrix validation on S&P 500 assets (T < N Environment)...")
    print(f"Data dimensions: {lookback_window} historical observations x {N_assets_sp500} assets.\n")
    
    # Slice a single out-of-sample execution window [cite: 166]
    window_data = synthetic_sp500_returns[0:lookback_window, :]
    
    # --- Framework 1: Ledoit-Wolf Shrinkage Matrix ---
    print("-> Calculating Ledoit-Wolf Shrinkage Matrix...")
    lw_cov = ledoit_wolf_covariance(window_data)
    print(f"   Matrix Singular? {np.any(np.linalg.eigvals(lw_cov) <= 0)}")
    
    # --- Framework 2: Robust Gerber (MAD Threshold) ---
    print("-> Calculating Robust Gerber Matrix via MAD scaling (c=0.4)...")
    gerber_mad_cov = robust_gerber_covariance_mad(window_data, c=0.4)
    print(f"   Matrix Symmetric Positive Definite? {np.all(np.linalg.eigvals(gerber_mad_cov) > 0)}")
    
    # --- Framework 3: Nested Clustered Optimization Optimization Vector ---
    print("-> Routing via Nested Clustered Optimization (NCO Pipeline)...")
    nco_weights = nested_clustered_optimization(window_data, gerber_mad_cov)
    print(f"   NCO Target weights derived. (Sum of allocations: {np.sum(nco_weights):.2f})")
    
    # --- Framework 4: Integrated Optimization with Dual CVaR Barriers & Rebalancing Costs ---
    print("-> Deploying ECOS Convex Solver with dual CVaR boundaries and L1 transaction costs...")
    optimal_allocation = optimize_portfolio_cvar(
        historical_scenarios=window_data, 
        cov_matrix=gerber_mad_cov, 
        w_initial=initial_w0,
        lambda_tc=0.0050 # 50 bps turnover penalty friction [cite: 105, 113]
    )
    
    # Output structural configuration statistics
    active_positions = np.sum(optimal_allocation > 1e-4)
    max_single_weight = np.max(optimal_allocation)
    
    print("\n" + "="*70)
    print("PORTFOLIO OPTIMIZATION RUN SUMMARY (S&P 500 TARGET WINDOW)")
    print("="*70)
    print(f"Allocated Active Positions : {active_positions} / {N_assets_sp500} assets")
    print(f"Maximum Asset Concentration: {max_single_weight * 100:.2f}%")
    print(f"Implied Transaction Cost   : {0.0050 * np.sum(np.abs(optimal_allocation - initial_w0)) * 10000:.2f} bps")
    print("="*70)