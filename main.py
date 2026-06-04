import cvxpy as cp
import numpy as np


def solve_initial_sparse_portfolio(mu, Q, lambda_param, gamma, long_only=False):
    """Solves the initial sparse portfolio optimization problem from the image.

    Parameters:
    -----------
    mu : ndarray
        Expected return vector (shape: n,)
    Q : ndarray
        Covariance matrix of asset returns (shape: n x n)
    lambda_param : float
        Risk-return tradeoff parameter (lambda > 0)
    gamma : float
        Sparsity regularization parameter (gamma > 0)
    long_only : bool, optional
        If True, adds w >= 0 constraint. Defaults to False (allows shorting).
    """
    n = len(mu)

    # Define the optimization variable
    w = cp.Variable(n)

    # Formulate the objective function exactly as shown in the image:
    # min 0.5 * w^T * Q * w - lambda * mu^T * w + gamma * ||w||_1
    portfolio_risk = 0.5 * cp.quad_form(w, Q)
    expected_return = lambda_param * (mu @ w)
    sparsity_penalty = gamma * cp.norm(w, 1)

    objective = cp.Minimize(portfolio_risk - expected_return + sparsity_penalty)

    # Constraints: sum(w_i) = 1
    constraints = [cp.sum(w) == 1]

    # Add long-only constraint if desired (forces weights to be >= 0)
    if long_only:
        constraints.append(w >= 0)

    # Solve the convex optimization problem
    prob = cp.Problem(objective, constraints)
    # Try running with default settings if ECOS is missing
    try:
        prob.solve()  # Let CVXPY automatically choose an available solver (like OSQP or QDLDL)
    except Exception:
        prob.solve(solver=cp.OSQP)  # Backup solver that handles quadratic forms incredibly well

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Optimization failed with status: {prob.status}")

    return w.value


# ==========================================
# Example Execution
# ==========================================
if __name__ == "__main__":
    # Example data for 4 assets
    mu_example = np.array([0.08, 0.12, 0.10, 0.05])

    # 4x4 positive semi-definite covariance matrix
    Q_example = np.array(
        [
            [0.04, 0.01, 0.02, 0.00],
            [0.01, 0.06, 0.01, 0.01],
            [0.02, 0.01, 0.05, 0.02],
            [0.00, 0.01, 0.02, 0.03],
        ]
    )

    # Parameters
    lambda_val = 0.5  # Risk-return tradeoff
    gamma_val = 0.05  # Sparsity penalty

    # Run optimization (allowing short positions as per the image's basic constraints)
    weights = solve_initial_sparse_portfolio(
        mu_example, Q_example, lambda_val, gamma_val, long_only=False
    )

    print("--- Optimized Portfolio Weights ---")
    for i, w_i in enumerate(weights):
        # Round near-zero values for clean output due to L1 regularization
        print(f"Asset {i+1}: {0.0 if np.isclose(w_i, 0, atol=1e-4) else w_i:.2%}")