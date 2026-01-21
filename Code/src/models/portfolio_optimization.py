import numpy as np
import cvxpy as cp

def mean_variance_optimization(
    mu: np.ndarray,
    cov: np.ndarray,
    risk_aversion: float = 1.0
) -> np.ndarray:
    """
    Solve mean-variance optimization
    """
    n = len(mu)
    w = cp.Variable(n)

    objective = cp.Maximize(mu @ w - risk_aversion * cp.quad_form(w, cov))
    constraints = [
        cp.sum(w) == 1,
        w >= 0
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    return w.value
