"""Reference solution for the averaging-models exercise."""

import math

import numpy as np


def bagging_variance(rho: float, sigma2: float, B: int) -> float:
    return float(rho * sigma2 + (1.0 - rho) * sigma2 / B)


def correlated_estimators(
    rho: float, sigma2: float, B: int, n_draws: int, rng: np.random.Generator
) -> np.ndarray:
    shared = rng.normal(size=(n_draws, 1))
    private = rng.normal(size=(n_draws, B))
    return np.sqrt(rho * sigma2) * shared + np.sqrt((1.0 - rho) * sigma2) * private


def trees_needed(rho: float, sigma2: float, tol: float) -> int:
    if rho == 0:
        raise ValueError("rho is 0, so the floor is 0 and no finite B gets within a fraction of it")
    return max(1, math.ceil((1.0 - rho) / (tol * rho)))


if __name__ == "__main__":
    print(f"sigma2=4, rho=0.5: B=1 gives {bagging_variance(0.5, 4.0, 1):.2f}, B=2 gives {bagging_variance(0.5, 4.0, 2):.2f}")
    print(f"rho=0.3, sigma2=1: B=50 gives {bagging_variance(0.3, 1.0, 50):.4f}, B=5000 gives {bagging_variance(0.3, 1.0, 5000):.5f}")

    rng = np.random.default_rng(0)
    draws = correlated_estimators(0.3, 1.0, 50, 20000, rng)
    print(f"simulated variance of the average of 50 models: {draws.mean(axis=1).var():.4f}")

    print(f"models needed to get within 5% of the floor at rho=0.3: {trees_needed(0.3, 1.0, 0.05)}")
