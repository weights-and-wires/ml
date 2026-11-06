"""Why averaging models cuts noise, and where it stops.

Each model's prediction at one spot is a shared part plus a private part.
Averaging B models cancels the private parts and leaves the shared part, so the
variance of the average has a floor. You write three functions, in this order:

1. `bagging_variance`: the variance of an average of B models.
2. `correlated_estimators`: simulate models with a chosen variance and correlation.
3. `trees_needed`: how many models bring the variance close to the floor.
"""

import numpy as np


def bagging_variance(rho: float, sigma2: float, B: int) -> float:
    """Compute the variance of the average of B models.

    Every model has variance `sigma2`, and every pair of models has
    correlation `rho`. The shared part, `rho * sigma2`, is not divided by `B`.
    The private part, `(1 - rho) * sigma2`, is divided by `B`.

    Args:
        rho: Correlation between any two models, from 0 to 1.
        sigma2: Variance of a single model.
        B: Number of models averaged.

    Returns:
        The variance of the average, as a float.
    """
    raise NotImplementedError("bagging_variance")


def correlated_estimators(
    rho: float, sigma2: float, B: int, n_draws: int, rng: np.random.Generator
) -> np.ndarray:
    """Simulate B models whose predictions share a part and keep a private part.

    Entry `[i, b]` is `sqrt(rho * sigma2) * shared[i] + sqrt((1 - rho) * sigma2) * private[i, b]`,
    where `shared` is one standard normal draw per row, `private` is an
    independent standard normal draw per entry, and a row is one redraw of the
    training set. Then every column has variance `sigma2` and every pair of
    columns has correlation `rho`.

    Args:
        rho: Correlation between any two columns, from 0 to 1.
        sigma2: Variance of each column.
        B: Number of models, the number of columns.
        n_draws: Number of redraws, the number of rows.
        rng: A numpy random Generator to draw from.

    Returns:
        An array of shape (n_draws, B).
    """
    raise NotImplementedError("correlated_estimators")


def trees_needed(rho: float, sigma2: float, tol: float) -> int:
    """Find the smallest B whose variance is within a fraction `tol` of the floor.

    The floor is `rho * sigma2`. The variance with `B` models is within a
    fraction `tol` of it when `(1 - rho) * sigma2 / B <= tol * rho * sigma2`.

    Args:
        rho: Correlation between any two models, from 0 to 1.
        sigma2: Variance of a single model.
        tol: Allowed gap above the floor, as a fraction of the floor.

    Returns:
        The smallest integer B, and at least 1.

    Raises:
        ValueError: If `rho` is 0, because then the floor is 0 and no finite B
            gets within a fraction of it.
    """
    raise NotImplementedError("trees_needed")


if __name__ == "__main__":
    print(f"sigma2=4, rho=0.5: B=1 gives {bagging_variance(0.5, 4.0, 1):.2f}, B=2 gives {bagging_variance(0.5, 4.0, 2):.2f}")
    print(f"rho=0.3, sigma2=1: B=50 gives {bagging_variance(0.3, 1.0, 50):.4f}, B=5000 gives {bagging_variance(0.3, 1.0, 5000):.5f}")

    rng = np.random.default_rng(0)
    draws = correlated_estimators(0.3, 1.0, 50, 20000, rng)
    print(f"simulated variance of the average of 50 models: {draws.mean(axis=1).var():.4f}")

    print(f"models needed to get within 5% of the floor at rho=0.3: {trees_needed(0.3, 1.0, 0.05)}")
