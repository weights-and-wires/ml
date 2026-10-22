"""Reference solution for the power-law exercise."""

import numpy as np


def power_law_eval(x, k: float, c: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return c * x**k


def fit_power_law(x, y) -> tuple:
    u = np.log(np.asarray(x, dtype=float))
    v = np.log(np.asarray(y, dtype=float))
    uc = u - u.mean()
    vc = v - v.mean()
    k = float(np.sum(uc * vc) / np.sum(uc**2))
    intercept = v.mean() - k * u.mean()
    return k, float(np.exp(intercept))


def extrapolate(x_train, y_train, x_new) -> np.ndarray:
    k, c = fit_power_law(x_train, y_train)
    return power_law_eval(x_new, k, c)


if __name__ == "__main__":
    # Five models, 10 million to 1 billion parameters, and the loss each one
    # sits above its floor.
    sizes = np.array([1e7, 3e7, 1e8, 3e8, 1e9])
    excess = 400.0 / sizes**0.34

    k, c = fit_power_law(sizes, excess)
    print(f"fitted exponent k = {k:.4f}, coefficient c = {c:.2f}")
    print(f"each doubling multiplies the loss by {2.0**k:.4f}")

    guess = extrapolate(sizes, excess, [1e10])
    print(f"predicted excess loss at 10 billion parameters: {guess[0]:.4f}")

    # The same fit on noisy measurements.
    rng = np.random.default_rng(3)
    noisy_sizes = np.geomspace(1e7, 1e9, 8)
    noisy = 400.0 / noisy_sizes**0.34 * np.exp(rng.normal(0.0, 0.03, 8))
    k_n, _ = fit_power_law(noisy_sizes, noisy)
    guess_n = extrapolate(noisy_sizes, noisy, [1e10])
    print(f"noisy fit: k = {k_n:.4f}, prediction at 10 billion = {guess_n[0]:.4f}")
