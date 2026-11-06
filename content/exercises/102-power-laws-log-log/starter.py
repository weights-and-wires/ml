"""Power laws on log paper.

Fill in the three functions below. `power_law_eval` evaluates y = c * x**k.
`fit_power_law` recovers k and c from measurements by fitting a straight line
to the logs. `extrapolate` fits and then predicts at sizes you never measured.
"""

import numpy as np


def power_law_eval(x, k: float, c: float) -> np.ndarray:
    """Evaluate the power law y = c * x**k.

    Args:
        x: Positive sizes. A float, a list, or an array.
        k: The exponent.
        c: The coefficient, the value of y at x = 1.

    Returns:
        A float array with the same shape as `x`.
    """
    raise NotImplementedError("power_law_eval")


def fit_power_law(x, y) -> tuple:
    """Fit y = c * x**k by least squares on (log x, log y).

    The line is log(y) = log(c) + k * log(x). Its slope is k. Its intercept is
    log(c), so c is the exponential of the intercept.

    Args:
        x: Positive sizes, shape (n,). A list or an array.
        y: Positive measurements, shape (n,).

    Returns:
        A tuple (k, c) of plain Python floats.
    """
    raise NotImplementedError("fit_power_law")


def extrapolate(x_train, y_train, x_new) -> np.ndarray:
    """Fit a power law to the training points and predict at `x_new`.

    Args:
        x_train: Positive sizes you measured, shape (n,).
        y_train: Positive measurements, shape (n,).
        x_new: Positive sizes to predict at, shape (m,). A list or an array.

    Returns:
        Predicted values, shape (m,).
    """
    raise NotImplementedError("extrapolate")


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
