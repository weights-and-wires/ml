"""Splitting the loss into a floor and two penalties.

Fill in the three functions below. `loss_terms` returns the floor, the size
penalty, and the data penalty. `predict_loss` adds them. `marginal_gain` says
how much the loss falls when the model grows and the data stays put.
`fit_power_law` is the function you wrote in the chapter "Power laws and the
straight line on log paper"; it is already filled in.
"""

import numpy as np

# (E, A, B, alpha, beta). A stylised fit, not a published one.
COEFFS = (1.69, 400.0, 1100.0, 0.34, 0.34)


# Written in the chapter "Power laws and the straight line on log paper".
def fit_power_law(x, y) -> tuple:
    u = np.log(np.asarray(x, dtype=float))
    v = np.log(np.asarray(y, dtype=float))
    uc = u - u.mean()
    k = float(np.sum(uc * (v - v.mean())) / np.sum(uc**2))
    return k, float(np.exp(v.mean() - k * u.mean()))


def loss_terms(N: float, D: float, coeffs: tuple) -> tuple:
    """Split the loss into its three parts.

    Args:
        N: Non-embedding parameter count.
        D: Number of training tokens.
        coeffs: A 5-tuple (E, A, B, alpha, beta), in that order.

    Returns:
        A tuple (floor, size_penalty, data_penalty) of floats, where floor is
        E, size_penalty is A / N**alpha, and data_penalty is B / D**beta.
    """
    raise NotImplementedError("loss_terms")


def predict_loss(N: float, D: float, coeffs: tuple) -> float:
    """Evaluate the scaling law L(N, D) = E + A / N**alpha + B / D**beta.

    Args:
        N: Non-embedding parameter count.
        D: Number of training tokens.
        coeffs: A 5-tuple (E, A, B, alpha, beta), in that order.

    Returns:
        The predicted loss as a float. It equals the sum of loss_terms.
    """
    raise NotImplementedError("predict_loss")


def marginal_gain(N: float, D: float, coeffs: tuple, factor: float = 2.0) -> float:
    """How much the loss falls when the model grows by `factor` and D stays fixed.

    Args:
        N: Non-embedding parameter count before the change.
        D: Number of training tokens, unchanged.
        coeffs: A 5-tuple (E, A, B, alpha, beta), in that order.
        factor: The multiplier on N. 2.0 means a model twice as large.

    Returns:
        The loss before the change minus the loss after it. Positive when the
        loss falls.
    """
    raise NotImplementedError("marginal_gain")


if __name__ == "__main__":
    for N, D in [(1e9, 2e9), (1e9, 2e10), (1e10, 2e9)]:
        floor, size_p, data_p = loss_terms(N, D, COEFFS)
        bigger = "size" if size_p > data_p else "data"
        print(
            f"N={N:.0e} D={D:.0e}: floor {floor:.3f} + size {size_p:.3f} + data {data_p:.3f}"
            f" = {predict_loss(N, D, COEFFS):.4f}   larger penalty: {bigger}"
        )

    print(f"\ndoubling N at (1e9, 2e9) buys {marginal_gain(1e9, 2e9, COEFFS):.4f}")
    gain_d = predict_loss(1e9, 2e9, COEFFS) - predict_loss(1e9, 4e9, COEFFS)
    print(f"doubling D at (1e9, 2e9) buys {gain_d:.4f}")

    # Recover alpha and A from a slice: the data is huge, and the floor is known.
    sizes = np.geomspace(1e7, 1e10, 8)
    D_big = 1e12
    excess = np.array([predict_loss(n, D_big, COEFFS) for n in sizes])
    excess = excess - COEFFS[0] - COEFFS[2] / D_big ** COEFFS[4]
    k, c = fit_power_law(sizes, excess)
    print(f"\nslice fit: alpha = {-k:.4f}, A = {c:.1f}")
