"""Scaling laws: fit, split the loss, price a run, and spend a budget.

Fill in the eleven functions below. `fit_power_law` and `extrapolate` fit a
power law on log axes and predict past the data. `loss_terms`, `predict_loss`,
and `marginal_gain` split a loss into a floor and two penalties. `transformer_params`
counts the weights that multiply, `training_flops` and `training_days` price a
run, `chinchilla_optimal` and `grid_optimal` find the best split of a compute
budget, and `lifetime_flops` adds the cost of serving.
"""

import numpy as np

# (E, A, B, alpha, beta). A stylised fit, not a published one.
COEFFS = (1.69, 400.0, 1100.0, 0.34, 0.34)

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
        Predicted values c * x_new**k, shape (m,).
    """
    raise NotImplementedError("extrapolate")


def loss_terms(N, D, coeffs: tuple = COEFFS) -> tuple:
    """Split the loss into its three parts.

    Args:
        N: Non-embedding parameter count. A float or an array.
        D: Number of training tokens. A float or an array.
        coeffs: A 5-tuple (E, A, B, alpha, beta), in that order.

    Returns:
        A tuple (floor, size_penalty, data_penalty), where floor is E,
        size_penalty is A / N**alpha, and data_penalty is B / D**beta. Do not
        call float() on N or D: grid_optimal passes arrays.
    """
    raise NotImplementedError("loss_terms")


def predict_loss(N, D, coeffs: tuple = COEFFS):
    """Evaluate the scaling law L(N, D) = E + A / N**alpha + B / D**beta.

    Args:
        N: Non-embedding parameter count. A float or an array.
        D: Number of training tokens. A float or an array.
        coeffs: A 5-tuple (E, A, B, alpha, beta), in that order.

    Returns:
        The predicted loss, the sum of loss_terms, with the shape of the
        inputs.
    """
    raise NotImplementedError("predict_loss")


def marginal_gain(N, D, coeffs: tuple = COEFFS, factor: float = 2.0) -> float:
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


def transformer_params(
    layers: int, d_model: int, d_ff: int, vocab: int, seq_len: int
) -> dict:
    """Count a transformer's parameters, broken down by component.

    Assumes four square attention matrices per block (Q, K, V, and the output
    matrix, each d_model by d_model), a feed-forward network with two matrices
    (d_model by d_ff, then d_ff by d_model), a token table of shape
    (vocab, d_model), a position table of shape (seq_len, d_model), and an
    output layer that reuses the token table (no extra parameters). Biases and
    layer-norm parameters are left out.

    Args:
        layers: Number of blocks.
        d_model: Model width.
        d_ff: Feed-forward hidden width.
        vocab: Vocabulary size.
        seq_len: Number of positions in the position table.

    Returns:
        A dict of integers with keys "attention", "feedforward", "embedding",
        "non_embedding" (attention plus feedforward, summed over all blocks),
        and "total" (non_embedding plus embedding).
    """
    raise NotImplementedError("transformer_params")


def training_flops(params, tokens) -> float:
    """Estimate training compute as 6 * params * tokens.

    Convert to float before multiplying, so that numpy integer inputs do not
    wrap around.

    Args:
        params: Non-embedding parameter count.
        tokens: Number of training tokens.

    Returns:
        The estimated training compute, in flops, as a float.
    """
    raise NotImplementedError("training_flops")


def training_days(flops: float, flops_per_second: float = 2e14) -> float:
    """Convert a flop count into days on one accelerator.

    Args:
        flops: Total flops to run.
        flops_per_second: Sustained speed of the accelerator.

    Returns:
        The time in days (86,400 seconds each), as a float.
    """
    raise NotImplementedError("training_days")


def chinchilla_optimal(compute: float) -> tuple:
    """Return the compute-optimal (N*, D*) for a budget, from the closed form.

    Minimises L(N, D) = E + A / N**alpha + B / D**beta subject to
    6 * N * D = compute, using COEFFS = (E, A, B, alpha, beta).

    Args:
        compute: The training budget in flops.

    Returns:
        A tuple (N_star, D_star) of floats with 6 * N_star * D_star equal to
        `compute`.
    """
    raise NotImplementedError("chinchilla_optimal")


def grid_optimal(compute: float, n_points: int = 2000) -> tuple:
    """Find the best split by searching a grid of model sizes.

    Try `n_points` sizes N, log-spaced from sqrt(compute / 6) / 100 to
    sqrt(compute / 6) * 100. For each, spend the rest of the budget on tokens,
    D = compute / (6 * N), evaluate predict_loss, and keep the size with the
    lowest loss.

    Args:
        compute: The training budget in flops.
        n_points: Number of grid sizes.

    Returns:
        A tuple (N, D) of floats from the grid, with 6 * N * D equal to
        `compute`.
    """
    raise NotImplementedError("grid_optimal")


def lifetime_flops(n_params, train_tokens, served_tokens) -> float:
    """Total flops to train a model and then serve it.

    Training costs 6 * N * D_train. Each served token costs 2 * N, the forward
    pass alone.

    Args:
        n_params: Non-embedding parameter count N.
        train_tokens: Tokens used in training.
        served_tokens: Tokens the model reads or writes after training.

    Returns:
        Total flops as a float. Convert the inputs to float before multiplying.
    """
    raise NotImplementedError("lifetime_flops")


if __name__ == "__main__":
    # Five models, 10 million to 1 billion parameters, and the loss each one
    # sits above its floor.
    sizes = np.array([1e7, 3e7, 1e8, 3e8, 1e9])
    excess = 400.0 / sizes**0.34
    k, c = fit_power_law(sizes, excess)
    print(f"fitted exponent k = {k:.4f}, coefficient c = {c:.2f}")
    print(f"each doubling multiplies the loss by {2.0**k:.4f}")
    print(f"predicted excess loss at 10 billion parameters: {extrapolate(sizes, excess, [1e10])[0]:.4f}")

    for N, D in [(1e9, 2e9), (1e9, 2e10), (1e10, 2e9)]:
        floor, size_p, data_p = loss_terms(N, D, COEFFS)
        bigger = "size" if size_p > data_p else "data"
        print(
            f"N={N:.0e} D={D:.0e}: floor {floor:.3f} + size {size_p:.3f} + data {data_p:.3f}"
            f" = {predict_loss(N, D, COEFFS):.4f}   larger penalty: {bigger}"
        )
    print(f"doubling N at (1e9, 2e9) buys {marginal_gain(1e9, 2e9, COEFFS):.4f}")

    toy = transformer_params(layers=2, d_model=8, d_ff=32, vocab=100, seq_len=16)
    print("toy model:", toy)
    big = transformer_params(48, 1600, 6400, 50257, 1024)
    flops = training_flops(big["non_embedding"], 300_000_000_000)
    print(f"1.5B-parameter model on 300B tokens: {flops:.3e} flops, {training_days(flops):.0f} days")

    for compute in (1e18, 1e20, 1e22, 1e24):
        n_star, d_star = chinchilla_optimal(compute)
        loss = predict_loss(n_star, d_star, COEFFS)
        print(f"C={compute:.0e}: N*={n_star:.3e}, D*={d_star:.3e}, D*/N*={d_star / n_star:.2f}, loss={loss:.4f}")

    n_grid, _ = grid_optimal(1e20)
    n_star, _ = chinchilla_optimal(1e20)
    print(f"grid search at 1e20: N={n_grid:.3e} ({n_grid / n_star - 1:+.2%} from the closed form)")

    # A model half the size, trained long enough to reach the same loss.
    n_big, d_big = chinchilla_optimal(1.2e20)
    target = predict_loss(n_big, d_big, COEFFS)
    n_small = n_big / 2
    E, A, B, alpha, beta = COEFFS
    d_small = (B / (target - E - A / n_small**alpha)) ** (1.0 / beta)
    for served in (0.0, 1e12):
        print(
            f"serving {served:.0e} tokens: balanced {lifetime_flops(n_big, d_big, served):.3e},"
            f" half-size {lifetime_flops(n_small, d_small, served):.3e} flops"
        )
