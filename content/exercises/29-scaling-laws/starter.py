"""Counting compute, and finding where it should go.

Fill in the five functions below. `transformer_params` and `training_flops`
turn an architecture into a parameter count and a compute estimate.
`chinchilla_optimal` evaluates the closed-form compute-optimal split derived in
the lesson. `fit_power_law` and `predict_loss` fit and evaluate the power law
the whole lesson is built on.

The Chinchilla-style constants below are data, not the answer: they are what
`chinchilla_optimal` and `predict_loss` are evaluated against, not something you
need to derive. The derivation you must supply is the closed form itself.
"""

import numpy as np

# E + A / N**alpha + B / D**beta, in that order. Close in spirit to the
# published fit, and the numbers this lesson's worked examples use.
CHINCHILLA_COEFFS = (1.69, 400.0, 1100.0, 0.34, 0.34)


def transformer_params(
    layers: int, d_model: int, d_ff: int, vocab: int, seq_len: int
) -> dict:
    """Count a transformer's parameters, broken down by component.

    Assumes four square attention projections per layer (W_Q, W_K, W_V, W_O,
    each d_model x d_model), a feed-forward sublayer of shape
    (d_model, d_ff) then (d_ff, d_model), a token embedding table of shape
    (vocab, d_model), a learned position embedding table of shape
    (seq_len, d_model), and an output projection tied to the token embedding
    (no extra parameters). Biases and layer-norm parameters are lower order
    and omitted.

    Args:
        layers: Number of transformer blocks.
        d_model: Model width.
        d_ff: Feed-forward hidden width.
        vocab: Vocabulary size.
        seq_len: Length of the learned position embedding table.

    Returns:
        A dict with integer keys "attention", "feedforward", "embedding",
        "non_embedding" (attention plus feedforward, summed over all layers),
        and "total" (non_embedding plus embedding).
    """
    raise NotImplementedError("transformer_params")


def training_flops(params: int, tokens: int) -> float:
    """Estimate training compute as 6 * params * tokens.

    `params` should be the non-embedding parameter count: embedding lookups
    are indexing operations, not matrix multiplies, and cost no flops.

    Args:
        params: Non-embedding parameter count.
        tokens: Number of training tokens.

    Returns:
        The estimated training compute, in flops.
    """
    raise NotImplementedError("training_flops")


def chinchilla_optimal(compute: float) -> tuple:
    """Return the compute-optimal (N*, D*) for a given compute budget.

    Minimises L(N, D) = E + A / N**alpha + B / D**beta subject to
    C = 6 * N * D, using CHINCHILLA_COEFFS = (E, A, B, alpha, beta). Solve the
    Lagrange stationarity condition in closed form; do not run a numerical
    optimiser.

    Args:
        compute: The training compute budget, in flops.

    Returns:
        A tuple (N_star, D_star) of floats satisfying compute == 6 * N_star *
        D_star to high precision.
    """
    raise NotImplementedError("chinchilla_optimal")


def fit_power_law(x: np.ndarray, y: np.ndarray) -> tuple:
    """Fit y = c * x**k by ordinary least squares in log space.

    Take logs of both arrays and fit a straight line: log(y) = log(c) +
    k * log(x). The slope is the exponent, the intercept exponentiates to the
    coefficient.

    Args:
        x: Positive values, shape (n,).
        y: Positive values, shape (n,).

    Returns:
        A tuple (k, c): the fitted exponent and coefficient.
    """
    raise NotImplementedError("fit_power_law")


def predict_loss(N: float, D: float, coeffs: tuple) -> float:
    """Evaluate the scaling law L(N, D) = E + A / N**alpha + B / D**beta.

    Args:
        N: Non-embedding parameter count.
        D: Number of training tokens.
        coeffs: A 5-tuple (E, A, B, alpha, beta).

    Returns:
        The predicted loss, as a float.
    """
    raise NotImplementedError("predict_loss")


if __name__ == "__main__":
    breakdown = transformer_params(layers=12, d_model=768, d_ff=3072, vocab=50257, seq_len=1024)
    print("GPT-2-small-sized architecture:")
    for key, value in breakdown.items():
        print(f"  {key:>14}: {value:,}")

    flops = training_flops(breakdown["non_embedding"], tokens=300_000_000_000)
    print(f"\ntraining compute for 300B tokens: {flops:.3e} flops")

    for compute in (1e18, 1e20, 1e22, 1e24):
        n_star, d_star = chinchilla_optimal(compute)
        ratio = d_star / n_star
        print(
            f"C={compute:.0e}: N*={n_star:.3e}, D*={d_star:.3e}, "
            f"D*/N*={ratio:.2f}, loss={predict_loss(n_star, d_star, CHINCHILLA_COEFFS):.4f}"
        )

    rng = np.random.default_rng(0)
    x = np.geomspace(1.0, 1000.0, 10)
    y = 3.0 * x ** 1.7
    k, c = fit_power_law(x, y)
    print(f"\nplanted exponent 1.7, coefficient 3.0 -> fit gives k={k:.4f}, c={c:.4f}")
