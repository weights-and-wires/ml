"""Splitting a compute budget.

Fill in the three functions below. `chinchilla_optimal` evaluates the closed
form for the best model size and token count. `grid_optimal` finds the same
answer by brute force. `lifetime_flops` adds the cost of serving the model to
the cost of training it. `predict_loss` is the function you wrote in the
chapter "Loss from size and data"; it is already filled in.
"""

import numpy as np

# (E, A, B, alpha, beta). A stylised fit, not a published one.
CHINCHILLA_COEFFS = (1.69, 400.0, 1100.0, 0.34, 0.34)


# Written in the chapter "Loss from size and data".
def predict_loss(N, D, coeffs: tuple = CHINCHILLA_COEFFS):
    E, A, B, alpha, beta = coeffs
    return E + A / N**alpha + B / D**beta


def chinchilla_optimal(compute: float) -> tuple:
    """Return the compute-optimal (N*, D*) for a budget, from the closed form.

    Minimises L(N, D) = E + A / N**alpha + B / D**beta subject to
    6 * N * D = compute, using CHINCHILLA_COEFFS = (E, A, B, alpha, beta).

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
    for compute in (1e18, 1e20, 1e22, 1e24):
        n_star, d_star = chinchilla_optimal(compute)
        loss = predict_loss(n_star, d_star)
        print(f"C={compute:.0e}: N*={n_star:.3e}, D*={d_star:.3e}, D*/N*={d_star / n_star:.2f}, loss={loss:.4f}")

    n_grid, d_grid = grid_optimal(1e20)
    n_star, _ = chinchilla_optimal(1e20)
    print(f"\ngrid search at 1e20: N={n_grid:.3e} ({n_grid / n_star - 1:+.2%} from the closed form)")

    # Same compute, three splits.
    for n, d in [(2.5e8, 8e10), (1e9, 2e10), (4e9, 5e9)]:
        print(f"N={n:.1e}, D={d:.1e}: 6ND={6 * n * d:.2e}, loss={predict_loss(n, d):.4f}")

    # A model half the size, trained long enough to reach the same loss.
    n_big, d_big = chinchilla_optimal(1.2e20)
    target = predict_loss(n_big, d_big)
    n_small = n_big / 2
    E, A, B, alpha, beta = CHINCHILLA_COEFFS
    d_small = (B / (target - E - A / n_small**alpha)) ** (1.0 / beta)
    for served in (0.0, 1e12):
        big = lifetime_flops(n_big, d_big, served)
        small = lifetime_flops(n_small, d_small, served)
        print(f"serving {served:.0e} tokens: balanced {big:.3e}, half-size {small:.3e} flops")
