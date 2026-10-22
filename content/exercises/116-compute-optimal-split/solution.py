"""Reference solution for the compute-optimal-split exercise."""

import numpy as np

# (E, A, B, alpha, beta). A stylised fit, not a published one.
CHINCHILLA_COEFFS = (1.69, 400.0, 1100.0, 0.34, 0.34)


# Written in the chapter "Loss from size and data".
def predict_loss(N, D, coeffs: tuple = CHINCHILLA_COEFFS):
    E, A, B, alpha, beta = coeffs
    return E + A / N**alpha + B / D**beta


def chinchilla_optimal(compute: float) -> tuple:
    E, A, B, alpha, beta = CHINCHILLA_COEFFS
    a = beta / (alpha + beta)
    G = (alpha * A / (beta * B)) ** (1.0 / (alpha + beta))
    n_star = G * 6.0 ** (-a) * compute**a
    d_star = compute / (6.0 * n_star)
    return n_star, d_star


def grid_optimal(compute: float, n_points: int = 2000) -> tuple:
    centre = np.sqrt(compute / 6.0)
    sizes = np.geomspace(0.01, 100.0, n_points) * centre
    tokens = compute / (6.0 * sizes)
    losses = predict_loss(sizes, tokens, CHINCHILLA_COEFFS)
    best = int(np.argmin(losses))
    return float(sizes[best]), float(tokens[best])


def lifetime_flops(n_params, train_tokens, served_tokens) -> float:
    n, d, s = float(n_params), float(train_tokens), float(served_tokens)
    return 6.0 * n * d + 2.0 * n * s


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
