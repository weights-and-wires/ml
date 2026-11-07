"""Reference solution: fit a scaling law, price a run, and split its budget."""

import numpy as np

# (E, A, B, alpha, beta). A stylised fit, not a published one.
COEFFS = (1.69, 400.0, 1100.0, 0.34, 0.34)


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
    return c * np.asarray(x_new, dtype=float) ** k


def loss_terms(N, D, coeffs: tuple = COEFFS) -> tuple:
    E, A, B, alpha, beta = coeffs
    return E, A / N**alpha, B / D**beta


def predict_loss(N, D, coeffs: tuple = COEFFS):
    floor, size_penalty, data_penalty = loss_terms(N, D, coeffs)
    return floor + size_penalty + data_penalty


def marginal_gain(N, D, coeffs: tuple = COEFFS, factor: float = 2.0) -> float:
    return predict_loss(N, D, coeffs) - predict_loss(factor * N, D, coeffs)


def transformer_params(
    layers: int, d_model: int, d_ff: int, vocab: int, seq_len: int
) -> dict:
    attention = layers * 4 * d_model * d_model
    feedforward = layers * 2 * d_model * d_ff
    embedding = vocab * d_model + seq_len * d_model
    non_embedding = attention + feedforward
    return {
        "attention": attention,
        "feedforward": feedforward,
        "embedding": embedding,
        "non_embedding": non_embedding,
        "total": non_embedding + embedding,
    }


def training_flops(params, tokens) -> float:
    return 6.0 * float(params) * float(tokens)


def training_days(flops: float, flops_per_second: float = 2e14) -> float:
    return float(flops) / flops_per_second / 86400.0


def chinchilla_optimal(compute: float) -> tuple:
    E, A, B, alpha, beta = COEFFS
    a = beta / (alpha + beta)
    G = (alpha * A / (beta * B)) ** (1.0 / (alpha + beta))
    n_star = G * 6.0 ** (-a) * compute**a
    d_star = compute / (6.0 * n_star)
    return n_star, d_star


def grid_optimal(compute: float, n_points: int = 2000) -> tuple:
    centre = np.sqrt(compute / 6.0)
    sizes = np.geomspace(0.01, 100.0, n_points) * centre
    tokens = compute / (6.0 * sizes)
    losses = predict_loss(sizes, tokens, COEFFS)
    best = int(np.argmin(losses))
    return float(sizes[best]), float(tokens[best])


def lifetime_flops(n_params, train_tokens, served_tokens) -> float:
    n, d, s = float(n_params), float(train_tokens), float(served_tokens)
    return 6.0 * n * d + 2.0 * n * s


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
