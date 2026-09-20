"""Reference solution for the scaling-laws exercise."""

import numpy as np

CHINCHILLA_COEFFS = (1.69, 400.0, 1100.0, 0.34, 0.34)


def transformer_params(
    layers: int, d_model: int, d_ff: int, vocab: int, seq_len: int
) -> dict:
    attention = layers * 4 * d_model**2
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


def training_flops(params: int, tokens: int) -> float:
    return 6.0 * float(params) * float(tokens)


def chinchilla_optimal(compute: float) -> tuple:
    E, A, B, alpha, beta = CHINCHILLA_COEFFS
    a = beta / (alpha + beta)
    b = alpha / (alpha + beta)
    g = (alpha * A / (beta * B)) ** (1.0 / (alpha + beta))
    n_star = g * 6.0 ** (-a) * compute**a
    d_star = compute / (6.0 * n_star)
    return n_star, d_star


def fit_power_law(x: np.ndarray, y: np.ndarray) -> tuple:
    lx = np.log(np.asarray(x, dtype=float))
    ly = np.log(np.asarray(y, dtype=float))
    lx_mean, ly_mean = lx.mean(), ly.mean()
    lx_centered = lx - lx_mean
    k = float(np.sum(lx_centered * (ly - ly_mean)) / np.sum(lx_centered**2))
    intercept = ly_mean - k * lx_mean
    return k, float(np.exp(intercept))


def predict_loss(N: float, D: float, coeffs: tuple) -> float:
    E, A, B, alpha, beta = coeffs
    return E + A / N**alpha + B / D**beta


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
