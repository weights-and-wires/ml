"""Checks for the scaling-laws exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, has

CHINCHILLA_COEFFS = (1.69, 400.0, 1100.0, 0.34, 0.34)


@check("predict_loss evaluates E + A / N**alpha + B / D**beta")
def test_predict_loss(sub):
    got = has(sub, "predict_loss")
    E, A, B, alpha, beta = CHINCHILLA_COEFFS
    N, D = 1e9, 2e10
    want = E + A / N**alpha + B / D**beta
    almost(got(N, D, CHINCHILLA_COEFFS), want, tol=1e-9 * want, label="predict_loss")


@check("training_flops is exactly 6 * params * tokens")
def test_training_flops(sub):
    got = has(sub, "training_flops")
    for params, tokens in [(1_000_000, 500_000), (7_000_000_000, 300_000_000_000)]:
        want = 6.0 * params * tokens
        almost(got(params, tokens), want, tol=1e-6 * want, label="training_flops")


@check("transformer_params matches a hand-computed breakdown")
def test_transformer_params_hand(sub):
    got = has(sub, "transformer_params")
    breakdown = got(layers=4, d_model=64, d_ff=256, vocab=1000, seq_len=128)
    want = {
        "attention": 65536,
        "feedforward": 131072,
        "embedding": 72192,
        "non_embedding": 196608,
        "total": 268800,
    }
    for key, value in want.items():
        if key not in breakdown:
            fail(f"transformer_params output has no {key!r} key")
        got_value = breakdown[key]
        if got_value != value:
            fail(
                f"transformer_params()[{key!r}] is {got_value:,}, expected "
                f"{value:,} for layers=4, d_model=64, d_ff=256, vocab=1000, "
                "seq_len=128"
            )


@check("Doubling d_model, with d_ff scaled to match, quadruples non_embedding")
def test_transformer_params_scaling(sub):
    got = has(sub, "transformer_params")
    small = got(layers=6, d_model=32, d_ff=128, vocab=500, seq_len=64)
    large = got(layers=6, d_model=64, d_ff=256, vocab=500, seq_len=64)
    ratio = large["non_embedding"] / small["non_embedding"]
    almost(
        ratio,
        4.0,
        tol=1e-9,
        label="non_embedding parameter ratio under a doubling of d_model",
    )


@check("fit_power_law recovers a planted exponent from clean data")
def test_fit_power_law_clean(sub):
    got = has(sub, "fit_power_law")
    x = np.geomspace(1.0, 1_000.0, 12)
    k_true, c_true = 1.7, 3.0
    y = c_true * x**k_true
    k, c = got(x, y)
    almost(k, k_true, tol=1e-6, label="exponent on clean data")
    almost(c, c_true, tol=1e-6, label="coefficient on clean data")


@check("fit_power_law stays close to the planted exponent under noise")
def test_fit_power_law_noisy(sub):
    got = has(sub, "fit_power_law")
    rng = np.random.default_rng(0)
    x = np.geomspace(1.0, 1e6, 40)
    k_true, c_true = 0.5, 12.0
    log_y = np.log(c_true) + k_true * np.log(x) + rng.normal(scale=0.03, size=x.size)
    y = np.exp(log_y)
    k, c = got(x, y)
    almost(k, k_true, tol=0.02, label="exponent under noise")
    almost(c, c_true, tol=0.03 * c_true, label="coefficient under noise")


@check("chinchilla_optimal satisfies the compute constraint C = 6 N* D*")
def test_chinchilla_constraint(sub):
    got = has(sub, "chinchilla_optimal")
    for compute in (1e18, 1e20, 1e22, 1e24):
        n_star, d_star = got(compute)
        if n_star <= 0 or d_star <= 0:
            fail(f"chinchilla_optimal({compute:.0e}) returned a non-positive value")
        close(
            6.0 * n_star * d_star,
            compute,
            rtol=1e-6,
            label=f"6 N* D* at compute={compute:.0e}",
        )


@check("chinchilla_optimal's token-to-parameter ratio lands near 20")
def test_chinchilla_ratio(sub):
    got = has(sub, "chinchilla_optimal")
    for compute in (1e18, 1e21, 1e24):
        n_star, d_star = got(compute)
        ratio = d_star / n_star
        if not (10.0 <= ratio <= 30.0):
            fail(
                f"at compute={compute:.0e}, D*/N* is {ratio:.2f}, expected "
                "something near 20 tokens per parameter for CHINCHILLA_COEFFS"
            )


@check("predict_loss decreases monotonically in both N and D")
def test_predict_loss_monotonic(sub):
    got = has(sub, "predict_loss")
    D = 1e11
    losses_over_n = [got(n, D, CHINCHILLA_COEFFS) for n in (1e8, 1e9, 1e10, 1e11)]
    if not all(a > b for a, b in zip(losses_over_n, losses_over_n[1:])):
        fail(
            "predict_loss should strictly decrease as N grows with D fixed, "
            f"but got {losses_over_n}"
        )

    N = 1e10
    losses_over_d = [got(N, d, CHINCHILLA_COEFFS) for d in (1e9, 1e10, 1e11, 1e12)]
    if not all(a > b for a, b in zip(losses_over_d, losses_over_d[1:])):
        fail(
            "predict_loss should strictly decrease as D grows with N fixed, "
            f"but got {losses_over_d}"
        )
