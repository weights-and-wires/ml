"""Checks for the scaling-laws exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, has, shape

COEFFS = (1.69, 400.0, 1100.0, 0.34, 0.34)
E_, A_, B_, ALPHA, BETA = COEFFS


def loss(n, d):
    return E_ + A_ / n**ALPHA + B_ / d**BETA


@check("fit_power_law recovers a planted law, the hand example, and a noisy exponent")
def test_fit(sub):
    fit = has(sub, "fit_power_law")
    x = np.geomspace(1.0, 1000.0, 12)
    k, c = fit(x, 3.0 * x**1.7)
    almost(k, 1.7, tol=1e-9, label="exponent")
    if abs(c - np.log(3.0)) < 1e-3:
        fail("fit_power_law returned log(c), the intercept. Return np.exp(intercept) instead.")
    almost(c, 3.0, tol=1e-8, label="coefficient")
    k, c = fit([1, 2, 4], [1.667, 1.317, 1.040])
    almost(k, -0.3403, tol=2e-4, label="hand-example exponent")
    almost(c, 1.6671, tol=2e-3, label="hand-example coefficient")
    if not (isinstance(k, float) and isinstance(c, float)):
        fail(f"fit_power_law should return plain floats, got {type(k).__name__} and {type(c).__name__}")
    rng = np.random.default_rng(0)
    xs = np.geomspace(1e7, 1e9, 30)
    ys = 400.0 / xs**0.34 * np.exp(rng.normal(0.0, 0.03, xs.size))
    k, c = fit(xs, ys)
    almost(k, -0.34, tol=0.01, label="exponent under noise")
    almost(c, 400.0, tol=40.0, label="coefficient under noise")


@check("extrapolate matches the true law on clean data and stays within 10 percent on noisy data")
def test_extrapolate(sub):
    ex = has(sub, "extrapolate")
    x = np.array([1.0, 3.0, 10.0, 30.0, 100.0])
    got = ex(x, 1.667 * x**-0.34, [300.0, 1000.0])
    shape(got, (2,), label="extrapolate output shape")
    close(got, 1.667 * np.array([300.0, 1000.0]) ** -0.34, rtol=1e-6, label="prediction")
    rng = np.random.default_rng(3)
    xs = np.geomspace(1.0, 100.0, 8)
    ys = 1.667 * xs**-0.34 * np.exp(rng.normal(0.0, 0.03, xs.size))
    near = ex(xs, ys, [100.0])[0]
    far = ex(xs, ys, [1000.0])[0]
    true_far = 1.667 * 1000.0**-0.34
    err_far = abs(far - true_far) / true_far
    if err_far > 0.10:
        fail(f"prediction at 1000 is {far:.4f}, true value {true_far:.4f}: {err_far:.1%} off, expected under 10%")
    true_near = 1.667 * 100.0**-0.34
    err_near = abs(near - true_near) / true_near
    if not err_near < err_far:
        fail(f"error inside the data ({err_near:.2%}) should be smaller than far outside it ({err_far:.2%})")


@check("loss_terms returns (floor, size penalty, data penalty) in the order (E, A, B, alpha, beta)")
def test_terms(sub):
    lt = has(sub, "loss_terms")
    terms = lt(1e9, 2e10, COEFFS)
    if len(terms) != 3:
        fail(f"loss_terms should return 3 values, got {len(terms)}")
    floor, size_p, data_p = terms
    almost(floor, 1.69, tol=1e-12, label="floor")
    almost(size_p, 0.34839, tol=2e-5, label="size penalty at N = 1e9")
    almost(data_p, 0.34597, tol=2e-5, label="data penalty at D = 2e10")
    # 1 + 2 / 4**0.5 + 3 / 16**0.25 = 1 + 1 + 1.5
    floor, size_p, data_p = lt(4.0, 16.0, (1.0, 2.0, 3.0, 0.5, 0.25))
    almost(size_p, 1.0, tol=1e-12, label="size penalty with odd coefficients")
    almost(data_p, 1.5, tol=1e-12, label="data penalty with odd coefficients")


@check("predict_loss gives 2.3844 for the example, equals the sum of the terms, and accepts arrays")
def test_predict(sub):
    pl, lt = has(sub, "predict_loss"), has(sub, "loss_terms")
    almost(pl(1e9, 2e10, COEFFS), 2.38436, tol=2e-5, label="predict_loss(1e9, 2e10)")
    for N, D in [(1e8, 1e9), (3e9, 5e10), (2e11, 4e12)]:
        almost(pl(N, D, COEFFS), sum(lt(N, D, COEFFS)), tol=1e-9, label=f"sum of terms at N={N:.0e}, D={D:.0e}")
    almost(pl(4.0, 16.0, (1.0, 2.0, 3.0, 0.5, 0.25)), 3.5, tol=1e-12, label="predict_loss with odd coefficients")
    over_n = [pl(n, 1e10, COEFFS) for n in (1e7, 1e8, 1e9, 1e10, 1e11)]
    over_d = [pl(1e9, d, COEFFS) for d in (1e8, 1e9, 1e10, 1e11, 1e12)]
    if not all(a > b for a, b in zip(over_n, over_n[1:])):
        fail(f"loss should fall as N grows, got {over_n}")
    if not all(a > b for a, b in zip(over_d, over_d[1:])):
        fail(f"loss should fall as D grows, got {over_d}")
    almost(pl(1e60, 1e60, COEFFS), 1.69, tol=1e-6, label="loss at enormous N and D")
    arr = pl(np.array([1e8, 1e9]), np.array([1e9, 2e10]), COEFFS)
    shape(arr, (2,), label="predict_loss on arrays")
    close(arr, [pl(1e8, 1e9, COEFFS), pl(1e9, 2e10, COEFFS)], rtol=1e-12, label="array and scalar results")


@check("marginal_gain matches A / N**alpha * (1 - factor**-alpha) and does not depend on D")
def test_gain(sub):
    mg, pl = has(sub, "marginal_gain"), has(sub, "predict_loss")
    for N, factor in [(1e9, 2.0), (3e8, 4.0), (5e10, 10.0)]:
        want = A_ / N**ALPHA * (1.0 - factor**-ALPHA)
        almost(mg(N, 2e10, COEFFS, factor), want, tol=1e-9, label=f"gain at N={N:.0e}, factor={factor:g}")
    almost(mg(1e9, 2e10, COEFFS), 0.07315, tol=2e-5, label="default factor of 2 at the example")
    almost(mg(1e9, 2e9, COEFFS), mg(1e9, 2e12, COEFFS), tol=1e-9, label="gain at D=2e9 versus D=2e12")
    gain_n = mg(1e9, 2e9, COEFFS)
    gain_d = pl(1e9, 2e9, COEFFS) - pl(1e9, 4e9, COEFFS)
    if not gain_d > 2 * gain_n:
        fail(f"at N=1e9, D=2e9 doubling the data should gain more than twice the model's {gain_n:.4f}, got {gain_d:.4f}")


@check("transformer_params matches the toy hand count, counts embeddings once, and scales with depth and width")
def test_params(sub):
    tp = has(sub, "transformer_params")
    got = tp(layers=2, d_model=8, d_ff=32, vocab=100, seq_len=16)
    want = {"attention": 512, "feedforward": 1024, "embedding": 928, "non_embedding": 1536, "total": 2464}
    for key, value in want.items():
        if key not in got:
            fail(f"transformer_params output has no {key!r} key")
        if got[key] != value:
            fail(f"transformer_params()[{key!r}] is {got[key]:,}, expected {value:,} for the toy model")
        if not isinstance(got[key], (int, np.integer)):
            fail(f"transformer_params()[{key!r}] should be an integer, got {type(got[key]).__name__}")
    two = tp(layers=2, d_model=64, d_ff=256, vocab=1000, seq_len=128)
    six = tp(layers=6, d_model=64, d_ff=256, vocab=1000, seq_len=128)
    if two["embedding"] != six["embedding"]:
        fail(f"embedding count changed with layers: {two['embedding']:,} versus {six['embedding']:,}")
    almost(six["non_embedding"] / two["non_embedding"], 3.0, tol=1e-12, label="non_embedding ratio for 6 blocks versus 2")
    small = tp(layers=6, d_model=32, d_ff=128, vocab=500, seq_len=64)
    large = tp(layers=6, d_model=64, d_ff=256, vocab=500, seq_len=64)
    almost(large["non_embedding"] / small["non_embedding"], 4.0, tol=1e-12, label="ratio under doubled width")
    for layers, d in [(3, 16), (12, 96)]:
        n = tp(layers=layers, d_model=d, d_ff=4 * d, vocab=1000, seq_len=100)["non_embedding"]
        if n != 12 * layers * d * d:
            fail(f"with d_ff = 4 * d_model, non_embedding should be 12 * layers * d_model**2 = {12 * layers * d * d:,}, got {n:,}")


@check("training_flops is 6 N D without int64 overflow, and training_days turns 48 blocks into 154 days")
def test_flops_days(sub):
    tf, td, tp = has(sub, "training_flops"), has(sub, "training_days"), has(sub, "transformer_params")
    almost(tf(1_000_000, 500_000), 3.0e12, tol=1.0, label="training_flops(1e6, 5e5)")
    almost(tf(1.5e9, 3e11) / 2.7e21, 1.0, tol=1e-12, label="training_flops(1.5e9, 3e11) relative to 2.7e21")
    got = tf(np.int64(7_000_000_000), np.int64(300_000_000_000))
    want = 6.0 * 7e9 * 3e11
    if not np.isfinite(got) or got <= 0 or abs(got - want) > 1e-9 * want:
        fail(f"training_flops(int64 7e9, int64 3e11) returned {got!r}, expected {want:.4e}. Convert to float before multiplying.")
    almost(td(1e20), 5.787037, tol=1e-5, label="days for 1e20 flops")
    almost(td(1e20, 1e15), 1.157407, tol=1e-5, label="days at 1e15 flops per second")
    almost(td(2e20) / td(1e20), 2.0, tol=1e-12, label="time doubles with flops")
    b = tp(48, 1600, 6400, 50257, 1024)
    if b["non_embedding"] != 1_474_560_000:
        fail(f"non_embedding for 48 blocks, width 1600, d_ff 6400 is {b['non_embedding']:,}, expected 1,474,560,000")
    flops = tf(b["non_embedding"], 300_000_000_000)
    almost(flops / 2.6542e21, 1.0, tol=1e-4, label="flops for 300B tokens, relative to 2.6542e21")
    almost(td(flops), 153.6, tol=0.1, label="days on one accelerator")


@check("chinchilla_optimal spends the budget, sits at the bottom of the U, and keeps D*/N* near 19.6")
def test_optimal(sub):
    co = has(sub, "chinchilla_optimal")
    for compute in (1e18, 1e20, 1e22, 1e24):
        n, d = co(compute)
        if n <= 0 or d <= 0:
            fail(f"chinchilla_optimal({compute:.0e}) returned a non-positive value")
        close(6.0 * n * d, compute, rtol=1e-9, label=f"6 N* D* at compute={compute:.0e}")
    n, d = co(1e20)
    close(n, 9.2225e8, rtol=1e-4, label="N* at 1e20")
    close(d, 1.8072e10, rtol=1e-4, label="D* at 1e20")
    for compute in (1e19, 1e21):
        n, d = co(compute)
        best = loss(n, d)
        for factor in (0.8, 0.95, 1.05, 1.25):
            other = loss(factor * n, compute / (6.0 * factor * n))
            if other <= best:
                fail(f"at compute={compute:.0e}, scaling N* by {factor} gave loss {other:.6f}, not above {best:.6f}")
        close(ALPHA * A_ / n**ALPHA, BETA * B_ / d**BETA, rtol=1e-9, label="alpha A / N**alpha versus beta B / D**beta")
    for compute in (1e18, 1e21, 1e24):
        n, d = co(compute)
        almost(d / n, 19.5954, tol=1e-3, label=f"D*/N* at compute={compute:.0e}")
    n1, _ = co(1e20)
    n2, _ = co(1e22)
    almost(n2 / n1, 10.0, tol=1e-9, label="N* ratio for a hundredfold budget")


@check("grid_optimal spends the budget and agrees with the closed form to within 1 percent")
def test_grid(sub):
    go, co = has(sub, "grid_optimal"), has(sub, "chinchilla_optimal")
    n, d = go(1e20)
    close(6.0 * n * d, 1e20, rtol=1e-9, label="6 N D from the grid")
    for compute in (1e18, 1e20, 1e23):
        ng, dg = go(compute)
        nc, dc = co(compute)
        if abs(ng / nc - 1) > 0.01 or abs(dg / dc - 1) > 0.01:
            fail(f"at compute={compute:.0e}, grid gave N={ng:.4e}, D={dg:.4e}; closed form gave N={nc:.4e}, D={dc:.4e}")


@check("lifetime_flops is 6 N D_train + 2 N served, and a half-size model breaks even at 2.86e10 served tokens")
def test_lifetime(sub):
    lf, co = has(sub, "lifetime_flops"), has(sub, "chinchilla_optimal")
    almost(lf(1e9, 2e10, 0) / 1.2e20, 1.0, tol=1e-12, label="training only")
    almost(lf(1e9, 2e10, 1e12) / 2.12e21, 1.0, tol=1e-12, label="training plus serving 1e12 tokens")
    got = lf(np.int64(1_000_000_000), np.int64(20_000_000_000), np.int64(1_000_000_000_000))
    if not np.isfinite(got) or abs(got / 2.12e21 - 1) > 1e-12:
        fail(f"lifetime_flops with numpy int64 inputs returned {got!r}, expected 2.12e21. Convert to float first.")
    almost(lf(1e9, 2e10, 2e12) - lf(1e9, 2e10, 1e12), 2e21, tol=1e9, label="each served token costs 2 N")
    n_big, d_big = co(1.2e20)
    n_small = n_big / 2
    d_small = (B_ / (loss(n_big, d_big) - E_ - A_ / n_small**ALPHA)) ** (1.0 / BETA)
    if lf(n_big, d_big, 0) >= lf(n_small, d_small, 0):
        fail("with nothing served, the balanced model should be the cheaper one overall")
    served = (6 * n_small * d_small - 6 * n_big * d_big) / (2 * (n_big - n_small))
    almost(served / 2.8553e10, 1.0, tol=1e-3, label="break-even served tokens, relative to 2.8553e10")
    close(lf(n_big, d_big, served), lf(n_small, d_small, served), rtol=1e-9, label="equal lifetime flops at break-even")
    if lf(n_small, d_small, 1e12) >= lf(n_big, d_big, 1e12):
        fail("after serving 1e12 tokens the half-size model should be the cheaper one")
