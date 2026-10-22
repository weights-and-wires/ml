"""Checks for the batch norm and layer norm exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, has, shape


@check("normalise_group turns [1, 2, 3] into about [-1.2247, 0, 1.2247]")
def test_normalise_group_by_hand(sub):
    normalise_group = has(sub, "normalise_group")
    out = normalise_group(np.array([1.0, 2.0, 3.0]), eps=0.0)
    close(out, np.array([-1.224744871, 0.0, 1.224744871]), atol=1e-6, label="[1, 2, 3] with eps = 0")
    out = normalise_group(np.array([1.0, 2.0, 3.0]))
    close(out, np.array([-1.2247, 0.0, 1.2247]), atol=1e-3, label="[1, 2, 3] with the default eps")


@check("normalise_group gives mean 0 and variance 1, and ignores shift and scale")
def test_normalise_group_properties(sub):
    normalise_group = has(sub, "normalise_group")
    rng = np.random.default_rng(0)
    x = rng.normal(loc=4.0, scale=3.0, size=50)
    out = normalise_group(x)
    almost(out.mean(), 0.0, tol=1e-9, label="mean")
    almost(out.var(), 1.0, tol=1e-3, label="variance")
    close(normalise_group(10.0 * x + 37.5), out, atol=1e-4, label="after scaling by 10 and shifting by 37.5")


@check("normalise_group stays finite on a constant group, because of eps")
def test_normalise_group_constant(sub):
    normalise_group = has(sub, "normalise_group")
    out = normalise_group(np.array([5.0, 5.0, 5.0]))
    if not np.all(np.isfinite(out)):
        fail(f"a constant group gave {out}; adding eps to the variance keeps the division finite")
    close(out, np.zeros(3), atol=1e-12, label="[5, 5, 5]")


@check("batchnorm_forward gives each column mean 0 and variance 1, and applies gamma and beta")
def test_batchnorm_forward(sub):
    batchnorm_forward = has(sub, "batchnorm_forward")
    rng = np.random.default_rng(0)
    x = rng.normal(loc=5.0, scale=3.0, size=(500, 6))
    out, cache = batchnorm_forward(x, np.ones(6), np.zeros(6))
    shape(out, x.shape, label="batchnorm_forward output")
    close(out.mean(axis=0), np.zeros(6), atol=1e-6, label="column means")
    close(out.var(axis=0), np.ones(6), atol=1e-3, label="column variances")
    gamma = np.linspace(0.5, 2.0, 6)
    beta = np.linspace(-1.0, 1.0, 6)
    out2, _ = batchnorm_forward(x, gamma, beta)
    close(out2.mean(axis=0), beta, atol=1e-6, label="column means with beta")
    close(out2.std(axis=0), gamma, atol=1e-2, label="column spreads with gamma")
    if not isinstance(cache, tuple) or len(cache) != 4:
        fail("the cache should be the tuple (xhat, istd, gamma, axis)")


@check("layernorm_forward gives each row mean 0 and variance 1, and applies gamma and beta")
def test_layernorm_forward(sub):
    layernorm_forward = has(sub, "layernorm_forward")
    rng = np.random.default_rng(1)
    x = rng.normal(loc=-2.0, scale=4.0, size=(50, 200))
    out, cache = layernorm_forward(x, np.ones(200), np.zeros(200))
    shape(out, x.shape, label="layernorm_forward output")
    close(out.mean(axis=1), np.zeros(50), atol=1e-6, label="row means")
    close(out.var(axis=1), np.ones(50), atol=1e-3, label="row variances")
    gamma = rng.normal(size=200) + 2.0
    beta = rng.normal(size=200)
    out2, _ = layernorm_forward(x, gamma, beta)
    close(out2, gamma * out + beta, atol=1e-9, label="output equals gamma * xhat + beta")
    if not isinstance(cache, tuple) or len(cache) != 4:
        fail("the cache should be the tuple (xhat, istd, gamma, axis)")


@check("a batch of one: batch norm returns beta exactly, layer norm does not")
def test_batch_of_one(sub):
    batchnorm_forward = has(sub, "batchnorm_forward")
    layernorm_forward = has(sub, "layernorm_forward")
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    beta = np.array([0.5, -1.0, 2.0, 0.0])
    out, _ = batchnorm_forward(x, np.ones(4), beta)
    close(out, beta[None, :], atol=1e-9, label="batch norm of a single example")
    ln, _ = layernorm_forward(x, np.ones(4), np.zeros(4))
    close(ln, np.array([[-1.3416, -0.4472, 0.4472, 1.3416]]), atol=1e-3, label="layer norm of the same example")


@check("layer norm ignores a shift of one example; batch norm does not")
def test_layernorm_invariance(sub):
    batchnorm_forward = has(sub, "batchnorm_forward")
    layernorm_forward = has(sub, "layernorm_forward")
    rng = np.random.default_rng(4)
    x = rng.normal(size=(10, 6))
    gamma, beta = np.ones(6), np.zeros(6)
    shifted = x.copy()
    shifted[0] += 37.5

    ln_before, _ = layernorm_forward(x, gamma, beta)
    ln_after, _ = layernorm_forward(shifted, gamma, beta)
    close(ln_after[0], ln_before[0], atol=1e-6, label="layer norm's row 0 after the shift")

    bn_before, _ = batchnorm_forward(x, gamma, beta)
    bn_after, _ = batchnorm_forward(shifted, gamma, beta)
    moved = float(np.abs(bn_after[0] - bn_before[0]).max())
    if moved < 1e-3:
        fail(
            "batch norm normalises down each column, so shifting one example moves the column "
            f"statistics and that example's output, but it moved only {moved:.3g}"
        )
