"""Checks for the modern transformer choices exercise."""

import numpy as np
import torch
import torch.nn as nn

from mlcheck import check, close, equal, fail, finite_difference, has, shape

EPS = 1e-5


def ref_rmsnorm_row(row, gain):
    return row / np.sqrt(np.mean(row**2) + EPS) * gain


def ref_gqa(q, k, v, n_groups):
    """Loop reference: head h reads group h // (H // n_groups)."""
    H, T, dh = q.shape
    per = H // n_groups
    out = np.zeros_like(q)
    for h in range(H):
        g = h // per
        for t in range(T):
            scores = np.array([q[h, t] @ k[g, s] / np.sqrt(dh) for s in range(t + 1)])
            w = np.exp(scores - scores.max())
            w /= w.sum()
            out[h, t] = sum(w[s] * v[g, s] for s in range(t + 1))
    return out


class RMS(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))


def torch_model(cfg):
    """Build the model out of real modules so that numel() does the counting."""
    d, H, G = cfg["d_model"], cfg["n_heads"], cfg["n_kv_heads"]
    dh = d // H
    bias = cfg["bias"]
    make_norm = (lambda: nn.LayerNorm(d)) if cfg["norm"] == "layer" else (lambda: RMS(d))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1, self.n2 = make_norm(), make_norm()
            self.q = nn.Linear(d, H * dh, bias=bias)
            self.k = nn.Linear(d, G * dh, bias=bias)
            self.v = nn.Linear(d, G * dh, bias=bias)
            self.o = nn.Linear(H * dh, d, bias=bias)
            self.up = nn.Linear(d, cfg["d_ff"], bias=bias)
            self.down = nn.Linear(cfg["d_ff"], d, bias=bias)
            if cfg["gated"]:
                self.gate = nn.Linear(d, cfg["d_ff"], bias=bias)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(cfg["vocab"], d)
            if cfg["max_positions"]:
                self.pos = nn.Embedding(cfg["max_positions"], d)
            self.blocks = nn.ModuleList([Block() for _ in range(cfg["n_layers"])])
            self.final = make_norm()
            self.head = nn.Linear(d, cfg["vocab"], bias=False)
            if cfg["tie_embeddings"]:
                self.head.weight = self.emb.weight

    return Model()


def make_cfg(**over):
    cfg = dict(d_model=32, n_layers=3, n_heads=4, n_kv_heads=4, d_ff=64, vocab=50,
               max_positions=0, gated=False, bias=False, norm="rms", tie_embeddings=False)
    cfg.update(over)
    return cfg


@check("rmsnorm turns [1, 2, 3, 6] into [0.2828, 0.5657, 0.8485, 1.6971], ignores input size, and does not recentre")
def test_rmsnorm_values(sub):
    rmsnorm = has(sub, "rmsnorm")
    row = np.array([1.0, 2.0, 3.0, 6.0])
    # Wrong implementation caught here: subtracting the mean first. That is layer norm and
    # gives [-1.069, -0.5345, 0, 1.6036].
    close(rmsnorm(row, np.ones(4)), row / np.sqrt(12.5 + EPS), atol=1e-9, label="rmsnorm of [1, 2, 3, 6]")
    gain = np.array([1.0, 2.0, 0.5, -1.0])
    close(rmsnorm(row, gain), row / np.sqrt(12.5 + EPS) * gain, atol=1e-9, label="with a gain")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 3, 5)) + 2.0
    g = rng.normal(size=5)
    got = rmsnorm(x, g)
    shape(got, (2, 3, 5), label="output shape for a (2, 3, 5) input")
    want = np.array([[ref_rmsnorm_row(x[i, j], g) for j in range(3)] for i in range(2)])
    close(got, want, atol=1e-9, label="row-by-row reference on an input with mean near 2")
    x = rng.normal(size=(6, 8)) + 3.0
    ones = np.ones(8)
    close(rmsnorm(10.0 * x, ones), rmsnorm(x, ones), atol=1e-5, label="rmsnorm(10 x) against rmsnorm(x)")
    ms = np.mean(rmsnorm(x, ones) ** 2, axis=-1)
    close(ms, np.ones(6), atol=1e-5, label="mean square of each output row")
    mean = rmsnorm(x, ones).mean(axis=-1)
    if np.max(np.abs(mean)) < 0.5:
        fail(f"output rows have means {np.round(mean, 3)}, close to 0. The inputs have a mean near 3, "
             "so an output mean near 0 means you subtracted the mean, which is layer norm")


@check("rmsnorm_backward matches finite differences for dx and for dgain")
def test_rmsnorm_backward(sub):
    rmsnorm = has(sub, "rmsnorm")
    back = has(sub, "rmsnorm_backward")
    rng = np.random.default_rng(2)
    x = rng.normal(size=(3, 5)) + 1.0
    g = rng.normal(size=5) + 1.0
    c = rng.normal(size=(3, 5))
    dx, dg = back(c, x, g)
    shape(dx, (3, 5), label="dx")
    shape(dg, (5,), label="dgain")
    dx_fd = finite_difference(lambda a: float(np.sum(c * rmsnorm(a, g))), x.copy())
    dg_fd = finite_difference(lambda a: float(np.sum(c * rmsnorm(x, a))), g.copy())
    close(dx, dx_fd, atol=1e-6, rtol=1e-5, label="dx against finite differences")
    close(dg, dg_fd, atol=1e-6, rtol=1e-5, label="dgain against finite differences")
    xx = rng.normal(size=(2, 4, 5))
    dx3, dg3 = back(rng.normal(size=(2, 4, 5)), xx, g)
    shape(dg3, (5,), label="dgain for a (2, 4, 5) input, summed over every leading axis")


@check("gqa_attention equals full multi-head when groups equal heads, and multi-query when groups is 1")
def test_gqa_extremes(sub):
    gqa = has(sub, "gqa_attention")
    rng = np.random.default_rng(3)
    H, T, dh = 4, 5, 3
    q = rng.normal(size=(H, T, dh))
    k = rng.normal(size=(H, T, dh))
    v = rng.normal(size=(H, T, dh))
    out = gqa(q, k, v, n_groups=H)
    shape(out, (H, T, dh), label="output shape")
    close(out, ref_gqa(q, k, v, H), atol=1e-9, label="n_groups = H, against per-head loops")
    k1, v1 = k[:1], v[:1]
    close(gqa(q, k1, v1, n_groups=1), ref_gqa(q, k1, v1, 1), atol=1e-9, label="n_groups = 1, one key-value head")
    k_same = np.repeat(k1, H, axis=0)
    v_same = np.repeat(v1, H, axis=0)
    close(gqa(q, k1, v1, n_groups=1), gqa(q, k_same, v_same, n_groups=H), atol=1e-9,
          label="multi-query equals multi-head with one head copied H times")


@check("gqa_attention gives consecutive query heads the same key-value group, and masks the future")
def test_gqa_grouping(sub):
    gqa = has(sub, "gqa_attention")
    rng = np.random.default_rng(4)
    H, G, T, dh = 6, 2, 5, 4
    q = rng.normal(size=(H, T, dh))
    k = rng.normal(size=(G, T, dh))
    v = rng.normal(size=(G, T, dh))
    # Wrong implementation caught here: np.tile(k, (H // G, 1, 1)) puts the groups in interleaved
    # order, so head 1 reads group 1 instead of group 0.
    close(gqa(q, k, v, n_groups=G), ref_gqa(q, k, v, G), atol=1e-9,
          label="6 query heads, 2 groups (heads 0, 1, 2 read group 0)")
    v2 = v.copy()
    v2[:, -1] += 10.0
    close(gqa(q, k, v2, G)[:, :-1], gqa(q, k, v, G)[:, :-1], atol=1e-9,
          label="outputs before the last position when the last value changes")
    try:
        gqa(q, k, v, n_groups=4)
    except ValueError:
        return
    fail("n_groups=4 does not divide 6 heads, so gqa_attention must raise ValueError")


@check("kv_cache_bytes gives 131072 bytes per token for Llama 3 8B and 2 GiB for Llama 2 7B at 4096 tokens")
def test_kv_bytes(sub):
    kv = has(sub, "kv_cache_bytes")
    equal(int(kv(32, 8, 128, 1)), 131072, label="32 layers, 8 key-value heads, d_head 128, 1 token")
    equal(int(kv(32, 32, 128, 4096)), 2 * 2**30, label="32 layers, 32 key-value heads, 4096 tokens")
    equal(int(kv(2, 3, 4, 5, bytes_per_value=4)), 2 * 2 * 3 * 4 * 5 * 4, label="2 layers, 3 heads, d_head 4, 5 tokens, 4 bytes")
    equal(int(kv(32, 1, 128, 1)) * 8, int(kv(32, 8, 128, 1)), label="multi-query is one eighth of 8 groups")


@check("count_params gives GPT-2 small's 124439808 and Llama 3 8B's 8030261248")
def test_count_known(sub):
    count = has(sub, "count_params")
    gpt2 = dict(d_model=768, n_layers=12, n_heads=12, n_kv_heads=12, d_ff=3072, vocab=50257,
                max_positions=1024, gated=False, bias=True, norm="layer", tie_embeddings=True)
    equal(int(count(gpt2)), 124439808, label="GPT-2 small")
    equal(int(count(dict(gpt2, tie_embeddings=False))), 124439808 + 50257 * 768, label="GPT-2 small with the head untied")
    llama3 = dict(d_model=4096, n_layers=32, n_heads=32, n_kv_heads=8, d_ff=14336, vocab=128256,
                  max_positions=0, gated=True, bias=False, norm="rms", tie_embeddings=False)
    equal(int(count(llama3)), 8030261248, label="Llama 3 8B")
    try:
        count(dict(llama3, n_heads=30))
    except ValueError:
        return
    fail("n_heads=30 does not divide d_model=4096, so count_params must raise ValueError")


@check("count_params equals the number of parameters in a real torch module, for every option")
def test_count_torch(sub):
    count = has(sub, "count_params")
    variants = [
        make_cfg(),
        make_cfg(gated=True, d_ff=48),
        make_cfg(n_kv_heads=2),
        make_cfg(n_kv_heads=1, bias=True),
        make_cfg(norm="layer", bias=True, max_positions=16),
        make_cfg(tie_embeddings=True),
        make_cfg(gated=True, bias=True, norm="layer", n_kv_heads=2, tie_embeddings=True, n_layers=2),
    ]
    for cfg in variants:
        torch.manual_seed(0)
        want = sum(p.numel() for p in torch_model(cfg).parameters())
        equal(int(count(cfg)), want, label=f"config {cfg}")


@check("residual_stack: with zero weights a pre-norm layer is the identity and a post-norm layer is rmsnorm")
def test_stack_forward(sub):
    stack = has(sub, "residual_stack")
    rng = np.random.default_rng(5)
    d = 6
    x = rng.normal(size=(4, d)) + 1.0
    gain = 1.0 + 0.3 * rng.normal(size=d)
    layers = [{"W1": rng.normal(size=(d, 8)), "W2": np.zeros((8, d)), "gain": gain}]
    dout = rng.normal(size=(4, d))
    out_pre, grads = stack(x, layers, "pre", dout)
    # Wrong implementation caught here: a pre-norm layer that adds the sublayer to the normed
    # copy, x_next = rmsnorm(x) + f(rmsnorm(x)). The stream is then rescaled on every layer.
    close(out_pre, x, atol=1e-9, label="pre-norm output with W2 = 0")
    close(grads[0], dout, atol=1e-9, label="pre-norm gradient with W2 = 0")
    out_post, _ = stack(x, layers, "post", dout)
    close(out_post, np.array([ref_rmsnorm_row(r, gain) for r in x]), atol=1e-9, label="post-norm output with W2 = 0")
    equal(len(grads), 1, label="one gradient per layer")


@check("residual_stack gradients match finite differences for both wirings")
def test_stack_gradients(sub):
    stack = has(sub, "residual_stack")
    rng = np.random.default_rng(6)
    d = 5
    layers = [{"W1": rng.normal(size=(d, 7)) * 0.7, "W2": rng.normal(size=(7, d)) * 0.7,
               "gain": 1.0 + 0.2 * rng.normal(size=d)} for _ in range(3)]
    x = rng.normal(size=(3, d))
    c = rng.normal(size=(3, d))
    for kind in ("pre", "post"):
        out, grads = stack(x.copy(), layers, kind, c)
        equal(len(grads), 3, label=f"{kind}: number of gradients")
        fd0 = finite_difference(lambda a: float(np.sum(c * stack(a, layers, kind, c)[0])), x.copy())
        close(grads[0], fd0, atol=1e-5, rtol=1e-4, label=f"{kind}: gradient at the stack input")
        x1 = stack(x.copy(), layers[:1], kind, c)[0]
        fd1 = finite_difference(lambda a: float(np.sum(c * stack(a, layers[1:], kind, c)[0])), x1.copy())
        close(grads[1], fd1, atol=1e-5, rtol=1e-4, label=f"{kind}: gradient at the input of layer 1")


@check("at 32 layers the pre-norm stream and its gradient grow with depth, and the post-norm ones stay near 1")
def test_depth(sub):
    has(sub, "residual_stack")
    profile = has(sub, "depth_profile")
    pre = np.array([profile("pre", 32, 64, s) for s in range(5)])
    post = np.array([profile("post", 32, 64, s) for s in range(5)])
    pre_rms, pre_grad = pre.mean(axis=0)
    post_rms, post_grad = post.mean(axis=0)
    if not pre_rms > 3.5:
        fail(f"pre-norm output rms should be above 3.5 after 32 layers (it grows like the square root of the depth), got {pre_rms:.3f}")
    if not pre_grad > 4.0:
        fail(f"pre-norm gradient at the first layer should be above 4 times the output gradient, got {pre_grad:.3f}")
    if abs(post_rms - 1.0) > 0.01:
        fail(f"post-norm output rms should be 1 (the last op is rmsnorm with gain 1), got {post_rms:.3f}")
    if not post_grad < 2.0:
        fail(f"post-norm gradient at the first layer should stay below 2 times the output gradient, got {post_grad:.3f}")
