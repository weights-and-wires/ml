"""Checks for the optimiser-scaling-rules exercise.

Plausible wrong implementations these checks catch:

* adam_lr_for_width that scales the hidden and output rates by 1/sqrt(width),
  the SGD-style rule, instead of 1/width. The first check compares exact
  values, and the activation check sees the update size grow like sqrt(width).
* newton_schulz that skips the division by the Frobenius norm. The iteration
  then diverges for any matrix whose largest singular value exceeds about 1.5,
  and the scale-invariance check fails.
"""

import numpy as np

from mlcheck import almost, check, close, fail, has

BASE = 32
GRID = 2.0 ** np.arange(-12, -1)


@check("adam_lr_for_width keeps the input rate and divides hidden and output rates by the width ratio")
def test_lr_rule(sub):
    f = has(sub, "adam_lr_for_width")
    almost(f(1e-3, 32, 512, "input"), 1e-3, tol=1e-15, label="input layer at 16x the width")
    for layer in ("hidden", "output"):
        almost(f(1e-3, 32, 512, layer), 1e-3 / 16.0, tol=1e-12,
               label=f"{layer} layer at 16x the width (expected 1/16, not 1/4)")
        almost(f(2e-3, 64, 64, layer), 2e-3, tol=1e-15, label=f"{layer} layer at the base width")
        almost(f(1e-3, 128, 32, layer), 4e-3, tol=1e-12, label=f"{layer} layer at a quarter of the width")
    try:
        f(1e-3, 32, 64, "embedding")
    except ValueError:
        return
    fail("adam_lr_for_width should raise ValueError for the layer name 'embedding'")


@check("mup_init returns the right shapes, std 1/sqrt(fan_in) inside, and std 1/fan_in at the output")
def test_mup_init(sub):
    f = has(sub, "mup_init")
    widths = [16, 256, 256, 1]
    Ws = f(widths, np.random.default_rng(0))
    got = [tuple(W.shape) for W in Ws]
    want = [(256, 16), (256, 256), (1, 256)]
    if got != want:
        fail(f"shapes: expected {want}, got {got}")
    almost(Ws[0].std(), 1 / np.sqrt(16), tol=0.02, label="input std (expected 0.25)")
    almost(Ws[1].std(), 1 / np.sqrt(256), tol=0.003, label="hidden std (expected 0.0625)")
    almost(Ws[2].std(), 1 / 256, tol=0.0015, label="output std (expected 1/256 = 0.0039)")
    again = f(widths, np.random.default_rng(0))
    close(again[1], Ws[1], label="same seed, same hidden matrix")
    wide = f([16, 1024, 1024, 1], np.random.default_rng(0))
    ratio = Ws[2].std() / wide[2].std()
    almost(ratio, 4.0, tol=1.2, label="output std at width 256 over width 1024 (expected about 4)")


@check("After one Adam step the hidden activations move by a width-independent amount under muP")
def test_activation_update_mup(sub):
    f = has(sub, "activation_update_size")
    sizes = [np.mean([f(n, "mup", seed=s) for s in range(4)]) for n in (32, 128, 512)]
    ratio = sizes[-1] / sizes[0]
    if not 0.5 < ratio < 2.0:
        fail(f"muP update size at widths 32, 128, 512 was {np.round(sizes, 5)}; "
             f"the ratio 512/32 was {ratio:.2f}, expected between 0.5 and 2")


@check("Under standard parameterisation the same update grows roughly in proportion to the width")
def test_activation_update_sp(sub):
    f = has(sub, "activation_update_size")
    sizes = [np.mean([f(n, "sp", seed=s) for s in range(4)]) for n in (32, 128, 512)]
    ratio = sizes[-1] / sizes[0]
    if not 8.0 < ratio < 20.0:
        fail(f"SP update size at widths 32, 128, 512 was {np.round(sizes, 5)}; "
             f"the ratio 512/32 was {ratio:.2f}, expected between 8 and 20 (16 would be exactly linear)")
    if not sizes[0] < sizes[1] < sizes[2]:
        fail(f"SP update sizes should rise with width, got {np.round(sizes, 5)}")


@check("The best learning rate stays put from width 32 to 512 under muP and falls under standard parameterisation")
def test_transfer(sub):
    f = has(sub, "best_lr")
    mup = [f(n, "mup", GRID) for n in (32, 512)]
    sp = [f(n, "sp", GRID) for n in (32, 512)]
    r_mup = mup[0] / mup[1]
    r_sp = sp[0] / sp[1]
    if not 0.5 <= r_mup <= 2.0:
        fail(f"muP best rates at widths 32 and 512 were {mup}; ratio {r_mup:.2f}, expected within 0.5 to 2")
    if r_sp < 4.0:
        fail(f"SP best rates at widths 32 and 512 were {sp}; ratio {r_sp:.2f}, expected at least 4")


@check("newton_schulz puts the singular values of a tall, a wide, and a hand-made matrix inside a band near 1")
def test_ns_band(sub):
    f = has(sub, "newton_schulz")
    rng = np.random.default_rng(0)
    for shape in [(64, 32), (32, 64), (50, 40)]:
        G = rng.normal(size=shape)
        out = f(G, 5)
        if out.shape != shape:
            fail(f"shape: expected {shape}, got {out.shape}")
        sv = np.linalg.svd(out, compute_uv=False)
        if sv.min() < 0.6 or sv.max() > 1.25:
            fail(f"singular values for shape {shape} should lie in [0.6, 1.25], got min {sv.min():.3f} max {sv.max():.3f}")
    out = f(np.diag([3.0, 1.0]), 5)
    close(np.diag(out), np.array([0.7530, 1.1337]), atol=5e-3, label="hand example diag(3, 1) after 5 steps")
    close(out - np.diag(np.diag(out)), np.zeros((2, 2)), atol=1e-9, label="off-diagonal entries of the hand example")


@check("newton_schulz divides by the Frobenius norm first, so scaling G changes nothing and huge G stays finite")
def test_ns_normalised(sub):
    f = has(sub, "newton_schulz")
    G = np.random.default_rng(1).normal(size=(40, 20))
    base = f(G, 5)
    for scale in (1e-3, 50.0, 1e4):
        out = f(scale * G, 5)
        if not np.all(np.isfinite(out)):
            fail(f"newton_schulz(G * {scale}) returned non-finite values; divide G by its Frobenius norm first")
        close(out, base, atol=1e-4, label=f"newton_schulz(G * {scale}) against newton_schulz(G)")
    U, _, Vt = np.linalg.svd(G, full_matrices=False)
    gap = np.linalg.norm(base - U @ Vt, 2)
    if gap > 0.5:
        fail(f"the output should stay close to the orthogonal factor U V^T; spectral-norm gap {gap:.3f}, expected under 0.5")


@check("muon_step accumulates momentum, orthogonalises, and bounds the update's spectral norm by about lr")
def test_muon(sub):
    f = has(sub, "muon_step")
    ns = has(sub, "newton_schulz")
    rng = np.random.default_rng(2)
    W = rng.normal(size=(24, 12))
    G = rng.normal(size=(24, 12))
    M = rng.normal(size=(24, 12))
    W1, M1 = f(W, G, M, 0.02, 0.9)
    close(M1, 0.9 * M + G, label="momentum buffer")
    close(W1, W - 0.02 * ns(0.9 * M + G), atol=1e-9, label="weights after one step")
    for scale in (1.0, 1e4):
        W2, _ = f(W, scale * G, np.zeros_like(W), 0.02, 0.9)
        norm = np.linalg.norm(W2 - W, 2)
        if norm > 0.02 * 1.25:
            fail(f"gradient scale {scale}: spectral norm of the update was {norm:.4f}, expected at most {0.02 * 1.25:.4f}")
        if norm < 0.02 * 0.6:
            fail(f"gradient scale {scale}: spectral norm of the update was {norm:.4f}, expected at least {0.02 * 0.6:.4f}")
