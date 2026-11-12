"""Checks for the masked diffusion exercise.

The check that would accept a wrong loss fails it with the expected value and
the value it returns:

- loss_over_all_positions averages the cross-entropy over all L positions, as
  an ordinary language-model loss does, and drops the mask and the 1/t weight.
  The hand-computed loss below has two masked and two unmasked positions with
  different probabilities, so the average over all four is not 1.0397.

Other wrong implementations the same checks reject: a loss that keeps the mask
but drops the 1/t weight, a sampler that commits the least confident
predictions and remasks the most confident ones, a sampler that overwrites
tokens that are already unmasked, and an unmasking schedule that rounds up.
"""

import numpy as np

from mlcheck import almost, check, close, equal, fail, finite_difference, has, shape

MASK = 4
# Table of logits for a toy 8-token model over a vocabulary of 4 tokens.
TABLE = np.array(
    [
        [3, 0, 0, 0],
        [0, 0, 5, 0],
        [0, 2, 0, 0],
        [0, 0, 0, 4],
        [1, 0, 0, 0],
        [0, 0, 2.5, 0],
        [0, 1.5, 0, 0],
        [0, 0, 0, 0.5],
    ],
    dtype=float,
)
# Probabilities for the hand-computed loss: L = 4, t = 0.5, masked = {1, 3}.
HAND_P = np.array([[0.2, 0.3, 0.5], [0.5, 0.25, 0.25], [0.1, 0.8, 0.1], [0.25, 0.5, 0.25]])
HAND_TARGETS = np.array([2, 0, 1, 0])
HAND_MASK = np.array([False, True, False, True])
HAND_LOSS = (np.log(2) + np.log(4)) / (0.5 * 4)  # 1.0397207708


def loss_over_all_positions(logits, targets, mask, t):
    """Wrong: ordinary mean cross-entropy over every position."""
    z = logits - logits.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-logp[np.arange(len(targets)), targets].mean())


@check("forward_mask keeps the unmasked tokens and writes the mask id at masked ones")
def test_forward_mask_values(sub):
    forward_mask = has(sub, "forward_mask")
    tokens = np.array([2, 0, 1, 0, 2, 1])
    noisy, mask = forward_mask(tokens, 0.5, np.random.default_rng(0), MASK)
    shape(noisy, (6,), "noisy")
    shape(mask, (6,), "mask")
    equal(noisy[mask], np.full(mask.sum(), MASK), "tokens at masked positions")
    equal(noisy[~mask], tokens[~mask], "tokens at unmasked positions")
    for t, want in ((0.0, 0), (1.0, 6)):
        _, m = forward_mask(tokens, t, np.random.default_rng(1), MASK)
        if int(np.sum(m)) != want:
            fail(f"t={t}: expected {want} masked positions, got {int(np.sum(m))}")


@check("forward_mask masks each position independently at rate t")
def test_forward_mask_rate(sub):
    forward_mask = has(sub, "forward_mask")
    tokens = np.zeros(20000, dtype=int)
    for t in (0.15, 0.7):
        _, mask = forward_mask(tokens, t, np.random.default_rng(3), MASK)
        almost(float(np.mean(mask)), t, 0.015, f"masked fraction at t={t}")
    _, mask = forward_mask(tokens, 0.5, np.random.default_rng(4), MASK)
    both = float(np.mean(mask[:-1] & mask[1:]))
    almost(both, 0.25, 0.015, "fraction of neighbouring pairs both masked (independence gives 0.25)")


@check("masked_diffusion_loss matches the hand-computed value")
def test_loss_hand(sub):
    loss = has(sub, "masked_diffusion_loss")
    got = loss(np.log(HAND_P), HAND_TARGETS, HAND_MASK, 0.5)
    almost(got, HAND_LOSS, 1e-9, "loss on the worked example (expected (ln 2 + ln 4) / 2)")
    # Wrong implementation named in the module comment: mean over all positions.
    wrong = loss_over_all_positions(np.log(HAND_P), HAND_TARGETS, HAND_MASK, 0.5)
    if abs(wrong - HAND_LOSS) < 1e-3:
        fail("the test case no longer separates the wrong loss from the right one")


@check("The loss scales as 1/t and ignores unmasked positions")
def test_loss_weight(sub):
    loss = has(sub, "masked_diffusion_loss")
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(6, 5))
    targets = rng.integers(0, 5, size=6)
    mask = np.array([True, False, True, True, False, False])
    a = loss(logits, targets, mask, 0.5)
    b = loss(logits, targets, mask, 0.25)
    almost(b / a, 2.0, 1e-9, "loss(t=0.25) / loss(t=0.5), expected 2 from the 1/t weight")
    other = logits.copy()
    other[~mask] = rng.normal(size=(3, 5))
    almost(loss(other, targets, mask, 0.5), a, 1e-12, "loss after changing logits at unmasked rows")
    almost(loss(logits, targets, np.zeros(6, dtype=bool), 0.5), 0.0, 1e-12, "loss with nothing masked")


@check("masked_diffusion_loss_grad matches finite differences")
def test_loss_grad(sub):
    loss = has(sub, "masked_diffusion_loss")
    grad = has(sub, "masked_diffusion_loss_grad")
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(5, 4))
    targets = rng.integers(0, 4, size=5)
    mask = np.array([True, False, True, True, False])
    t = 0.6
    got = grad(logits, targets, mask, t)
    shape(got, (5, 4), "gradient")
    want = finite_difference(lambda z: loss(z.reshape(5, 4), targets, mask, t), logits.ravel())
    close(got.ravel(), want, rtol=1e-4, atol=1e-7, label="gradient with respect to logits")
    close(got[~mask], np.zeros((2, 4)), label="gradient rows at unmasked positions")


@check("num_unmasked_after follows floor(L k / N)")
def test_schedule(sub):
    f = has(sub, "num_unmasked_after")
    equal([f(8, k, 4) for k in range(1, 5)], [2, 4, 6, 8], "L=8, N=4")
    equal([f(10, k, 4) for k in range(1, 5)], [2, 5, 7, 10], "L=10, N=4 (floor of 2.5, 5, 7.5, 10)")
    equal([f(5, k, 3) for k in range(1, 4)], [1, 3, 5], "L=5, N=3")


@check("One step fills the most confident predictions and leaves the rest masked")
def test_step(sub):
    step = has(sub, "low_confidence_remask_step")
    x = np.full(8, MASK)
    out = step(x, TABLE, MASK, 2)
    want = np.full(8, MASK)
    want[1], want[3] = 2, 3  # confidences 0.98 and 0.948 are the two largest
    equal(out, want, "tokens after step 1 (positions 1 and 3 filled)")
    # Already-unmasked tokens stay, even when the model now predicts something else.
    x2 = np.full(8, MASK)
    x2[1] = 0
    out2 = step(x2, TABLE, MASK, 3)
    want2 = np.full(8, MASK)
    want2[1], want2[3], want2[0] = 0, 3, 0
    equal(out2, want2, "tokens when position 1 was already unmasked as 0")
    equal(x, np.full(8, MASK), "the input array after the call (it must not change)")


@check("sample reproduces the expected sequence and unmasking order")
def test_sample(sub):
    sample = has(sub, "sample")
    x, history = sample(lambda z: TABLE, 8, 4, MASK)
    equal(x, np.array([0, 2, 1, 3, 0, 2, 1, 3]), "final tokens")
    if len(history) != 5:
        fail(f"expected 5 snapshots, got {len(history)}")
    counts = [int(np.sum(h != MASK)) for h in history]
    equal(counts, [0, 2, 4, 6, 8], "unmasked count at each snapshot")
    equal(history[1], np.array([MASK, 2, MASK, 3, MASK, MASK, MASK, MASK]), "after step 1")
    equal(history[2], np.array([0, 2, MASK, 3, MASK, 2, MASK, MASK]), "after step 2")
    equal(history[3], np.array([0, 2, 1, 3, MASK, 2, 1, MASK]), "after step 3")
