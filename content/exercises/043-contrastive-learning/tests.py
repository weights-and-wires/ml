"""Checks for the contrastive-learning exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, finite_difference, has, shape

# The hand example from the lesson: three unit rows per view, tau = 0.5.
Z1 = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
Z2 = np.array([[0.8, 0.6], [0.6, 0.8], [0.0, 1.0]])
TAU = 0.5
# Same directions as Z1, but with lengths 2, 3, and 0.5.
Z1_BIG = np.array([[2.0, 0.0], [0.0, 3.0], [0.3, 0.4]])
INFO_NCE_12 = 1.0437858  # row softmax over view 2, positives on the diagonal
INFO_NCE_21 = 1.0854929  # the same matrix read down its columns
CLIP_12 = 1.0646393


@check("l2_normalize gives every row length 1 and keeps its direction: [3, 4] becomes [0.6, 0.8]")
def test_normalize(sub):
    fn = has(sub, "l2_normalize")
    got = fn(np.array([[3.0, 4.0], [0.0, -2.0]]))
    shape(got, (2, 2), label="l2_normalize output")
    close(got, np.array([[0.6, 0.8], [0.0, -1.0]]), atol=1e-12, label="normalised rows")


# Plausible wrong implementation caught here: InfoNCE that skips the L2 normalisation, or
# that takes the softmax over columns instead of rows. Z1_BIG has rows of length 2, 3, and
# 0.5, so a missing normalisation changes the value. The positives sit off the row maximum,
# so info_nce(Z1, Z2) and info_nce(Z2, Z1) differ, and a column softmax returns the other one.
@check("info_nce matches the hand example, and reading rows versus columns gives different values")
def test_info_nce_hand(sub):
    fn = has(sub, "info_nce")
    close(fn(Z1, Z2, TAU), INFO_NCE_12, atol=1e-6, label="info_nce(Z1, Z2) on unit rows")
    close(fn(Z1_BIG, Z2, TAU), INFO_NCE_12, atol=1e-6, label="info_nce with unnormalised rows of the same directions")
    close(fn(Z2, Z1, TAU), INFO_NCE_21, atol=1e-6, label="info_nce(Z2, Z1), the column reading")


@check("info_nce equals log N when every embedding is the same, and has a closed form for orthogonal pairs")
def test_info_nce_limits(sub):
    fn = has(sub, "info_nce")
    rng = np.random.default_rng(0)
    for n in (2, 5, 8):
        v = rng.normal(size=(1, 4))
        z = np.repeat(v, n, axis=0)
        close(fn(z, 3.0 * z, 0.1), np.log(n), atol=1e-9, label=f"loss for {n} identical embeddings")
    eye = np.eye(4)
    want = np.log(1 + 3 * np.exp(-1 / 0.1))
    close(fn(eye, eye, 0.1), want, atol=1e-9, label="loss for four orthogonal perfect pairs, tau = 0.1")
    big = fn(10.0 * eye, 10.0 * eye, 0.001)
    if not np.isfinite(big):
        fail(f"info_nce overflowed at tau = 0.001: got {big}, expected a finite number near 0")


@check("info_nce_grad matches finite differences for both views")
def test_grad_fd(sub):
    fn = has(sub, "info_nce_grad")
    loss = has(sub, "info_nce")
    rng = np.random.default_rng(1)
    z1 = rng.normal(size=(4, 3)) * 1.7
    z2 = rng.normal(size=(4, 3)) * 0.6
    dz1, dz2 = fn(z1, z2, 0.4)
    shape(dz1, z1.shape, label="dz1")
    shape(dz2, z2.shape, label="dz2")
    want1 = finite_difference(lambda a: loss(a, z2, 0.4), z1)
    want2 = finite_difference(lambda b: loss(z1, b, 0.4), z2)
    close(dz1, want1, rtol=1e-5, atol=1e-7, label="dz1 against finite differences")
    close(dz2, want2, rtol=1e-5, atol=1e-7, label="dz2 against finite differences")


@check("The gradient of each row is perpendicular to the row, because only its direction matters")
def test_grad_perp(sub):
    fn = has(sub, "info_nce_grad")
    rng = np.random.default_rng(2)
    z1 = rng.normal(size=(5, 4)) * 3.0
    z2 = rng.normal(size=(5, 4)) * 0.2
    dz1, dz2 = fn(z1, z2, 0.3)
    close(np.sum(dz1 * z1, axis=1), np.zeros(5), atol=1e-10, label="row dot product of dz1 and z1")
    close(np.sum(dz2 * z2, axis=1), np.zeros(5), atol=1e-10, label="row dot product of dz2 and z2")


@check("clip_loss is symmetric and equals the mean of the row and column cross-entropies")
def test_clip(sub):
    fn = has(sub, "clip_loss")
    close(fn(Z1, Z2, TAU), CLIP_12, atol=1e-6, label="clip_loss on the hand example")
    close(fn(Z1, Z2, TAU), fn(Z2, Z1, TAU), atol=1e-12, label="clip_loss after swapping its arguments")
    rng = np.random.default_rng(3)
    img, txt = rng.normal(size=(6, 5)), rng.normal(size=(6, 5))
    close(fn(img, txt, 0.2), fn(txt, img, 0.2), atol=1e-12, label="clip_loss on random data, arguments swapped")
    row = has(sub, "info_nce")(img, txt, 0.2)
    col = has(sub, "info_nce")(txt, img, 0.2)
    close(fn(img, txt, 0.2), 0.5 * (row + col), atol=1e-12, label="clip_loss against the mean of two info_nce calls")


@check("alignment is 0 for matching pairs, and uniformity is 0 when collapsed and -8 for opposite points")
def test_metrics(sub):
    align = has(sub, "alignment")
    unif = has(sub, "uniformity")
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([[0.0, 5.0], [0.0, 0.1]])
    close(align(a, b), 1.0, atol=1e-12, label="alignment: squared distances 2 and 0, mean 1")
    close(align(a, 7 * a), 0.0, atol=1e-12, label="alignment of a view with a scaled copy")
    same = np.tile(np.array([[0.3, -0.4]]), (5, 1))
    close(unif(same), 0.0, atol=1e-12, label="uniformity of five identical points")
    close(unif(np.array([[1.0, 0.0], [-2.0, 0.0]])), -8.0, atol=1e-10, label="uniformity of two opposite points, t = 2")
    angles = np.deg2rad([0.0, 120.0, 240.0])
    tri = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    close(unif(tri), -6.0, atol=1e-10, label="uniformity of three points 120 degrees apart, t = 2")
    almost(unif(tri, t=1.0), -3.0, 1e-10, label="uniformity at t = 1")


@check("knn_accuracy votes by cosine similarity: k = 1 follows the nearest row, k = 3 follows the majority")
def test_knn(sub):
    fn = has(sub, "knn_accuracy")
    deg = np.deg2rad([5.0, 30.0, -35.0, 90.0])
    train = 4.0 * np.stack([np.cos(deg), np.sin(deg)], axis=1)
    ytr = np.array([1, 0, 0, 1])
    test = np.array([[1.0, 0.0]])
    yte = np.array([0])
    close(fn(train, ytr, test, yte, k=1), 0.0, atol=1e-12, label="accuracy with k = 1")
    close(fn(train, ytr, test, yte, k=3), 1.0, atol=1e-12, label="accuracy with k = 3")


def _make_data(rng, n_per, classes=4, d=12):
    means = np.array([[3, 3], [3, -3], [-3, 3], [-3, -3]], float)
    y = np.repeat(np.arange(classes), n_per)
    x = np.zeros((len(y), d))
    x[:, :2] = means[y] + 0.5 * rng.normal(size=(len(y), 2))
    x[:, 2:] = 3.0 * rng.normal(size=(len(y), d - 2))
    return x, y


@check("A linear encoder trained for 100 steps recovers the planted clusters; a collapsed encoder stays near chance")
def test_train(sub):
    train = has(sub, "train_encoder")
    knn = has(sub, "knn_accuracy")
    d = 12
    rng = np.random.default_rng(0)
    xtr, ytr = _make_data(rng, 50)
    xte, yte = _make_data(rng, 25)

    def make_views(r):
        x = xtr[r.choice(len(xtr), 64, replace=False)]
        out = []
        for _ in range(2):
            v = x.copy()
            v[:, :2] += 0.3 * r.normal(size=(len(x), 2))
            v[:, 2:] = 3.0 * r.normal(size=(len(x), d - 2))
            out.append(v)
        return out[0], out[1]

    w = train(make_views, d, 2, 0.5, 100, 0.2, 1)
    shape(w, (d, 2), label="encoder weights")
    acc = knn(xtr @ w, ytr, xte @ w, yte, k=5)
    if acc < 0.9:
        fail(f"k-NN accuracy on the trained embedding is {acc:.3f}, expected at least 0.9 (chance is 0.25)")
    collapsed = knn(np.ones((len(ytr), 2)), ytr, np.ones((len(yte), 2)), yte, k=5)
    if collapsed > 0.4:
        fail(f"k-NN accuracy on a constant embedding is {collapsed:.3f}, expected near chance (0.25)")
