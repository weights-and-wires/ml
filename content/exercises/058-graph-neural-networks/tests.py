"""Checks for the graph neural network exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, finite_difference, has, shape

# The star from the lesson: node 1 is the centre, nodes 0, 2, and 3 are leaves.
STAR = np.array(
    [[0, 1, 0, 0], [1, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 0]],
    dtype=float,
)
H2 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]])
W2 = np.array([[1.0], [-1.0]])
A_HAT_STAR = np.array(
    [
        [0.5, 0.25 * np.sqrt(2.0), 0.0, 0.0],
        [0.25 * np.sqrt(2.0), 0.25, 0.25 * np.sqrt(2.0), 0.25 * np.sqrt(2.0)],
        [0.0, 0.25 * np.sqrt(2.0), 0.5, 0.0],
        [0.0, 0.25 * np.sqrt(2.0), 0.0, 0.5],
    ]
)


def _path(n):
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i, i + 1] = A[i + 1, i] = 1.0
    return A


@check("normalised_adjacency matches the hand example on the four-node star")
def test_norm_adj(sub):
    fn = has(sub, "normalised_adjacency")
    out = fn(STAR.copy())
    shape(out, (4, 4), label="A_hat")
    # Wrong implementation caught here: row normalisation D^-1 (A + I), or
    # D^-1/2 A D^-1/2 without self-loops. Both give different entries on a star,
    # where the centre and the leaves have different degrees.
    close(out, A_HAT_STAR, atol=1e-9, label="A_hat for the star")


@check("The normalised adjacency is symmetric, and sqrt(degree) is a fixed vector of it")
def test_norm_adj_properties(sub):
    fn = has(sub, "normalised_adjacency")
    rng = np.random.default_rng(0)
    upper = np.triu(rng.uniform(size=(9, 9)) < 0.35, 1)
    A = (upper | upper.T).astype(float)
    out = fn(A.copy())
    close(out, out.T, atol=1e-12, label="A_hat minus its transpose")
    d = A.sum(axis=1) + 1.0
    # A row-normalised matrix is not symmetric, and A_hat @ sqrt(d) = sqrt(d) fails for it.
    close(out @ np.sqrt(d), np.sqrt(d), atol=1e-9, label="A_hat @ sqrt(d + 1) against sqrt(d + 1)")


@check("gcn_layer matches the hand example, with and without the ReLU")
def test_gcn_forward(sub):
    fn = has(sub, "gcn_layer")
    out = fn(A_HAT_STAR, H2, W2)
    shape(out, (4, 1), label="layer output")
    close(out.ravel(), np.array([0.1464466, 0.8106602, 0.0, 0.6464466]), atol=1e-6, label="relu output")
    lin = fn(A_HAT_STAR, H2, W2, relu=False)
    close(lin.ravel(), np.array([0.1464466, 0.8106602, -0.3535534, 0.6464466]), atol=1e-6, label="linear output")


@check("gcn_layer_backward matches finite differences for the features and the weights")
def test_gcn_backward(sub):
    fn = has(sub, "gcn_layer")
    back = has(sub, "gcn_layer_backward")
    rng = np.random.default_rng(1)
    upper = np.triu(rng.uniform(size=(6, 6)) < 0.4, 1)
    A = (upper | upper.T).astype(float)
    A_hat = has(sub, "normalised_adjacency")(A)
    H = rng.normal(size=(6, 3))
    W = rng.normal(size=(3, 2))
    C = rng.normal(size=(6, 2))
    for relu in (True, False):
        dH, dW = back(A_hat, H, W, C, relu=relu)
        shape(dH, H.shape, label="dH")
        shape(dW, W.shape, label="dW")
        want_H = finite_difference(lambda h: float(np.sum(C * fn(A_hat, h, W, relu=relu))), H)
        want_W = finite_difference(lambda w: float(np.sum(C * fn(A_hat, H, w, relu=relu))), W)
        close(dH, want_H, rtol=1e-4, atol=1e-6, label=f"dH against finite differences (relu={relu})")
        close(dW, want_W, rtol=1e-4, atol=1e-6, label=f"dW against finite differences (relu={relu})")


@check("message_pass computes sum, mean, and max over neighbours, and leaves an isolated node at zero")
def test_message_pass(sub):
    fn = has(sub, "message_pass")
    h = H2[:, :1]
    close(fn(STAR, h, "sum").ravel(), np.array([0.0, 4.0, 0.0, 0.0]), atol=1e-9, label="sum")
    close(fn(STAR, h, "mean").ravel(), np.array([0.0, 4.0 / 3.0, 0.0, 0.0]), atol=1e-9, label="mean")
    close(fn(STAR, h, "max").ravel(), np.array([0.0, 2.0, 0.0, 0.0]), atol=1e-9, label="max")
    # Node 3 has no neighbours; node 1 touches nodes 0 and 2.
    A = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=float)
    X = np.array([[1.0, -5.0], [3.0, 2.0], [2.0, 7.0], [9.0, 9.0]])
    close(fn(A, X, "sum"), np.array([[3, 2], [3, 2], [3, 2], [0, 0]], dtype=float), atol=1e-9, label="sum, two columns")
    close(fn(A, X, "mean"), np.array([[3, 2], [1.5, 1], [3, 2], [0, 0]], dtype=float), atol=1e-9, label="mean, two columns")
    close(fn(A, X, "max"), np.array([[3, 2], [2, 7], [3, 2], [0, 0]], dtype=float), atol=1e-9, label="max, two columns")


@check("Permuting the nodes permutes the layer output the same way, and the readout does not change")
def test_permutation(sub):
    layer = has(sub, "gcn_layer")
    norm = has(sub, "normalised_adjacency")
    pool = has(sub, "readout")
    mp = has(sub, "message_pass")
    rng = np.random.default_rng(2)
    upper = np.triu(rng.uniform(size=(7, 7)) < 0.4, 1)
    A = (upper | upper.T).astype(float)
    H = rng.normal(size=(7, 3))
    W = rng.normal(size=(3, 4))
    perm = rng.permutation(7)
    P = np.eye(7)[perm]
    A_p, H_p = P @ A @ P.T, P @ H
    out = layer(norm(A), H, W)
    out_p = layer(norm(A_p), H_p, W)
    close(out_p, P @ out, atol=1e-10, label="layer(permuted graph) against permuted layer(graph)")
    for kind in ("sum", "mean", "max"):
        close(pool(out_p, kind), pool(out, kind), atol=1e-10, label=f"readout {kind} after permuting")
        close(mp(A_p, H_p, kind), P @ mp(A, H, kind), atol=1e-10, label=f"message_pass {kind} equivariance")
    shape(pool(out), (4,), label="readout shape")


@check("Stacking weightless layers shrinks the spread of node features")
def test_oversmoothing(sub):
    fn = has(sub, "oversmoothing_curve")
    rng = np.random.default_rng(3)
    # A path is bipartite. Without self-loops the features swap sides forever
    # and the spread never falls; with them, every layer averages and it does.
    A = _path(8)
    H = rng.normal(size=(8, 3))
    curve = np.asarray(fn(A, H, 40))
    shape(curve, (41,), label="curve")
    scale = 1.0 / np.sqrt(A.sum(axis=1) + 1.0)
    S = H * scale[:, None]
    almost(curve[0], float(np.mean(S.max(axis=0) - S.min(axis=0))), 1e-9, label="spread at depth 0")
    if np.any(np.diff(curve) > 1e-9):
        fail(f"the spread rose at some layer; first values are {np.round(curve[:6], 4)}")
    if curve[-1] > 0.1 * curve[0]:
        fail(f"after 40 layers the spread is {curve[-1]:.4f}, expected under 10% of the start {curve[0]:.4f}")
    # A hand value: a single step on the star with the features from the lesson.
    one = np.asarray(fn(STAR, H2, 1))
    almost(one[1], 0.35355, 1e-4, label="spread after one layer on the star")


@check("A two-layer GCN classifies a two-community graph from six labels, where raw features cannot")
def test_train(sub):
    make = has(sub, "two_community_graph")
    train = has(sub, "train_gcn")
    A, X, y = make()
    idx = np.array([0, 1, 2, 30, 31, 32])
    pred = np.asarray(train(A, X, y, idx, hidden=8, steps=200, lr=0.5, seed=0))
    shape(pred, y.shape, label="predictions")
    rest = np.setdiff1d(np.arange(len(y)), idx)
    acc = float(np.mean(pred[rest] == y[rest]))
    if acc < 0.9:
        fail(f"accuracy on the {len(rest)} unlabelled nodes is {acc:.3f}, expected at least 0.9")
