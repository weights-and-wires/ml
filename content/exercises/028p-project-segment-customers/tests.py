"""Checks for the customer segments and churn project."""

import numpy as np

from mlcheck import almost, check, close, fail, finite, has, shape


def _blobs(rng, per=60, gap=8.0):
    centres = gap * np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    X = np.concatenate([c + rng.normal(size=(per, 2)) for c in centres])
    return X, np.repeat(np.arange(4), per)


def _pipeline(sub, seed=0):
    X, segment, churn = sub.make_data(seed)
    rng = np.random.default_rng(seed)
    seg = sub.find_segments(X, rng)
    return X, segment, churn, seg, rng


@check("standardise gives every column mean 0 and standard deviation 1, and survives a constant column")
def test_standardise(sub):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3)) * np.array([1.0, 50.0, 0.01]) + np.array([5.0, -100.0, 3.0])
    Z = has(sub, "standardise")(X)
    shape(Z, (200, 3))
    close(Z.mean(axis=0), np.zeros(3), atol=1e-9, label="column means")
    close(Z.std(axis=0), np.ones(3), atol=1e-9, label="column standard deviations (ddof=0)")
    flat = has(sub, "standardise")(np.c_[X, np.full(200, 7.0)])
    finite(flat, label="output with a constant column")
    close(flat[:, 3], np.zeros(200), atol=1e-12, label="the constant column")


@check("explained_variance_ratio matches the eigenvalues of the covariance matrix")
def test_variance_ratio(sub):
    rng = np.random.default_rng(1)
    Q = np.linalg.qr(rng.normal(size=(4, 4)))[0]
    X = (rng.normal(size=(500, 4)) * np.array([3.0, 2.0, 1.0, 0.5])) @ Q + 10.0
    ratio = has(sub, "explained_variance_ratio")(X)
    want = np.sort(np.linalg.eigvalsh(np.cov(X.T, bias=True)))[::-1]
    close(ratio, want / want.sum(), atol=1e-9, label="ratios")


@check("pca_scores returns uncorrelated top-eigenvalue coordinates, and choose_components keeps only clear directions")
def test_pca_scores(sub):
    rng = np.random.default_rng(2)
    Q = np.linalg.qr(rng.normal(size=(5, 5)))[0]
    Z = (rng.normal(size=(400, 5)) * np.array([3.0, 2.0, 1.0, 0.5, 0.2])) @ Q
    S = has(sub, "pca_scores")(Z, 2)
    shape(S, (400, 2))
    cov = np.cov(S.T, bias=True)
    top = np.sort(np.linalg.eigvalsh(np.cov(Z.T, bias=True)))[::-1][:2]
    close(np.diag(cov), top, rtol=1e-6, label="variance of each score column")
    almost(cov[0, 1], 0.0, 1e-8, label="covariance between the two score columns")
    _choose_components_case(sub)


def _choose_components_case(sub):
    ratios = np.array([0.30, 0.20, 0.18, 0.10, 0.10, 0.05, 0.03, 0.02, 0.01, 0.01])
    got = has(sub, "choose_components")(ratios)
    if got != 3:
        fail(f"with ten columns the cut is 0.15, so 3 ratios qualify; got {got}")
    flat = np.full(8, 1 / 8)
    if has(sub, "choose_components")(flat) != 0:
        fail("eight equal ratios are all average, so none should be kept")


@check("kmeans_plusplus_init draws seeds from the data and spreads them over far-apart blobs")
def test_seeding(sub):
    rng = np.random.default_rng(3)
    X, truth = _blobs(rng, per=40, gap=40.0)
    seeds_used = set()
    for t in range(25):
        centers = has(sub, "kmeans_plusplus_init")(X, 4, np.random.default_rng(t))
        shape(centers, (4, 2))
        rows = [int(np.argmin(np.sum((X - c) ** 2, axis=1))) for c in centers]
        close(X[rows], centers, atol=1e-12, label="every seed is a row of X")
        blobs = {int(truth[r]) for r in rows}
        if len(blobs) != 4:
            fail(f"seeding {t} put seeds in only {len(blobs)} of 4 well-separated blobs")
        seeds_used.add(rows[0])
    if len(seeds_used) < 5:
        fail("the first seed looks fixed; it must be a random row")


@check("kmeans recovers four planted blobs, and choose_k finds the elbow at 4")
def test_kmeans_and_k(sub):
    rng = np.random.default_rng(4)
    X, truth = _blobs(rng)
    centers, labels, inertia = has(sub, "kmeans")(X, 4, np.random.default_rng(0))
    shape(centers, (4, 2))
    shape(labels, (len(X),))
    want = float(np.sum((X - centers[labels]) ** 2))
    almost(inertia, want, 1e-6 * want, label="returned inertia")
    if sub.purity(labels, truth) < 0.999:
        fail("kmeans with k = 4 should recover four blobs spaced 8 apart")
    k, inertias = has(sub, "choose_k")(X, range(1, 8), np.random.default_rng(0))
    if len(inertias) != 7:
        fail(f"inertias should hold one value per k, got {len(inertias)}")
    if not np.all(np.diff(inertias) < 1e-9):
        fail("best-of-restarts inertia should never rise as k grows")
    if k != 4:
        fail(f"expected k = 4, got {k}; inertias were {np.round(inertias, 0)}")


@check("purity ignores how the clusters are numbered and drops when clusters mix segments")
def test_purity(sub):
    truth = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    labels = np.array([5, 5, 5, 9, 9, 9, 1, 1, 1, 1])
    almost(has(sub, "purity")(labels, truth), 1.0, 1e-12, label="perfect clusters, renamed")
    mixed = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 0])
    almost(sub.purity(mixed, truth), 0.8, 1e-12, label="two rows sit in clusters led by another segment")
    almost(sub.purity(np.zeros(10, dtype=int), truth), 0.4, 1e-12, label="one cluster for everything")


@check("gmm_m_step reproduces the chapter's two-point refit and the plain mean for hard shares")
def test_m_step(sub):
    X = np.array([[0.0], [2.0]])
    r = np.array([[0.880797, 0.119203], [0.119203, 0.880797]])
    means, covs, weights = has(sub, "gmm_m_step")(X, r, reg_covar=0.0)
    close(weights, [0.5, 0.5], atol=1e-9, label="weights")
    close(means.ravel(), [0.2384, 1.7616], atol=1e-4, label="means")
    close(covs.ravel(), [0.42, 0.42], atol=1e-3, label="variances")
    rng = np.random.default_rng(5)
    X = rng.normal(size=(30, 3))
    hard = np.zeros((30, 2))
    hard[:12, 0] = 1.0
    hard[12:, 1] = 1.0
    means, covs, weights = sub.gmm_m_step(X, hard, reg_covar=0.0)
    close(means[0], X[:12].mean(axis=0), atol=1e-12, label="mean of group 0")
    close(covs[1], np.cov(X[12:].T, bias=True), atol=1e-12, label="covariance of group 1")
    close(weights, [0.4, 0.6], atol=1e-12, label="weights")


@check("fit_gmm never lowers the log-likelihood, and its shares are valid")
def test_fit_gmm(sub):
    rng = np.random.default_rng(6)
    X, truth = _blobs(rng, per=50)
    X = X * np.array([1.0, 0.5]) + 0.3 * X[:, ::-1]
    start = X[rng.choice(len(X), 4, replace=False)]
    means, covs, weights, resp, history = has(sub, "fit_gmm")(X, start, iters=30)
    if len(history) != 31:
        fail(f"history should hold 31 values (one per E-step), got {len(history)}")
    steps = np.diff(history)
    if steps.min() < -1e-6:
        fail(f"log-likelihood fell by {-steps.min():.3g} in one round; EM must not do that")
    if history[-1] <= history[0]:
        fail(f"log-likelihood did not improve: {history[0]:.2f} to {history[-1]:.2f}")
    shape(resp, (len(X), 4))
    close(resp.sum(axis=1), np.ones(len(X)), atol=1e-9, label="rows of shares")
    shape(covs, (4, 2, 2))
    close(weights.sum(), 1.0, atol=1e-9, label="weights")


@check("grow_tree and predict_tree learn a threshold, respect max_depth, and fit deeper trees more closely")
def test_tree(sub):
    rng = np.random.default_rng(7)
    X = rng.uniform(size=(300, 3))
    y = np.where(X[:, 1] > 0.4, 3.0, 1.0)
    tree = has(sub, "grow_tree")(X, y, max_depth=1, min_leaf=5)
    if tree["feature"] != 1:
        fail(f"the best first split is on feature 1, got feature {tree['feature']}")
    if not 0.35 < tree["threshold"] < 0.45:
        fail(f"threshold should sit near 0.4, got {tree['threshold']:.3f}")
    close(sub.predict_tree(tree, X), y, atol=1e-9, label="depth-1 predictions on a step function")
    stump = sub.grow_tree(X, y, max_depth=0)
    close(sub.predict_tree(stump, X), np.full(300, y.mean()), atol=1e-9, label="depth 0 predicts the mean")

    def depth(node):
        return 0 if node["feature"] is None else 1 + max(depth(node["left"]), depth(node["right"]))

    z = np.sin(6 * X[:, 0]) + X[:, 2] + 0.3 * rng.normal(size=300)
    errors = []
    for d in (1, 2, 4, 6):
        t = sub.grow_tree(X, z, max_depth=d, min_leaf=5)
        if depth(t) > d:
            fail(f"max_depth={d} produced a tree of depth {depth(t)}")
        errors.append(float(np.mean((sub.predict_tree(t, X) - z) ** 2)))
    if not np.all(np.diff(errors) < 0):
        fail(f"training error should fall as depth grows, got {np.round(errors, 4)}")


@check("bagged trees: bootstrap masks, averaged predictions, and an out-of-bag error that is honest")
def test_bagging(sub):
    rng = np.random.default_rng(8)
    n = 400
    X = rng.uniform(-1, 1, size=(n, 3))
    f = lambda A: (A[:, 0] > 0.0) * 2.0 + A[:, 1]
    y = f(X) + 0.5 * rng.normal(size=n)
    trees, in_bag = has(sub, "fit_bagged_trees")(X, y, 60, 4, np.random.default_rng(0))
    if len(trees) != 60:
        fail(f"expected 60 trees, got {len(trees)}")
    shape(in_bag, (60, n))
    almost(in_bag.mean(), 0.632, 0.03, label="fraction of rows in each bootstrap sample")
    manual = np.mean([sub.predict_tree(t, X[:25]) for t in trees], axis=0)
    close(has(sub, "bagged_predict")(trees, X[:25]), manual, atol=1e-12, label="bagged_predict")
    oob = has(sub, "oob_error")(X, y, trees, in_bag)
    train = float(np.mean((sub.bagged_predict(trees, X) - y) ** 2))
    Xt = rng.uniform(-1, 1, size=(2000, 3))
    yt = f(Xt) + 0.5 * rng.normal(size=2000)
    test = float(np.mean((sub.bagged_predict(trees, Xt) - yt) ** 2))
    if oob <= train:
        fail(f"out-of-bag error {oob:.4f} should exceed training error {train:.4f}; are bagged rows predicted by trees that saw them?")
    almost(oob, test, 0.06, label="out-of-bag error against held-out error")


@check("end to end: the pipeline keeps 3 components, picks k = 4, and recovers the planted segments")
def test_segments(sub):
    for seed in (0, 1):
        X, segment, _, seg, _ = _pipeline(sub, seed)
        if seg["n_components"] != 3:
            fail(f"seed {seed}: kept {seg['n_components']} components, expected the 3 planted ones")
        if seg["k"] != 4:
            fail(f"seed {seed}: chose k = {seg['k']}, expected 4")
        km = sub.purity(seg["labels"], segment)
        em = sub.purity(np.argmax(seg["resp"], axis=1), segment)
        if km < 0.88:
            fail(f"seed {seed}: k-means purity {km:.3f}, expected at least 0.88 (did you standardise first?)")
        if em < 0.90:
            fail(f"seed {seed}: mixture purity {em:.3f}, expected at least 0.90")
        steps = np.diff(seg["history"])
        if steps.min() < -1e-6:
            fail(f"seed {seed}: log-likelihood fell by {-steps.min():.3g} during EM")
        close(seg["resp"].sum(axis=1), np.ones(len(X)), atol=1e-9, label="shares")


@check("end to end: a bagged ensemble beats one tree and the base rate on held-out churn")
def test_churn(sub):
    X, segment, churn, seg, rng = _pipeline(sub, 0)
    F = np.concatenate([seg["Z"], seg["resp"]], axis=1)
    tr, te = slice(0, 1000), slice(1000, None)
    one = sub.grow_tree(F[tr], churn[tr], max_depth=6)
    trees, in_bag = sub.fit_bagged_trees(F[tr], churn[tr], 100, 6, rng)
    p_tree = sub.predict_tree(one, F[te])
    p_forest = sub.bagged_predict(trees, F[te])
    base = sub.brier(churn[te], np.full(500, churn[tr].mean()))
    b_tree = sub.brier(churn[te], p_tree)
    b_forest = sub.brier(churn[te], p_forest)
    if b_forest > base - 0.006:
        fail(f"forest Brier {b_forest:.4f} should beat the base rate {base:.4f} by at least 0.006")
    if b_forest > b_tree - 0.008:
        fail(f"forest Brier {b_forest:.4f} should beat one deep tree {b_tree:.4f} by at least 0.008")
    oob = sub.oob_error(F[tr], churn[tr], trees, in_bag)
    almost(oob, b_forest, 0.03, label="out-of-bag error against held-out Brier score")
