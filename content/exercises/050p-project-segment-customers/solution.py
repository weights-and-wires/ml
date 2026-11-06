"""Reference solution: find customer segments, then predict who leaves.

The pipeline: standardise the columns, keep the principal components that
carry real structure, cluster with k-means++ and pick k, soften the clusters
with a Gaussian mixture fitted by EM, and predict churn with a bagged ensemble
of shallow regression trees scored out of bag.
"""

from __future__ import annotations

import numpy as np

COLUMNS = [
    "sessions_per_week", "minutes_per_session", "monthly_spend", "support_tickets",
    "days_since_login", "feature_count", "referrals",
    "support_chats", "ad_clicks", "app_version_age",
]


def make_data(seed: int = 0, n: int = 1500):
    """Build the customer table. Written for you; do not change it.

    Planted pattern:
      * Four segments with sizes 35, 30, 20, and 15 percent. Each customer has
        a hidden 3-number "taste" vector near its segment's centre.
      * Columns 0 to 7 are noisy mixes of that taste vector, so the segments
        live in a 3-dimensional subspace of those eight columns.
      * Columns 8 and 9 are pure noise with large units that carry no structure.
      * Every column has its own offset and scale, so units differ wildly.
      * Churn is 1 with probability 0.05, plus a bump by segment, plus 0.45
        when column 3 is above 0.2 and column 1 is below -0.2 (after
        standardising): many support tickets and short sessions. That
        two-column threshold rule is something a tree finds and a straight
        line captures poorly.

    Returns:
        X: raw features, shape (n, 10).
        segment: planted segment of each row, shape (n,), values 0 to 3.
        churn: 0.0 or 1.0 per row, shape (n,).
    """
    rng = np.random.default_rng(seed)
    segment = rng.choice(4, size=n, p=[0.35, 0.30, 0.20, 0.15])
    centres = 1.45 * np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
    spread = np.array([1.0, 0.8, 1.0, 0.9])[segment][:, None]
    taste = centres[segment] + spread * rng.normal(size=(n, 3))
    loadings = np.linalg.qr(np.random.default_rng(7).normal(size=(8, 3)))[0].T * 2.0  # (3, 8), same for every seed
    signal = taste @ loadings + 0.5 * rng.normal(size=(n, 8))
    noise = rng.normal(size=(n, 2))
    scales = np.array([0.5, 3.0, 12.0, 80.0, 2.0, 150.0, 7.0, 0.8, 25.0, 60.0])
    offsets = np.array([4.0, 20.0, 60.0, 1.0, 9.0, 30.0, 2.0, 5.0, 50.0, 200.0])
    X = np.concatenate([signal, noise], axis=1) * scales + offsets
    Z = (X - X.mean(axis=0)) / X.std(axis=0)
    risky = (Z[:, 3] > 0.2) & (Z[:, 1] < -0.2)
    p = 0.05 + np.array([0.0, 0.10, 0.04, 0.18])[segment] + 0.45 * risky
    churn = (rng.uniform(size=n) < p).astype(float)
    return X, segment, churn


# ---------------------------------------------------------------- PCA


def standardise(X: np.ndarray) -> np.ndarray:
    """Centre each column and divide by its standard deviation (ddof=0).

    Returns an array of shape (n, d). A column with zero spread stays at 0.
    """
    sd = X.std(axis=0)
    return (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0)


def explained_variance_ratio(Z: np.ndarray) -> np.ndarray:
    """Share of the total variance on each principal direction.

    Returns shape (d,), non-increasing, summing to 1. Z is centred inside.
    """
    s = np.linalg.svd(Z - Z.mean(axis=0), compute_uv=False)
    return s**2 / np.sum(s**2)


def choose_components(ratios: np.ndarray, factor: float = 1.5) -> int:
    """Number of components to keep: those that clearly beat an average column.

    On standardised data every column carries 1/d of the variance, so pure
    noise directions sit near 1/d. Keep a direction only when its ratio is
    above `factor / d`.
    """
    return int(np.sum(ratios > factor / len(ratios)))


def pca_scores(Z: np.ndarray, k: int) -> np.ndarray:
    """Coordinates of each row on the top k principal directions, shape (n, k)."""
    Zc = Z - Z.mean(axis=0)
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    return Zc @ Vt[:k].T


# ---------------------------------------------------------------- k-means


def assign_nearest(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Index of the closest centre for each row. Written for you (chapter 048)."""
    scores = np.sum(centers**2, axis=1)[None, :] - 2.0 * (X @ centers.T)
    return np.argmin(scores, axis=1)


def lloyd(X: np.ndarray, init: np.ndarray, iters: int = 100):
    """Lloyd's loop from the given centres. Written for you (chapter 048).

    Returns (centers, assignments, inertia), where inertia is the sum of
    squared gaps from each row to its centre.
    """
    centers = init.copy()
    assignments = assign_nearest(X, centers)
    for _ in range(iters):
        for j in range(len(centers)):
            members = X[assignments == j]
            if len(members) > 0:
                centers[j] = members.mean(axis=0)
        new = assign_nearest(X, centers)
        if np.array_equal(new, assignments):
            break
        assignments = new
    return centers, assignments, float(np.sum((X - centers[assignments]) ** 2))


def kmeans_plusplus_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Seed k centres: the first at random, each next one drawn with probability
    proportional to the squared distance to the nearest centre so far.

    Returns shape (k, d), every row a copy of a row of X.
    """
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]))
    centers[0] = X[rng.integers(n)]
    sq = np.sum((X - centers[0]) ** 2, axis=1)
    for j in range(1, k):
        total = sq.sum()
        p = sq / total if total > 0 else np.full(n, 1.0 / n)
        centers[j] = X[rng.choice(n, p=p)]
        sq = np.minimum(sq, np.sum((X - centers[j]) ** 2, axis=1))
    return centers


def kmeans(X: np.ndarray, k: int, rng: np.random.Generator, n_init: int = 5):
    """Best of n_init seeded runs of Lloyd's loop.

    Returns (centers, labels, inertia) of the run with the lowest inertia.
    """
    best = None
    for _ in range(n_init):
        run = lloyd(X, kmeans_plusplus_init(X, k, rng))
        if best is None or run[2] < best[2]:
            best = run
    return best


def choose_k(X: np.ndarray, ks, rng: np.random.Generator, drop: float = 0.15):
    """Pick k at the elbow: the first k where one more cluster helps too little.

    Compute the best inertia for each k in `ks` (increasing). Return the
    smallest k whose next inertia is not below (1 - drop) times its own, that
    is, adding a cluster would cut inertia by less than `drop`. If no k
    qualifies, return the last one.

    Returns (k, inertias) with inertias a list aligned with `ks`.
    """
    ks = list(ks)
    inertias = [kmeans(X, k, rng)[2] for k in ks]
    for i in range(len(ks) - 1):
        if inertias[i + 1] > (1.0 - drop) * inertias[i]:
            return ks[i], inertias
    return ks[-1], inertias


def purity(labels: np.ndarray, truth: np.ndarray) -> float:
    """Fraction of rows that carry the majority planted segment of their cluster.

    Invariant to renaming the clusters, because each cluster is matched to
    whichever segment fills it most.
    """
    total = 0
    for c in np.unique(labels):
        total += np.bincount(truth[labels == c].astype(int)).max()
    return float(total / len(labels))


# ---------------------------------------------------------------- Gaussian mixture


def log_gaussian(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Log density of each row under N(mean, cov), shape (n,). Written for you (chapter 049)."""
    d = X.shape[1]
    gap = X - mean
    _, logdet = np.linalg.slogdet(cov)
    squared = np.sum((gap @ np.linalg.inv(cov)) * gap, axis=1)
    return -0.5 * (d * np.log(2.0 * np.pi) + logdet + squared)


def gmm_e_step(X, means, covs, weights):
    """E-step. Written for you (chapter 049).

    Returns (responsibilities, log_likelihood): shares of shape (n, k), whose
    rows sum to 1, and the total log-likelihood as a float.
    """
    scores = np.stack(
        [np.log(weights[j]) + log_gaussian(X, means[j], covs[j]) for j in range(len(weights))], axis=1
    )
    biggest = scores.max(axis=1, keepdims=True)
    log_px = biggest[:, 0] + np.log(np.exp(scores - biggest).sum(axis=1))
    return np.exp(scores - log_px[:, None]), float(log_px.sum())


def gmm_m_step(X: np.ndarray, resp: np.ndarray, reg_covar: float = 1e-6):
    """Refit weights, means, and covariances from the shares.

    Returns (means (k, d), covs (k, d, d), weights (k,)). Each covariance is
    measured around its own new mean, plus reg_covar times the identity.
    """
    n, d = X.shape
    counts = resp.sum(axis=0)
    means = (resp.T @ X) / counts[:, None]
    covs = np.empty((resp.shape[1], d, d))
    for j in range(resp.shape[1]):
        gap = X - means[j]
        covs[j] = (resp[:, j, None] * gap).T @ gap / counts[j] + reg_covar * np.eye(d)
    return means, covs, counts / n


def fit_gmm(X: np.ndarray, init_means: np.ndarray, iters: int = 50, reg_covar: float = 1e-6):
    """Run EM from the given starting means.

    Start with equal weights and every covariance equal to
    `np.var(X, axis=0).mean() * np.eye(d)`. Run the E-step, then repeat the
    M-step and E-step `iters` times.

    Returns (means, covs, weights, resp, history), where history holds the
    log-likelihood after every E-step, so it has iters + 1 entries.
    """
    k, d = init_means.shape
    means = init_means.copy()
    covs = np.stack([np.var(X, axis=0).mean() * np.eye(d) for _ in range(k)])
    weights = np.full(k, 1.0 / k)
    resp, log_l = gmm_e_step(X, means, covs, weights)
    history = [log_l]
    for _ in range(iters):
        means, covs, weights = gmm_m_step(X, resp, reg_covar)
        resp, log_l = gmm_e_step(X, means, covs, weights)
        history.append(log_l)
    return means, covs, weights, resp, history


# ---------------------------------------------------------------- trees and ensembles


def best_split(X: np.ndarray, y: np.ndarray, min_leaf: int = 5):
    """Best squared-error split by a sorted sweep. Written for you (chapter 038).

    Returns (feature, threshold, gain) or None when no split leaves min_leaf
    rows on each side. Gain is the drop in the sum of squared errors.
    """
    n, d = X.shape
    if n < 2 * min_leaf:
        return None
    best = None
    total_sse = float(np.sum((y - y.mean()) ** 2))
    counts = np.arange(1, n)
    for j in range(d):
        order = np.argsort(X[:, j], kind="mergesort")
        xs, ys = X[order, j], y[order]
        cs, cs2 = np.cumsum(ys), np.cumsum(ys**2)
        ls, ls2 = cs[:-1], cs2[:-1]
        rs, rs2 = cs[-1] - ls, cs2[-1] - ls2
        sse = (ls2 - ls**2 / counts) + (rs2 - rs**2 / (n - counts))
        ok = (xs[:-1] != xs[1:]) & (counts >= min_leaf) & (n - counts >= min_leaf)
        sse = np.where(ok, sse, np.inf)
        i = int(np.argmin(sse))
        if np.isfinite(sse[i]) and (best is None or sse[i] < best[0]):
            best = (float(sse[i]), j, float(0.5 * (xs[i] + xs[i + 1])))
    if best is None:
        return None
    return best[1], best[2], total_sse - best[0]


def grow_tree(X: np.ndarray, y: np.ndarray, max_depth: int, min_leaf: int = 5) -> dict:
    """Grow a regression tree as nested dicts.

    A leaf is {"value": mean of y, "feature": None}. An internal node also holds
    "feature", "threshold", "left" (rows with X[:, feature] <= threshold), and
    "right". Stop when depth reaches max_depth or `best_split` returns None or
    a gain of 1e-12 or less.
    """
    node = {"value": float(np.mean(y)), "feature": None}
    if max_depth == 0:
        return node
    split = best_split(X, y, min_leaf)
    if split is None or split[2] <= 1e-12:
        return node
    feature, threshold, _ = split
    left = X[:, feature] <= threshold
    node.update(
        feature=feature,
        threshold=threshold,
        left=grow_tree(X[left], y[left], max_depth - 1, min_leaf),
        right=grow_tree(X[~left], y[~left], max_depth - 1, min_leaf),
    )
    return node


def predict_tree(tree: dict, X: np.ndarray) -> np.ndarray:
    """Send every row down the tree and return the leaf values, shape (n,)."""
    out = np.empty(X.shape[0])

    def walk(node, rows):
        if node["feature"] is None:
            out[rows] = node["value"]
            return
        goes_left = X[rows, node["feature"]] <= node["threshold"]
        walk(node["left"], rows[goes_left])
        walk(node["right"], rows[~goes_left])

    walk(tree, np.arange(X.shape[0]))
    return out


def fit_bagged_trees(X, y, n_trees: int, max_depth: int, rng: np.random.Generator, min_leaf: int = 5):
    """Grow n_trees trees, each on a bootstrap sample of the rows.

    Returns (trees, in_bag) with in_bag a boolean array of shape
    (n_trees, n) that is True where a row appears in that tree's sample.
    """
    n = X.shape[0]
    trees, in_bag = [], np.zeros((n_trees, n), dtype=bool)
    for t in range(n_trees):
        idx = rng.integers(0, n, size=n)
        in_bag[t, idx] = True
        trees.append(grow_tree(X[idx], y[idx], max_depth, min_leaf))
    return trees, in_bag


def bagged_predict(trees: list, X: np.ndarray) -> np.ndarray:
    """Average the trees' predictions, shape (n,)."""
    return np.mean([predict_tree(t, X) for t in trees], axis=0)


def oob_error(X, y, trees, in_bag) -> float:
    """Mean squared error of each row predicted only by trees that never saw it.

    Rows that every tree saw are skipped.
    """
    total = np.zeros(len(y))
    count = np.zeros(len(y))
    for t, tree in enumerate(trees):
        out = ~in_bag[t]
        if out.any():
            total[out] += predict_tree(tree, X[out])
            count[out] += 1
    seen = count > 0
    return float(np.mean((total[seen] / count[seen] - y[seen]) ** 2))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared gap between predicted probability and the 0/1 outcome."""
    return float(np.mean((p - y) ** 2))


# ---------------------------------------------------------------- the pipeline


def find_segments(X: np.ndarray, rng: np.random.Generator):
    """Run the whole unsupervised half. Returns a dict with keys
    "Z", "n_components", "scores", "k", "labels", "resp", "history"."""
    Z = standardise(X)
    n_components = choose_components(explained_variance_ratio(Z))
    scores = pca_scores(Z, n_components)
    k, _ = choose_k(scores, range(1, 9), rng)
    centers, labels, _ = kmeans(scores, k, rng)
    _, _, _, resp, history = fit_gmm(scores, centers)
    return {"Z": Z, "n_components": n_components, "scores": scores, "k": k,
            "labels": labels, "resp": resp, "history": history}


if __name__ == "__main__":
    X, segment, churn = make_data(0)
    rng = np.random.default_rng(0)
    seg = find_segments(X, rng)
    ratios = explained_variance_ratio(seg["Z"])
    print("explained variance ratios:", np.round(ratios, 3))
    print(f"kept {seg['n_components']} components, chose k = {seg['k']}")
    print(f"k-means purity: {purity(seg['labels'], segment):.3f}")
    print(f"EM purity: {purity(seg['resp'].argmax(axis=1), segment):.3f}")
    print(f"log-likelihood {seg['history'][0]:.0f} -> {seg['history'][-1]:.0f}")

    F = np.concatenate([seg["Z"], seg["resp"]], axis=1)
    tr, te = slice(0, 1000), slice(1000, None)
    one = grow_tree(F[tr], churn[tr], max_depth=6)
    trees, in_bag = fit_bagged_trees(F[tr], churn[tr], 100, 6, rng)
    print(f"held-out Brier: base rate {brier(churn[te], churn[tr].mean()):.4f}, "
          f"one tree {brier(churn[te], predict_tree(one, F[te])):.4f}, "
          f"forest {brier(churn[te], bagged_predict(trees, F[te])):.4f}")
    print(f"out-of-bag error {oob_error(F[tr], churn[tr], trees, in_bag):.4f}")
