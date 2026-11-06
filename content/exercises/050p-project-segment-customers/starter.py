"""Project: find customer segments, then predict who leaves.

Fill in every function that raises NotImplementedError, in file order. The
functions marked "Written for you" are your own code from earlier chapters,
already working, so the project stays about the pipeline.
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
    

    Steps:
      1. Take the column means and standard deviations with axis=0.
      2. Where a deviation is 0, divide by 1 instead (np.where).
      3. Subtract the mean, then divide.
    """
    raise NotImplementedError



def explained_variance_ratio(Z: np.ndarray) -> np.ndarray:
    """Share of the total variance on each principal direction.

    Returns shape (d,), non-increasing, summing to 1. Z is centred inside.
    

    Steps:
      1. Subtract the column means.
      2. Call np.linalg.svd with compute_uv=False to get the singular values.
      3. Square them and divide by their sum.
    """
    raise NotImplementedError



def choose_components(ratios: np.ndarray, factor: float = 1.5) -> int:
    """Number of components to keep: those that clearly beat an average column.

    On standardised data every column carries 1/d of the variance, so pure
    noise directions sit near 1/d. Keep a direction only when its ratio is
    above `factor / d`.
    

    Steps:
      1. The cut is factor / len(ratios).
      2. Count the ratios strictly above it.
    """
    raise NotImplementedError



def pca_scores(Z: np.ndarray, k: int) -> np.ndarray:
    """Coordinates of each row on the top k principal directions, shape (n, k).

    Steps:
      1. Centre Z, then take the SVD with full_matrices=False.
      2. The rows of Vt are the directions; keep the first k.
      3. Project the centred data onto them.
    """
    raise NotImplementedError



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
    

    Steps:
      1. The first centre is a random row (rng.integers).
      2. Keep each row's squared distance to its nearest chosen centre.
      3. Draw the next centre with probabilities proportional to those squared distances (rng.choice with p=), then update with np.minimum.
    """
    raise NotImplementedError



def kmeans(X: np.ndarray, k: int, rng: np.random.Generator, n_init: int = 5):
    """Best of n_init seeded runs of Lloyd's loop.

    Returns (centers, labels, inertia) of the run with the lowest inertia.
    

    Steps:
      1. Seed with kmeans_plusplus_init and run lloyd, n_init times.
      2. Keep the run whose inertia (the third value lloyd returns) is lowest.
    """
    raise NotImplementedError



def choose_k(X: np.ndarray, ks, rng: np.random.Generator, drop: float = 0.15):
    """Pick k at the elbow: the first k where one more cluster helps too little.

    Compute the best inertia for each k in `ks` (increasing). Return the
    smallest k whose next inertia is not below (1 - drop) times its own, that
    is, adding a cluster would cut inertia by less than `drop`. If no k
    qualifies, return the last one.

    Returns (k, inertias) with inertias a list aligned with `ks`.
    

    Steps:
      1. Compute the best inertia for each k with `kmeans`.
      2. Walk through consecutive pairs; stop at the first k whose successor's inertia is above (1 - drop) times its own.
      3. Return that k and the whole inertia list.
    """
    raise NotImplementedError



def purity(labels: np.ndarray, truth: np.ndarray) -> float:
    """Fraction of rows that carry the majority planted segment of their cluster.

    Invariant to renaming the clusters, because each cluster is matched to
    whichever segment fills it most.
    

    Steps:
      1. For each distinct cluster, count the segments of its rows with np.bincount.
      2. Take the largest count and add it to a running total.
      3. Divide the total by the number of rows.
    """
    raise NotImplementedError



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
    

    Steps:
      1. Soft counts: resp.sum(axis=0). Weights are counts / n.
      2. Means: (resp.T @ X) / counts[:, None].
      3. For each blob, measure gaps from its NEW mean, weight them by that blob's column of resp, and divide by its count. Add reg_covar * identity.
    """
    raise NotImplementedError



def fit_gmm(X: np.ndarray, init_means: np.ndarray, iters: int = 50, reg_covar: float = 1e-6):
    """Run EM from the given starting means.

    Start with equal weights and every covariance equal to
    `np.var(X, axis=0).mean() * np.eye(d)`. Run the E-step, then repeat the
    M-step and E-step `iters` times.

    Returns (means, covs, weights, resp, history), where history holds the
    log-likelihood after every E-step, so it has iters + 1 entries.
    

    Steps:
      1. Start with the given means, equal weights, and covariances np.var(X, axis=0).mean() * np.eye(d).
      2. Run gmm_e_step once and record the log-likelihood.
      3. Repeat iters times: gmm_m_step, gmm_e_step, record.
    """
    raise NotImplementedError



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
    

    Steps:
      1. Make a leaf dict with the mean of y and feature None.
      2. Return it if max_depth is 0, if best_split returns None, or if the gain is 1e-12 or less.
      3. Otherwise split rows with X[:, feature] <= threshold and recurse on each side with max_depth - 1.
    """
    raise NotImplementedError



def predict_tree(tree: dict, X: np.ndarray) -> np.ndarray:
    """Send every row down the tree and return the leaf values, shape (n,).

    Steps:
      1. Fill an output array of length n.
      2. Recurse with the row indices that reach each node; at a leaf, write its value to those rows.
      3. At an internal node, send indices left or right by the threshold.
    """
    raise NotImplementedError



def fit_bagged_trees(X, y, n_trees: int, max_depth: int, rng: np.random.Generator, min_leaf: int = 5):
    """Grow n_trees trees, each on a bootstrap sample of the rows.

    Returns (trees, in_bag) with in_bag a boolean array of shape
    (n_trees, n) that is True where a row appears in that tree's sample.
    

    Steps:
      1. For each tree, draw n row indices with replacement (rng.integers).
      2. Mark those rows True in that tree's row of in_bag.
      3. Grow the tree on X[idx], y[idx].
    """
    raise NotImplementedError



def bagged_predict(trees: list, X: np.ndarray) -> np.ndarray:
    """Average the trees' predictions, shape (n,).

    Steps:
      1. Predict with every tree and average over trees (axis=0).
    """
    raise NotImplementedError



def oob_error(X, y, trees, in_bag) -> float:
    """Mean squared error of each row predicted only by trees that never saw it.

    Rows that every tree saw are skipped.
    

    Steps:
      1. For each tree, find the rows it did not see (~in_bag[t]) and predict only those.
      2. Add each prediction to a running total and count it.
      3. Average per row, skip rows with count 0, and return the mean squared error.
    """
    raise NotImplementedError



def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared gap between predicted probability and the 0/1 outcome."""
    return float(np.mean((p - y) ** 2))


# ---------------------------------------------------------------- the pipeline


def find_segments(X: np.ndarray, rng: np.random.Generator):
    """Run the whole unsupervised half. Returns a dict with keys
    "Z", "n_components", "scores", "k", "labels", "resp", "history".

    Steps:
      1. Z = standardise(X); keep choose_components(explained_variance_ratio(Z)) components with pca_scores.
      2. Pick k with choose_k over range(1, 9), then run kmeans on the scores with that k.
      3. Start fit_gmm from the k-means centres and collect the dict.
    """
    raise NotImplementedError


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
