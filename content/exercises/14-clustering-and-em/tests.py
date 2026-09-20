"""Checks for the clustering and EM exercise."""

import numpy as np

from mlcheck import check, close, fail, has, shape


def planted_clusters(n_per: int = 60, seed: int = 0, centers=None):
    """Well-separated Gaussian blobs with known ground truth."""
    rng = np.random.default_rng(seed)
    if centers is None:
        centers = np.array(
            [[0.0, 0.0], [8.0, 0.0], [4.0, 7.0], [-4.0, 6.0], [8.0, 8.0]]
        )
    X = np.concatenate([c + 0.8 * rng.normal(size=(n_per, 2)) for c in centers])
    return X, centers


@check("kmeans_objective sums squared distances to the assigned center")
def test_objective(sub):
    got = has(sub, "kmeans_objective")
    X = np.array([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]])
    centers = np.array([[0.0, 0.0], [5.0, 0.0]])
    assignments = np.array([0, 0, 1])
    # distances squared: 0, 1, 1 -> total 2
    close(got(X, centers, assignments), 2.0, label="kmeans_objective")


@check("lloyd's objective never increases across iterations")
def test_lloyd_monotone(sub):
    lloyd = has(sub, "lloyd")
    X, _ = planted_clusters(seed=1)
    rng = np.random.default_rng(5)
    idx = rng.choice(X.shape[0], size=5, replace=False)
    init = X[idx]
    centers, assignments, history = lloyd(X, 5, init, 15)
    shape(centers, (5, 2), label="lloyd centers")
    shape(assignments, (X.shape[0],), label="lloyd assignments")
    diffs = np.diff(history)
    if np.any(diffs > 1e-9):
        fail(
            "the objective increased at some iteration, which contradicts "
            f"alternating minimisation: history was {[round(h, 3) for h in history]}"
        )
    final = has(sub, "kmeans_objective")(X, centers, assignments)
    close(final, history[-1], rtol=1e-6, label="final objective matches history")


@check("kmeans++ seeding beats uniform seeding on a planted-cluster problem")
def test_kmeans_plusplus(sub):
    kpp = has(sub, "kmeans_plusplus_init")
    lloyd = has(sub, "lloyd")
    X, _ = planted_clusters(seed=2)
    k = 5

    pp_objs, uniform_objs = [], []
    for seed in range(12):
        rng_pp = np.random.default_rng(1000 + seed)
        init_pp = kpp(X, k, rng_pp)
        shape(init_pp, (k, 2), label="kmeans_plusplus_init output")
        _, _, hist_pp = lloyd(X, k, init_pp, 25)
        pp_objs.append(hist_pp[-1])

        rng_uniform = np.random.default_rng(2000 + seed)
        idx = rng_uniform.choice(X.shape[0], size=k, replace=False)
        _, _, hist_uniform = lloyd(X, k, X[idx], 25)
        uniform_objs.append(hist_uniform[-1])

    pp_mean, uniform_mean = float(np.mean(pp_objs)), float(np.mean(uniform_objs))
    if pp_mean >= uniform_mean:
        fail(
            "k-means++ should reach a lower objective on average on well-separated "
            f"clusters: got mean {pp_mean:.1f} against uniform's {uniform_mean:.1f}"
        )


@check("gmm_e_step returns responsibilities that sum to one per point")
def test_responsibilities_sum_to_one(sub):
    e_step = has(sub, "gmm_e_step")
    X, centers = planted_clusters(n_per=20, seed=3, centers=np.array([[0.0, 0.0], [5.0, 0.0]]))
    covs = np.stack([np.eye(2), np.eye(2)])
    weights = np.array([0.5, 0.5])
    resp, ll = e_step(X, centers, covs, weights)
    shape(resp, (X.shape[0], 2), label="responsibilities")
    close(resp.sum(axis=1), np.ones(X.shape[0]), atol=1e-9, label="row sums")
    if not np.isfinite(ll):
        fail(f"log-likelihood is not finite: {ll}")


@check("gmm_e_step's log-likelihood matches a direct log-sum-exp computation")
def test_log_likelihood_value(sub):
    e_step = has(sub, "gmm_e_step")
    rng = np.random.default_rng(4)
    X = rng.normal(size=(30, 2))
    means = np.array([[0.5, -0.5], [-1.0, 1.0]])
    covs = np.stack([np.eye(2) * 1.5, np.eye(2) * 0.7])
    weights = np.array([0.4, 0.6])

    _, ll = e_step(X, means, covs, weights)

    d = X.shape[1]
    log_weighted = np.empty((X.shape[0], 2))
    for j in range(2):
        diff = X - means[j]
        precision = np.linalg.inv(covs[j])
        maha = np.einsum("ij,jk,ik->i", diff, precision, diff)
        _, logdet = np.linalg.slogdet(covs[j])
        log_weighted[:, j] = (
            np.log(weights[j]) - 0.5 * (d * np.log(2 * np.pi) + logdet + maha)
        )
    want = float(np.sum(np.log(np.sum(np.exp(log_weighted), axis=1))))
    close(ll, want, rtol=1e-6, label="mixture log-likelihood")


@check("gmm_m_step recovers the weighted mean and covariance from responsibilities")
def test_m_step(sub):
    m_step = has(sub, "gmm_m_step")
    rng = np.random.default_rng(6)
    X = rng.normal(loc=2.0, scale=1.0, size=(200, 2))
    # Hard responsibilities: every point belongs entirely to component 0, none
    # to component 1, so component 0's parameters must be the sample statistics.
    resp = np.zeros((200, 2))
    resp[:, 0] = 1.0

    means, covs, weights = m_step(X, resp, reg_covar=0.0)
    shape(means, (2, 2), label="means")
    shape(covs, (2, 2, 2), label="covs")
    close(weights[0], 1.0, atol=1e-9, label="weight of the occupied component")
    close(means[0], X.mean(axis=0), rtol=1e-8, label="mean of the occupied component")
    empirical_cov = np.cov(X.T, bias=True)
    close(covs[0], empirical_cov, rtol=1e-6, label="covariance of the occupied component")


@check("EM's mixture log-likelihood increases monotonically")
def test_em_monotone(sub):
    fit_gmm = has(sub, "fit_gmm")
    X, _ = planted_clusters(n_per=40, seed=7)
    means, covs, weights, resp, ll_history = fit_gmm(X, 5, 20, np.random.default_rng(8))
    shape(resp, (X.shape[0], 5), label="final responsibilities")
    diffs = np.diff(ll_history)
    if np.any(diffs < -1e-6):
        fail(
            "the log-likelihood decreased at some EM iteration, which contradicts "
            f"the ELBO argument: history was {[round(v, 2) for v in ll_history]}"
        )
    if ll_history[-1] <= ll_history[0] + 1.0:
        fail(
            "EM barely moved the log-likelihood; it should improve substantially "
            f"from {ll_history[0]:.2f} to something well above it, got {ll_history[-1]:.2f}"
        )


@check("EM with shared isotropic covariance driven to zero reproduces Lloyd's assignments")
def test_zero_variance_limit(sub):
    lloyd = has(sub, "lloyd")
    e_step = has(sub, "gmm_e_step")
    X, _ = planted_clusters(n_per=50, seed=9)
    k = 5
    init = has(sub, "kmeans_plusplus_init")(X, k, np.random.default_rng(10))
    centers, assignments, _ = lloyd(X, k, init, 30)

    d = X.shape[1]
    tiny = 1e-6
    covs = np.stack([tiny * np.eye(d) for _ in range(k)])
    weights = np.full(k, 1.0 / k)
    resp, _ = e_step(X, centers, covs, weights)
    em_assignments = np.argmax(resp, axis=1)

    if not np.array_equal(em_assignments, assignments):
        fail(
            "as the shared covariance shrinks to zero, EM's hardened assignment "
            "should match Lloyd's nearest-center assignment exactly, but "
            f"{np.mean(em_assignments != assignments):.1%} of points disagree"
        )
