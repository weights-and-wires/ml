"""Checks for the nearest-neighbours and matrix-factorisation exercise."""

import numpy as np

from mlcheck import almost, check, close, equal, fail, has, shape


def five_points():
    X = np.array([[0, 0], [1, 0], [0, 2], [3, 3], [4, 3]], dtype=float)
    y = np.array([0, 0, 1, 1, 1])
    price = np.array([10.0, 12.0, 20.0, 30.0, 34.0])
    return X, y, price, np.array([[1.0, 1.0]])


def planted_table(seed=1, n_users=60, n_items=40, rank=2, density=0.3):
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n_users, rank))
    V = rng.normal(size=(n_items, rank))
    R = U @ V.T + 0.2 * rng.normal(size=(n_users, n_items))
    observed = rng.uniform(size=R.shape) < density
    train = rng.uniform(size=R.shape) < 0.8
    return R, (observed & train).astype(float), (observed & ~train).astype(float)


@check("pairwise_sq_dists gives 2, 1, 2, 8, 13 on the five points and matches a double loop")
def test_dists(sub):
    f = has(sub, "pairwise_sq_dists")
    X, _, _, q = five_points()
    got = f(q, X)
    shape(got, (1, 5), label="distances")
    close(got, np.array([[2.0, 1.0, 2.0, 8.0, 13.0]]), atol=1e-12, label="distances to (1, 1)")
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(7, 4)) * 50, rng.normal(size=(5, 4)) * 50
    want = np.array([[np.sum((a - b) ** 2) for b in B] for a in A])
    close(f(A, B), want, rtol=1e-9, label="distances on random rows")
    if np.min(f(A, A)) < 0:
        fail(f"a squared distance came out negative: {np.min(f(A, A))}; clip rounding error at 0")


@check("knn_predict votes among the k nearest, breaks distance ties by index and vote ties by the smaller label")
def test_knn_predict(sub):
    f = has(sub, "knn_predict")
    X, y, _, q = five_points()
    for k in (1, 2, 3):
        got = f(X, y, q, k)
        equal(np.asarray(got), np.array([0]), label=f"label at k={k}")
    # k=5 includes everything: three 1s beat two 0s.
    equal(np.asarray(f(X, y, q, 5)), np.array([1]), label="label at k=5")
    # Vote tie: neighbours are rows 1 and 2 with labels 1 and 0, so the smaller label wins.
    X1 = np.array([[0.0], [1.0], [2.0], [3.0]])
    y1 = np.array([1, 1, 0, 0])
    equal(np.asarray(f(X1, y1, np.array([[1.5]]), 2)), np.array([0]), label="vote tie")
    rng = np.random.default_rng(3)
    Xtr = rng.normal(size=(80, 3))
    ytr = rng.integers(0, 3, size=80)
    Xte = rng.normal(size=(25, 3))
    got = np.asarray(f(Xtr, ytr, Xte, 5))
    want = []
    for x in Xte:
        order = sorted(range(80), key=lambda i: (np.sum((Xtr[i] - x) ** 2), i))[:5]
        want.append(int(np.argmax(np.bincount(ytr[order], minlength=3))))
    equal(got, np.array(want), label="labels against a brute-force loop")


@check("knn_regress averages the k nearest targets: 12, 11, and 14 on the five points")
def test_knn_regress(sub):
    f = has(sub, "knn_regress")
    X, _, price, q = five_points()
    for k, want in ((1, 12.0), (2, 11.0), (3, 14.0)):
        got = np.asarray(f(X, price, q, k))
        shape(got, (1,), label=f"output at k={k}")
        almost(float(got[0]), want, tol=1e-12, label=f"prediction at k={k}")
    rng = np.random.default_rng(4)
    Xtr = rng.normal(size=(60, 2))
    ytr = rng.normal(size=60)
    got = np.asarray(f(Xtr, ytr, Xtr[:5] + 0.01, 1))
    close(got, ytr[:5], atol=1e-12, label="k=1 returns the nearest row's target")


@check("loo_knn_error predicts each row from the other rows, so random labels score near 0.5")
def test_loo(sub):
    f = has(sub, "loo_knn_error")
    X = np.array([[0.0], [1.0], [10.0], [11.0]])
    almost(float(f(X, np.array([0, 0, 1, 1]), 1)), 0.0, tol=1e-12, label="two clean groups")
    almost(float(f(X, np.array([0, 1, 0, 1]), 1)), 1.0, tol=1e-12, label="labels that swap in each group")
    # A wrong implementation that lets each row vote for itself (the query is in its own
    # neighbour list) returns exactly 0.0 here, because k=1 finds the row's own label.
    rng = np.random.default_rng(5)
    Xr = rng.normal(size=(200, 3))
    yr = rng.integers(0, 2, size=200)
    got = float(f(Xr, yr, 1))
    if not 0.35 < got < 0.65:
        fail(f"leave-one-out error on random labels should be near 0.5, got {got:.3f}; "
             "does a row count itself as its own neighbour?")


@check("distance_ratio is a seeded value in (0, 1) that rises toward 1 as the dimension grows")
def test_ratio(sub):
    f = has(sub, "distance_ratio")
    a = f(5, 100, np.random.default_rng(0))
    b = f(5, 100, np.random.default_rng(0))
    if a != b:
        fail(f"same seed gave {a} and {b}")
    means = []
    for d in (2, 10, 100, 1000):
        means.append(float(np.mean([f(d, 200, np.random.default_rng(s)) for s in range(10)])))
    if not all(0 < m < 1 for m in means):
        fail(f"ratios must lie in (0, 1), got {means}")
    if not all(means[i] < means[i + 1] for i in range(3)):
        fail(f"mean ratios for d = 2, 10, 100, 1000 should increase, got {np.round(means, 3)}")
    if means[0] > 0.2 or means[-1] < 0.8:
        fail(f"expected below 0.2 at d=2 and above 0.8 at d=1000, got {np.round(means, 3)}")
    # Draw order: the n points first, then the query.
    rng = np.random.default_rng(7)
    X = rng.uniform(size=(30, 4))
    q = rng.uniform(size=(1, 4))
    dist = np.sqrt(((X - q) ** 2).sum(axis=1))
    almost(f(4, 30, np.random.default_rng(7)), float(dist.min() / dist.max()), tol=1e-9, label="ratio")


@check("masked_loss counts only observed entries and adds the ridge fee on U and V")
def test_masked_loss(sub):
    f = has(sub, "masked_loss")
    R = np.array([[2.0, 7.0], [3.0, 4.0]])
    M = np.array([[1.0, 0.0], [1.0, 1.0]])
    U = np.array([[1.0], [2.0]])
    V = np.array([[1.0], [1.0]])
    # P = [[1, 1], [2, 2]]; observed residuals 1, 1, 2 give 1 + 1 + 4 = 6; fee: 1 * (1 + 4 + 1 + 1) = 7.
    almost(float(f(R, M, U, V, 1.0)), 13.0, tol=1e-12, label="loss with lam=1")
    R2 = R.copy()
    R2[0, 1] = -500.0
    almost(float(f(R2, M, U, V, 1.0)), 13.0, tol=1e-12, label="loss after changing an unobserved entry")


@check("als_step equals the ridge solve on each row's observed items, and ignores unobserved entries")
def test_als_step(sub):
    f = has(sub, "als_step")
    R = np.array([[2.0, 0.0, 5.0]])
    M = np.array([[1.0, 0.0, 1.0]])
    V = np.array([[1.0], [2.0], [3.0]])
    got = np.asarray(f(R, M, np.zeros((1, 1)), V, 1.0))
    close(got, np.array([[17 / 11]]), atol=1e-12, label="hand example 17/11")
    # A wrong implementation that ignores the mask fits the zero in column 1 and returns 1.1333.
    rng = np.random.default_rng(6)
    R = rng.normal(size=(12, 9))
    M = (rng.uniform(size=R.shape) < 0.4).astype(float)
    M[3] = 0.0
    V = rng.normal(size=(9, 3))
    U0 = rng.normal(size=(12, 3))
    got = np.asarray(f(R, M, U0, V, 0.7))
    shape(got, (12, 3), label="new U")
    want = np.zeros((12, 3))
    for i in range(12):
        o = M[i] > 0
        if o.any():
            A = np.vstack([V[o], np.sqrt(0.7) * np.eye(3)])
            b = np.concatenate([R[i, o], np.zeros(3)])
            want[i] = np.linalg.lstsq(A, b, rcond=None)[0]
    close(got, want, atol=1e-9, label="rows against an augmented least squares")
    R_changed = R.copy()
    R_changed[M == 0] = 1000.0
    close(np.asarray(f(R_changed, M, U0, V, 0.7)), got, atol=1e-9,
          label="result after changing unobserved entries")
    equal(got[3], np.zeros(3), label="a user with no ratings")


@check("The user-bias baseline and masked_rmse match a hand table")
def test_baselines(sub):
    bias = has(sub, "user_bias_baseline")
    rmse = has(sub, "masked_rmse")
    R = np.array([[4.0, 0.0, 2.0], [5.0, 5.0, 0.0], [0.0, 0.0, 0.0]])
    M = np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    # Mean of the four observed ratings is 4. User 0 averages 3 (offset -1), user 1 averages 5 (offset +1).
    got = np.asarray(bias(R, M))
    want = np.array([[3.0, 3.0, 3.0], [5.0, 5.0, 5.0], [4.0, 4.0, 4.0]])
    close(got, want, atol=1e-12, label="baseline table")
    almost(float(rmse(R, M, np.full((3, 3), 4.0))), float(np.sqrt(1.5)), tol=1e-12, label="RMSE of the global mean")
    almost(float(rmse(R, M, got)), float(np.sqrt(0.5)), tol=1e-12, label="RMSE of the user-bias table")
    almost(float(rmse(R, M, R + 99.0 * (1 - M))), 0.0, tol=1e-12, label="RMSE ignores unobserved entries")


@check("fit_als never raises the masked loss, and its last history entry is the loss of the returned factors")
def test_fit_monotone(sub):
    f = has(sub, "fit_als")
    loss = has(sub, "masked_loss")
    R, M, _ = planted_table()
    U, V, hist = f(R, M, 2, 1.0, 10, np.random.default_rng(0))
    shape(np.asarray(U), (60, 2), label="U")
    shape(np.asarray(V), (40, 2), label="V")
    if len(hist) != 11:
        fail(f"history should hold the starting loss plus 10 sweeps (11 entries), got {len(hist)}")
    for a, b in zip(hist, hist[1:]):
        if b > a + 1e-9:
            fail(f"the masked loss rose from {a:.6f} to {b:.6f}")
    almost(float(hist[-1]), float(loss(R, M, U, V, 1.0)), tol=1e-6, label="last history entry")
    if hist[-1] > 0.5 * hist[0]:
        fail(f"loss should fall by more than half in 10 sweeps: {hist[0]:.1f} to {hist[-1]:.1f}")


@check("fit_als beats the global-mean and user-bias baselines on held-out entries of a planted rank-2 table")
def test_fit_holdout(sub):
    f = has(sub, "fit_als")
    rmse = has(sub, "masked_rmse")
    bias = has(sub, "user_bias_baseline")
    R, Mtr, Mte = planted_table(seed=2, n_users=120, n_items=80, density=0.15)
    U, V, _ = f(R, Mtr, 2, 1.0, 30, np.random.default_rng(0))
    got = float(rmse(R, Mte, np.asarray(U) @ np.asarray(V).T))
    mu = np.sum(R * Mtr) / np.sum(Mtr)
    base_mean = float(rmse(R, Mte, np.full(R.shape, mu)))
    base_user = float(rmse(R, Mte, np.asarray(bias(R, Mtr))))
    if got > 0.6 * min(base_mean, base_user):
        fail(f"held-out RMSE {got:.3f} should be under 60% of the better baseline "
             f"(global mean {base_mean:.3f}, user bias {base_user:.3f})")
