"""Checks for the spread-and-shadows exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, has, shape

U = np.array([1.0, 1.0]) / np.sqrt(2.0)
W = np.array([1.0, -1.0]) / np.sqrt(2.0)
HAND = np.array([[2.0, 0.0], [0.0, 2.0], [4.0, 4.0]])


def _cloud(seed=0, n=60, d=4):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d))
    X = rng.normal(size=(n, d)) @ A + rng.normal(size=d) * 5.0
    return X - X.mean(axis=0)


@check("center subtracts the column mean, and adding it back recovers X")
def test_center(sub):
    got = has(sub, "center")
    rng = np.random.default_rng(0)
    X = rng.normal(loc=5.0, scale=2.0, size=(40, 4))
    Xc, mean = got(X)
    shape(Xc, X.shape, label="centred data")
    shape(mean, (4,), label="mean vector")
    close(Xc.mean(axis=0), np.zeros(4), atol=1e-10, label="column means of the centred data")
    close(Xc + mean, X, atol=1e-10, label="undoing the centring")
    Xc3, mean3 = got(HAND)
    close(mean3, np.array([2.0, 2.0]), atol=1e-12, label="the middle of the three-point cloud")


@check("variance_along gives 6 along (1,1)/sqrt2 and 2 along (1,-1)/sqrt2 on the three points")
def test_variance_hand(sub):
    got = has(sub, "variance_along")
    Xc = HAND - np.array([2.0, 2.0])
    almost(float(got(Xc, U)), 6.0, tol=1e-9, label="variance along u")
    almost(float(got(Xc, W)), 2.0, tol=1e-9, label="variance along w")
    almost(float(got(Xc, np.array([1.0, 0.0]))), 4.0, tol=1e-9, label="variance along the x axis")


@check("variance_along matches np.var with ddof=1 on random data, and u @ C @ u")
def test_variance_random(sub):
    got = has(sub, "variance_along")
    Xc = _cloud(seed=1)
    rng = np.random.default_rng(2)
    C = np.cov(Xc, rowvar=False)
    for _ in range(4):
        u = rng.normal(size=4)
        u /= np.linalg.norm(u)
        want = float(np.var(Xc @ u, ddof=1))
        almost(float(got(Xc, u)), want, tol=1e-8 * max(1.0, want), label="variance of the scores")
        almost(float(got(Xc, u)), float(u @ C @ u), tol=1e-8 * max(1.0, want), label="u @ C @ u")


@check("sweep_directions follows 4 + 2 sin(2 theta) and peaks at 45 degrees")
def test_sweep(sub):
    got = has(sub, "sweep_directions")
    Xc = HAND - np.array([2.0, 2.0])
    angles = np.radians(np.arange(0, 180, 5))
    out = np.asarray(got(Xc, angles))
    shape(out, angles.shape, label="sweep output")
    close(out, 4.0 + 2.0 * np.sin(2.0 * angles), atol=1e-9, label="variance against angle")
    best = np.degrees(angles[np.argmax(out)])
    almost(float(best), 45.0, tol=1e-9, label="best angle in degrees")


@check("shadow_split gives kept 12 and lost 4 along u, and kept 4 and lost 12 along w")
def test_split_hand(sub):
    got = has(sub, "shadow_split")
    Xc = HAND - np.array([2.0, 2.0])
    kept, lost = got(Xc, U)
    almost(float(kept), 12.0, tol=1e-9, label="kept along u")
    almost(float(lost), 4.0, tol=1e-9, label="lost along u")
    kept, lost = got(Xc, W)
    almost(float(kept), 4.0, tol=1e-9, label="kept along w")
    almost(float(lost), 12.0, tol=1e-9, label="lost along w")


@check("kept plus lost equals the total squared length, with lost measured from the leftover vectors")
def test_split_pythagoras(sub):
    got = has(sub, "shadow_split")
    Xc = _cloud(seed=3)
    rng = np.random.default_rng(4)
    total = float(np.sum(Xc**2))
    for _ in range(4):
        u = rng.normal(size=4)
        u /= np.linalg.norm(u)
        kept, lost = got(Xc, u)
        almost(float(kept) + float(lost), total, tol=1e-8 * total, label="kept + lost")
        # distance from each point to the line, found without reusing the scores
        proj = Xc @ np.outer(u, u)
        almost(float(lost), float(np.sum((Xc - proj) ** 2)), tol=1e-8 * total, label="lost vs distance to the line")


@check("the angle that keeps the most also loses the least")
def test_kept_most_lost_least(sub):
    got = has(sub, "shadow_split")
    rng = np.random.default_rng(5)
    Xc = rng.normal(size=(80, 2)) @ np.array([[2.0, 1.0], [0.0, 0.6]])
    Xc -= Xc.mean(axis=0)
    angles = np.radians(np.arange(0, 180, 3))
    pairs = [got(Xc, np.array([np.cos(a), np.sin(a)])) for a in angles]
    kept = np.array([p[0] for p in pairs])
    lost = np.array([p[1] for p in pairs])
    if int(np.argmax(kept)) != int(np.argmin(lost)):
        fail(f"most kept is at angle index {int(np.argmax(kept))}, but least lost is at {int(np.argmin(lost))}")
    if np.any(np.diff(kept + lost) > 1e-8 * np.sum(Xc**2)) or np.any(np.diff(kept + lost) < -1e-8 * np.sum(Xc**2)):
        fail("kept + lost should stay the same at every angle")
