"""Checks for the PPO advantage and clip exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, finite_difference, has, shape


@check("policy_ratio gives 1 when nothing changed and [1.1, 1.4, 0.7, 1.3] on the worked probabilities")
def test_ratio(sub):
    ratio = has(sub, "policy_ratio")
    lp = np.log(np.array([0.3, 0.1, 0.6]))
    close(ratio(lp, lp), np.ones(3), atol=1e-12, label="unchanged policy")
    old = np.array([0.5, 0.2, 0.3, 0.2])
    new = np.array([0.55, 0.28, 0.21, 0.26])
    close(ratio(np.log(new), np.log(old)), [1.1, 1.4, 0.7, 1.3], atol=1e-10, label="worked ratios")


@check("advantage gives [1, 1, -1, -1] on the worked rewards and values")
def test_advantage_hand(sub):
    adv = has(sub, "advantage")
    out = adv(np.array([3.3, 3.1, 2.4, 2.6]), np.array([2.8, 2.6, 2.9, 3.1]))
    shape(out, (4,), label="advantage")
    close(out, [1.0, 1.0, -1.0, -1.0], atol=1e-6, label="advantages")


@check("advantage has mean 0 and standard deviation 1, and ignores a shared shift or scale of the rewards")
def test_advantage_normalised(sub):
    adv = has(sub, "advantage")
    rng = np.random.default_rng(0)
    r, v = rng.normal(size=32) * 3 + 10, rng.normal(size=32)
    a = adv(r, v)
    almost(float(a.mean()), 0.0, tol=1e-9, label="mean")
    almost(float(a.std()), 1.0, tol=1e-6, label="standard deviation")
    close(adv(r + 5.0, v + 5.0), a, atol=1e-9, label="after adding 5 to rewards and values")
    close(adv(4.0 * r, 4.0 * v), a, atol=1e-6, label="after multiplying rewards and values by 4")


@check("advantage is all zeros, not nan, when every sample matches its prediction")
def test_advantage_zero_variance(sub):
    adv = has(sub, "advantage")
    r = np.array([1.0, 2.0, 3.0])
    out = adv(r, r.copy())
    if not np.all(np.isfinite(out)):
        fail(f"advantage returned non-finite values {out} when rewards equal values")
    close(out, np.zeros(3), atol=1e-6, label="advantage with no spread")


@check("clipped_objective is 0.05 on the four worked samples, against 0.125 without the clip")
def test_objective_hand(sub):
    obj = has(sub, "clipped_objective")
    ratio = np.array([1.1, 1.4, 0.7, 1.3])
    a = np.array([1.0, 1.0, -1.0, -1.0])
    almost(obj(ratio, a, 0.2), 0.05, tol=1e-10, label="clipped objective")
    almost(obj(ratio, a, 10.0), 0.125, tol=1e-10, label="a huge eps means no clip")


@check("clipped_objective has zero gradient where the clip bites: a good answer above 1 + eps, a bad one below 1 - eps")
def test_objective_flat(sub):
    obj = has(sub, "clipped_objective")
    eps = 0.2

    def grad(rho, a):
        return float(finite_difference(lambda x: obj(x, np.array([a]), eps), np.array([rho]))[0])

    almost(grad(1.5, 1.0), 0.0, tol=1e-6, label="positive advantage, ratio 1.5")
    almost(grad(0.5, -1.0), 0.0, tol=1e-6, label="negative advantage, ratio 0.5")
    almost(grad(1.0, 2.0), 2.0, tol=1e-6, label="positive advantage inside the band")


@check("clipped_objective keeps pulling back a bad answer whose ratio rose above 1 + eps")
def test_objective_pessimistic(sub):
    obj = has(sub, "clipped_objective")
    eps = 0.2
    g = finite_difference(lambda x: obj(x, np.array([-1.5]), eps), np.array([1.4]))[0]
    almost(float(g), -1.5, tol=1e-6, label="gradient for advantage -1.5 at ratio 1.4")
    almost(obj(np.array([1.4]), np.array([-1.5]), eps), -2.1, tol=1e-10, label="objective keeps the full penalty")


@check("Gradient ascent on a good answer's ratio stops once the ratio leaves the band")
def test_ascent_stops(sub):
    ratio_fn = has(sub, "policy_ratio")
    obj = has(sub, "clipped_objective")
    eps, lr = 0.2, 0.1
    theta = np.array([0.0])  # log new probability minus log old probability
    for _ in range(100):
        g = finite_difference(lambda t: obj(ratio_fn(t, np.zeros(1)), np.array([1.0]), eps), theta)
        theta = theta + lr * g
    rho = float(ratio_fn(theta, np.zeros(1))[0])
    if rho < 1.0 + eps - 1e-6:
        fail(f"the ratio should climb to 1 + eps = {1 + eps}, but stopped at {rho:.4f}")
    if rho > (1.0 + eps) * np.exp(lr * (1.0 + eps)) + 1e-6:
        fail(f"the ratio should stop near 1 + eps, but reached {rho:.4f}")
