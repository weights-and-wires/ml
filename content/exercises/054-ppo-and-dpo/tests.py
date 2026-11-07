"""Checks for the PPO and DPO exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, finite_difference, has, shape


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def _batch(seed, n):
    rng = np.random.default_rng(seed)
    return tuple(rng.uniform(-4.0, -0.2, size=n) for _ in range(4))


@check("policy_ratio gives 1 when nothing changed and [1.1, 1.4, 0.7, 1.3] on the worked probabilities")
def test_ratio(sub):
    ratio = has(sub, "policy_ratio")
    lp = np.log(np.array([0.3, 0.1, 0.6]))
    close(ratio(lp, lp), np.ones(3), atol=1e-12, label="unchanged policy")
    old = np.array([0.5, 0.2, 0.3, 0.2])
    new = np.array([0.55, 0.28, 0.21, 0.26])
    close(ratio(np.log(new), np.log(old)), [1.1, 1.4, 0.7, 1.3], atol=1e-10, label="worked ratios")


@check("advantage gives [1, 1, -1, -1] on the worked rewards, and zeros (not nan) when every sample matches its prediction")
def test_advantage_hand(sub):
    adv = has(sub, "advantage")
    out = adv(np.array([3.3, 3.1, 2.4, 2.6]), np.array([2.8, 2.6, 2.9, 3.1]))
    shape(out, (4,), label="advantage")
    close(out, [1.0, 1.0, -1.0, -1.0], atol=1e-6, label="advantages")
    r = np.array([1.0, 2.0, 3.0])
    flat = adv(r, r.copy())
    if not np.all(np.isfinite(flat)):
        fail(f"advantage returned non-finite values {flat} when rewards equal values")
    close(flat, np.zeros(3), atol=1e-6, label="advantage with no spread")


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


@check("clipped_objective is 0.05 on the four worked samples, against 0.125 without the clip")
def test_objective_hand(sub):
    obj = has(sub, "clipped_objective")
    ratio = np.array([1.1, 1.4, 0.7, 1.3])
    a = np.array([1.0, 1.0, -1.0, -1.0])
    almost(obj(ratio, a, 0.2), 0.05, tol=1e-10, label="clipped objective")
    almost(obj(ratio, a, 10.0), 0.125, tol=1e-10, label="a huge eps means no clip")


@check("clipped_objective is flat where the clip bites, and keeps pulling back a bad answer whose ratio rose above 1 + eps")
def test_objective_gradient(sub):
    obj = has(sub, "clipped_objective")
    eps = 0.2

    def grad(rho, a):
        return float(finite_difference(lambda x: obj(x, np.array([a]), eps), np.array([rho]))[0])

    almost(grad(1.5, 1.0), 0.0, tol=1e-6, label="positive advantage, ratio 1.5")
    almost(grad(0.5, -1.0), 0.0, tol=1e-6, label="negative advantage, ratio 0.5")
    almost(grad(1.0, 2.0), 2.0, tol=1e-6, label="positive advantage inside the band")
    almost(grad(1.4, -1.5), -1.5, tol=1e-6, label="negative advantage -1.5 at ratio 1.4")
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


@check("implicit_reward is beta times the difference of log probabilities, and recovers a planted reward up to one constant")
def test_implicit_reward(sub):
    f = has(sub, "implicit_reward")
    close(f(np.array([-1.0, -3.0]), np.array([-2.0, -2.0]), 0.1), [0.1, -0.1], atol=1e-12, label="beta = 0.1")
    rng = np.random.default_rng(0)
    p, q = rng.normal(size=(3, 4)), rng.normal(size=(3, 4))
    out = f(p, q, 2.5)
    shape(out, (3, 4), label="output")
    close(out, 2.5 * (p - q), atol=1e-12, label="beta = 2.5")

    optimal = has(sub, "optimal_policy")
    rng = np.random.default_rng(6)
    beta = 0.6
    logits = rng.normal(size=5)
    ref = np.exp(logits) / np.exp(logits).sum()
    true_reward = rng.normal(size=5) * 2.0
    pi_star = optimal(np.log(ref), true_reward, beta)
    diff = f(np.log(pi_star), np.log(ref), beta) - true_reward
    close(diff, np.full(5, diff[0]), atol=1e-9, label="recovered minus planted reward")
    almost(float(diff[0]), -beta * np.log(np.sum(ref * np.exp(true_reward / beta))), tol=1e-9, label="the constant is -beta log Z")


@check("dpo_loss is log 2 at the start, equals Bradley-Terry on the implicit rewards, and ignores a shift of both reference log probabilities")
def test_dpo_loss(sub):
    loss = has(sub, "dpo_loss")
    bt = has(sub, "bradley_terry_loss")
    implicit = has(sub, "implicit_reward")
    ref_w, ref_l = np.log(np.array([0.5])), np.log(np.array([0.2]))
    almost(loss(ref_w, ref_l, ref_w, ref_l, 1.0), np.log(2.0), tol=1e-12, label="policy equals reference")
    almost(loss(ref_w + 0.5, ref_l - 0.5, ref_w, ref_l, 1.0), 0.313262, tol=1e-5, label="margin 1")
    pw, pl, rw, rl = _batch(3, 10)
    beta = 0.3
    almost(loss(pw, pl, rw, rl, beta), bt(implicit(pw, rw, beta), implicit(pl, rl, beta)), tol=1e-10, label="dpo_loss against Bradley-Terry on implicit rewards")
    shift = np.random.default_rng(14).uniform(-5.0, 5.0, size=10)
    almost(loss(pw, pl, rw + shift, rl + shift, beta), loss(pw, pl, rw, rl, beta), tol=1e-10, label="loss after shifting both reference log probabilities")


@check("dpo_grad matches finite differences, and its size is beta * (1 - sigmoid(u)) / n")
def test_dpo_grad(sub):
    loss = has(sub, "dpo_loss")
    grad = has(sub, "dpo_grad")
    n = 6
    pw, pl, rw, rl = _batch(5, n)
    beta = 0.4
    gw, gl = grad(pw, pl, rw, rl, beta)
    shape(gw, (n,), label="grad_chosen")
    shape(gl, (n,), label="grad_rejected")
    fd = finite_difference(lambda x: loss(x[:n], x[n:], rw, rl, beta), np.concatenate([pw, pl]))
    close(np.concatenate([gw, gl]), fd, rtol=1e-4, atol=1e-8, label="dpo_grad against finite differences")

    beta = 0.5
    rw, rl = np.zeros(2), np.zeros(2)
    pw, pl = np.array([-2.0, 2.0]), np.array([2.0, -2.0])  # margins -2 and +2
    gw, gl = grad(pw, pl, rw, rl, beta)
    u = beta * ((pw - rw) - (pl - rl))
    want = beta * (1.0 - _sig(u)) / 2
    close(np.abs(gw), want, atol=1e-10, label="size of the chosen gradient")
    close(gl, -gw, atol=1e-12, label="rejected gradient is the negation")
    if not abs(gw[0]) > 5 * abs(gw[1]):
        fail(f"the pair ranked wrong should get far more push than the one ranked right, got {abs(gw[0]):.4f} and {abs(gw[1]):.4f}")


@check("Fifty DPO steps lower the loss and raise the chosen log probability relative to the rejected one")
def test_training(sub):
    grad = has(sub, "dpo_grad")
    loss = has(sub, "dpo_loss")
    rng = np.random.default_rng(7)
    n, beta, lr = 8, 0.2, 0.5
    rw = rng.uniform(-3.0, -0.5, size=n)
    rl = rng.uniform(-3.0, -0.5, size=n)
    pw, pl = rw.copy(), rl.copy()
    gap0, loss0 = float(np.mean(pw - pl)), loss(pw, pl, rw, rl, beta)
    for _ in range(50):
        gw, gl = grad(pw, pl, rw, rl, beta)
        pw, pl = pw - lr * gw, pl - lr * gl
    gap1, loss1 = float(np.mean(pw - pl)), loss(pw, pl, rw, rl, beta)
    if gap1 <= gap0 + 1e-6:
        fail(f"the gap between chosen and rejected log probabilities should widen: {gap0:.4f} then {gap1:.4f}")
    if loss1 >= loss0:
        fail(f"the DPO loss should fall: {loss0:.4f} then {loss1:.4f}")
