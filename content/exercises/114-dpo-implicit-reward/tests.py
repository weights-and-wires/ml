"""Checks for the DPO exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, finite_difference, has, shape


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def _batch(seed, n):
    rng = np.random.default_rng(seed)
    return tuple(rng.uniform(-4.0, -0.2, size=n) for _ in range(4))


@check("implicit_reward is beta times the difference of log probabilities, for any shape")
def test_implicit_reward(sub):
    f = has(sub, "implicit_reward")
    close(f(np.array([-1.0, -3.0]), np.array([-2.0, -2.0]), 0.1), [0.1, -0.1], atol=1e-12, label="beta = 0.1")
    rng = np.random.default_rng(0)
    p, q = rng.normal(size=(3, 4)), rng.normal(size=(3, 4))
    out = f(p, q, 2.5)
    shape(out, (3, 4), label="output")
    close(out, 2.5 * (p - q), atol=1e-12, label="beta = 2.5")


@check("implicit_reward recovers a planted reward from optimal_policy, up to one constant per prompt")
def test_planted_reward(sub):
    implicit = has(sub, "implicit_reward")
    optimal = has(sub, "optimal_policy")
    rng = np.random.default_rng(6)
    beta = 0.6
    logits = rng.normal(size=5)
    ref = np.exp(logits) / np.exp(logits).sum()
    true_reward = rng.normal(size=5) * 2.0
    pi_star = optimal(np.log(ref), true_reward, beta)
    recovered = implicit(np.log(pi_star), np.log(ref), beta)
    diff = recovered - true_reward
    close(diff, np.full(5, diff[0]), atol=1e-9, label="recovered minus planted reward")
    almost(float(diff[0]), -beta * np.log(np.sum(ref * np.exp(true_reward / beta))), tol=1e-9, label="the constant is -beta log Z")


@check("dpo_loss is log 2 when the policy equals the reference and 0.3133 when the margin is 1")
def test_loss_hand(sub):
    loss = has(sub, "dpo_loss")
    ref_w, ref_l = np.log(np.array([0.5])), np.log(np.array([0.2]))
    almost(loss(ref_w, ref_l, ref_w, ref_l, 1.0), np.log(2.0), tol=1e-12, label="policy equals reference")
    almost(loss(ref_w + 0.5, ref_l - 0.5, ref_w, ref_l, 1.0), 0.313262, tol=1e-5, label="margin 1")
    almost(loss(np.array([-1.0]), np.array([-3.0]), np.array([-2.0]), np.array([-2.0]), 0.1), -np.log(_sig(0.2)), tol=1e-10, label="beta 0.1, margin 0.2")


@check("dpo_loss equals bradley_terry_loss applied to the two implicit rewards")
def test_loss_is_bt(sub):
    dpo = has(sub, "dpo_loss")
    bt = has(sub, "bradley_terry_loss")
    implicit = has(sub, "implicit_reward")
    pw, pl, rw, rl = _batch(3, 10)
    beta = 0.3
    almost(dpo(pw, pl, rw, rl, beta), bt(implicit(pw, rw, beta), implicit(pl, rl, beta)), tol=1e-10, label="dpo_loss against Bradley-Terry on implicit rewards")


@check("dpo_loss does not change when both reference log probabilities of a prompt shift by one constant")
def test_z_cancels(sub):
    dpo = has(sub, "dpo_loss")
    pw, pl, rw, rl = _batch(4, 8)
    shift = np.random.default_rng(14).uniform(-5.0, 5.0, size=8)
    almost(dpo(pw, pl, rw + shift, rl + shift, 0.5), dpo(pw, pl, rw, rl, 0.5), tol=1e-10, label="loss after shifting both reference log probabilities")


@check("dpo_grad matches finite differences, with shapes (n,) and (n,)")
def test_grad_fd(sub):
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


@check("The gradient's size is beta * (1 - sigmoid(u)) / n, so a pair ranked wrong is pushed harder than one ranked right")
def test_weight(sub):
    grad = has(sub, "dpo_grad")
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
    if np.any(np.abs(gw) >= beta / 2):
        fail("the weight beta * (1 - sigmoid(u)) must stay below beta, so each gradient entry is below beta / n")


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
