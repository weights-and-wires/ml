"""Checks for the vanishing and exploding gradients exercise."""

import numpy as np

from mlcheck import check, close, fail, has


@check("spectral_radius is the largest absolute eigenvalue of a diagonal matrix")
def test_radius_diag(sub):
    radius = has(sub, "spectral_radius")
    close(radius(np.diag([0.5, 0.9])), 0.9, atol=1e-12, label="rho of diag(0.5, 0.9)")
    close(radius(np.diag([-2.0, 1.0, 0.3])), 2.0, atol=1e-12, label="rho of diag(-2, 1, 0.3), the sign must not matter")


@check("spectral_radius of a rotation is 1, although it has no real eigenvalue")
def test_radius_rotation(sub):
    radius = has(sub, "spectral_radius")
    angle = 0.7
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    close(radius(rotation), 1.0, atol=1e-10, label="rho of a rotation")
    close(radius(1.05 * rotation), 1.05, atol=1e-10, label="rho of a scaled rotation")


@check("spectral_radius of a random symmetric matrix matches an independent computation")
def test_radius_symmetric(sub):
    radius = has(sub, "spectral_radius")
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 6))
    S = (A + A.T) / 2
    want = float(np.max(np.abs(np.linalg.eigvalsh(S))))
    close(radius(S), want, atol=1e-10, label="rho of a symmetric matrix")


@check("gradient_scale_over_time computes (kappa * rho) ** T")
def test_scale(sub):
    scale = has(sub, "gradient_scale_over_time")
    close(scale(np.diag([0.5, 0.9]), 1.0, 10), 0.3486784401, rtol=1e-9, label="0.9 ** 10")
    close(scale(np.diag([0.5, 0.9]), 0.5, 3), 0.091125, rtol=1e-9, label="(0.5 * 0.9) ** 3")
    angle = 1.0
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    close(scale(1.05 * rotation, 1.0, 100), 131.50125784630401, rtol=1e-8, label="1.05 ** 100")


@check("steps_until_below returns the smallest T, and None when the gradient never falls")
def test_steps(sub):
    steps = has(sub, "steps_until_below")
    if steps(0.9, 1.0, 0.01) != 44:
        fail(f"steps_until_below(0.9, 1.0, 0.01) returned {steps(0.9, 1.0, 0.01)}, expected 44 (0.9**44 = 0.0097, 0.9**43 = 0.0108)")
    if steps(0.5, 1.0, 0.25) != 3:
        fail(f"steps_until_below(0.5, 1.0, 0.25) returned {steps(0.5, 1.0, 0.25)}, expected 3, because 0.5**2 equals 0.25 and is not below it")
    for rho, kappa in [(1.0, 1.0), (1.05, 1.0), (2.0, 0.5)]:
        if steps(rho, kappa, 0.1) is not None:
            fail(f"steps_until_below({rho}, {kappa}, 0.1) should be None because kappa * rho = {rho * kappa} is 1 or more")


@check("steps_until_below agrees with counting steps one at a time")
def test_steps_count(sub):
    steps = has(sub, "steps_until_below")
    for rho, kappa, threshold in [(0.99, 0.9, 1e-3), (1.2, 0.7, 1e-6), (0.3, 1.0, 0.5), (0.97, 1.0, 1e-4)]:
        rate = rho * kappa
        count = 1
        while rate**count >= threshold:
            count += 1
        got = steps(rho, kappa, threshold)
        if got != count:
            fail(f"steps_until_below({rho}, {kappa}, {threshold}) returned {got}, counting one step at a time gives {count}")


@check("The gradient your bptt measures stays within a factor of two of the prediction at rho = 0.9 and 1.05")
def test_measured_vs_predicted(sub):
    backward = has(sub, "bptt")
    predict = has(sub, "gradient_scale_over_time")
    d, T = 6, 100
    for rho in (0.9, 1.05):
        rng = np.random.default_rng(8)
        q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        W_hh = rho * q
        x = np.zeros((T, 1, d))
        dH = np.zeros((T, 1, d))
        g = rng.normal(size=(1, d))
        dH[-1] = g / np.linalg.norm(g)
        h0 = rng.normal(size=(1, d)) * 1e-3
        dh0 = backward(dH, x, h0, np.zeros((d, d)), W_hh, np.zeros(d))[4]
        measured = float(np.linalg.norm(dh0))
        predicted = predict(W_hh, 1.0, T)
        ratio = measured / predicted
        if not 0.5 < ratio < 2.0:
            fail(f"at rho={rho}, measured {measured:.3e} and predicted {predicted:.3e} differ by a factor of {ratio:.2f}")
