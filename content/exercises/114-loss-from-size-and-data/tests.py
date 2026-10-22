"""Checks for the loss-from-size-and-data exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, has

COEFFS = (1.69, 400.0, 1100.0, 0.34, 0.34)


@check("loss_terms returns (floor, size penalty, data penalty) for the chapter's example")
def test_terms(sub):
    terms = has(sub, "loss_terms")(1e9, 2e10, COEFFS)
    if len(terms) != 3:
        fail(f"loss_terms should return 3 values, got {len(terms)}")
    floor, size_p, data_p = terms
    almost(floor, 1.69, tol=1e-12, label="floor")
    almost(size_p, 0.34839, tol=2e-5, label="size penalty at N = 1e9")
    almost(data_p, 0.34597, tol=2e-5, label="data penalty at D = 2e10")


@check("predict_loss gives 2.3844 for the chapter's example, and equals the sum of the three terms")
def test_predict(sub):
    pl, lt = has(sub, "predict_loss"), has(sub, "loss_terms")
    almost(pl(1e9, 2e10, COEFFS), 2.38436, tol=2e-5, label="predict_loss(1e9, 2e10)")
    for N, D in [(1e8, 1e9), (3e9, 5e10), (2e11, 4e12)]:
        almost(pl(N, D, COEFFS), sum(lt(N, D, COEFFS)), tol=1e-9, label=f"sum of terms at N={N:.0e}, D={D:.0e}")


@check("The tuple is unpacked in the order (E, A, B, alpha, beta)")
def test_order(sub):
    odd = (1.0, 2.0, 3.0, 0.5, 0.25)
    # 1 + 2 / 4**0.5 + 3 / 16**0.25 = 1 + 1 + 1.5
    almost(has(sub, "predict_loss")(4.0, 16.0, odd), 3.5, tol=1e-12, label="predict_loss with odd coefficients")
    floor, size_p, data_p = has(sub, "loss_terms")(4.0, 16.0, odd)
    almost(size_p, 1.0, tol=1e-12, label="size penalty with odd coefficients")
    almost(data_p, 1.5, tol=1e-12, label="data penalty with odd coefficients")


@check("Loss falls as N or D grows, and approaches the floor E when both are huge")
def test_shape(sub):
    pl = has(sub, "predict_loss")
    over_n = [pl(n, 1e10, COEFFS) for n in (1e7, 1e8, 1e9, 1e10, 1e11)]
    over_d = [pl(1e9, d, COEFFS) for d in (1e8, 1e9, 1e10, 1e11, 1e12)]
    if not all(a > b for a, b in zip(over_n, over_n[1:])):
        fail(f"loss should fall as N grows, got {over_n}")
    if not all(a > b for a, b in zip(over_d, over_d[1:])):
        fail(f"loss should fall as D grows, got {over_d}")
    almost(pl(1e60, 1e60, COEFFS), 1.69, tol=1e-6, label="loss at enormous N and D")
    stuck = pl(1e60, 2e10, COEFFS)
    almost(stuck, 1.69 + 1100.0 / 2e10**0.34, tol=1e-6, label="loss with huge N but D = 2e10")


@check("marginal_gain matches the closed form A / N**alpha * (1 - factor**-alpha)")
def test_gain_closed_form(sub):
    mg = has(sub, "marginal_gain")
    E, A, B, alpha, beta = COEFFS
    for N, factor in [(1e9, 2.0), (3e8, 4.0), (5e10, 10.0)]:
        want = A / N**alpha * (1.0 - factor**-alpha)
        almost(mg(N, 2e10, COEFFS, factor), want, tol=1e-9, label=f"gain at N={N:.0e}, factor={factor:g}")
    almost(mg(1e9, 2e10, COEFFS), 0.07315, tol=2e-5, label="default factor of 2 at the chapter's example")


@check("The gain from a bigger model does not depend on the data, and growing the data helps more when data is scarce")
def test_gain_independent(sub):
    mg, pl = has(sub, "marginal_gain"), has(sub, "predict_loss")
    almost(mg(1e9, 2e9, COEFFS), mg(1e9, 2e12, COEFFS), tol=1e-9, label="gain at D=2e9 versus D=2e12")
    gain_n = mg(1e9, 2e9, COEFFS)
    gain_d = pl(1e9, 2e9, COEFFS) - pl(1e9, 4e9, COEFFS)
    if not gain_d > 2 * gain_n:
        fail(f"at N=1e9, D=2e9 doubling the data should gain more than twice the model's {gain_n:.4f}, got {gain_d:.4f}")
