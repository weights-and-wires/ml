"""Checks for the sigmoid and probability exercise."""

import numpy as np

from mlcheck import check, close, fail, finite, has, shape


@check("sigmoid is correct at ordinary scores")
def test_sigmoid_values(sub):
    sigmoid = has(sub, "sigmoid")
    z = np.array([-2.0, 0.0, 2.0, 3.2])
    want = np.array([0.11920292202211755, 0.5, 0.8807970779778823, 0.9608342772032357])
    close(sigmoid(z), want, atol=1e-12, label="sigmoid at -2, 0, 2, 3.2")


@check("sigmoid is finite, warning-free, and correct at magnitudes as large as 800")
def test_sigmoid_stability(sub):
    sigmoid = has(sub, "sigmoid")
    z = np.array([-800.0, -20.0, 0.0, 20.0, 800.0])
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            out = sigmoid(z)
    except FloatingPointError as err:
        fail(f"sigmoid raised a floating-point error ({err}); only exponentiate a number that is zero or negative")
    finite(out, label="sigmoid")
    close(out, np.array([0.0, 2.061153622438558e-09, 0.5, 0.9999999979388463, 1.0]), atol=1e-12)


@check("sigmoid keeps the shape of its input and satisfies sigmoid(-z) = 1 - sigmoid(z)")
def test_sigmoid_symmetry(sub):
    sigmoid = has(sub, "sigmoid")
    rng = np.random.default_rng(0)
    z = rng.normal(scale=6.0, size=(4, 5))
    out = sigmoid(z)
    shape(out, (4, 5), label="sigmoid output")
    close(sigmoid(-z), 1.0 - out, atol=1e-12, label="sigmoid(-z) versus 1 - sigmoid(z)")


@check("logit undoes sigmoid")
def test_logit_round_trip(sub):
    sigmoid = has(sub, "sigmoid")
    logit = has(sub, "logit")
    close(logit(np.array([0.5])), np.array([0.0]), atol=1e-12, label="logit(0.5)")
    close(logit(np.array([0.8807970779778823])), np.array([2.0]), atol=1e-9, label="logit(0.8808)")
    z = np.linspace(-10.0, 10.0, 41)
    close(logit(sigmoid(z)), z, atol=1e-8, label="logit(sigmoid(z))")


@check("a step of +1 in the score multiplies the odds by e")
def test_odds_ratio(sub):
    sigmoid = has(sub, "sigmoid")
    rng = np.random.default_rng(1)
    z = rng.uniform(-4.0, 4.0, size=10)
    p0 = sigmoid(z)
    p1 = sigmoid(z + 1.0)
    ratio = (p1 / (1.0 - p1)) / (p0 / (1.0 - p0))
    close(ratio, np.full(10, np.e), rtol=1e-9, label="odds after / odds before")


@check("predict_proba scores the two lesson emails and any large score")
def test_predict_proba(sub):
    predict_proba = has(sub, "predict_proba")
    X = np.array([[1.0, 0.0], [1.0, 2.0]])
    w = np.array([0.5, -1.0])
    p = predict_proba(w, X)
    shape(p, (2,), label="predict_proba output")
    close(p, np.array([0.6224593312018546, 0.18242552380635635]), atol=1e-12, label="email probabilities")
    big = predict_proba(np.array([800.0]), np.array([[1.0], [-1.0]]))
    finite(big, label="predict_proba at a score of 800")
    close(big, np.array([1.0, 0.0]), atol=1e-12, label="predict_proba at +-800")
