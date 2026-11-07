"""Checks for the sigmoid and log loss exercise."""

import numpy as np

from mlcheck import check, close, fail, finite, has, shape


def sigmoid_ref(z):
    return 1.0 / (1.0 + np.exp(-z))


EMAILS_X = np.array([[1.0, 0.0], [1.0, 2.0]])
EMAILS_Y = np.array([1.0, 0.0])
EMAILS_W = np.array([0.5, -1.0])


@check("sigmoid is correct at ordinary scores and finite, warning-free, and correct at +-800")
def test_sigmoid_values_and_stability(sub):
    sigmoid = has(sub, "sigmoid")
    z = np.array([-2.0, 0.0, 2.0, 3.2])
    want = np.array([0.11920292202211755, 0.5, 0.8807970779778823, 0.9608342772032357])
    close(sigmoid(z), want, atol=1e-12, label="sigmoid at -2, 0, 2, 3.2")
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


@check("logit undoes sigmoid, and a step of +1 in the score multiplies the odds by e")
def test_logit_and_odds(sub):
    sigmoid = has(sub, "sigmoid")
    logit = has(sub, "logit")
    close(logit(np.array([0.5])), np.array([0.0]), atol=1e-12, label="logit(0.5)")
    close(logit(np.array([0.8807970779778823])), np.array([2.0]), atol=1e-9, label="logit(0.8808)")
    z = np.linspace(-10.0, 10.0, 41)
    close(logit(sigmoid(z)), z, atol=1e-8, label="logit(sigmoid(z))")
    rng = np.random.default_rng(1)
    z = rng.uniform(-4.0, 4.0, size=10)
    p0 = sigmoid(z)
    p1 = sigmoid(z + 1.0)
    ratio = (p1 / (1.0 - p1)) / (p0 / (1.0 - p0))
    close(ratio, np.full(10, np.e), rtol=1e-9, label="odds after / odds before")


@check("predict_proba scores the two lesson emails and any large score")
def test_predict_proba(sub):
    predict_proba = has(sub, "predict_proba")
    p = predict_proba(EMAILS_W, EMAILS_X)
    shape(p, (2,), label="predict_proba output")
    close(p, np.array([0.6224593312018546, 0.18242552380635635]), atol=1e-12, label="email probabilities")
    big = predict_proba(np.array([800.0]), np.array([[1.0], [-1.0]]))
    finite(big, label="predict_proba at a score of 800")
    close(big, np.array([1.0, 0.0]), atol=1e-12, label="predict_proba at +-800")


@check("softplus is finite at +-800 and matches log(1 + exp(z)) at moderate scores")
def test_softplus(sub):
    softplus = has(sub, "softplus")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            extreme = softplus(np.array([-800.0, 800.0]))
    except FloatingPointError as err:
        fail(f"softplus raised a floating-point error ({err}); keep the exponent at or below zero")
    finite(extreme, label="softplus at +-800")
    close(extreme, np.array([0.0, 800.0]), atol=1e-9, label="softplus at -800 and 800")
    z = np.linspace(-20.0, 20.0, 41)
    close(softplus(z), np.log1p(np.exp(z)), rtol=1e-10, atol=1e-12, label="softplus at moderate scores")
    close(softplus(np.array([0.0])), np.array([np.log(2.0)]), atol=1e-12, label="softplus(0) = ln 2")


@check("naive_log_loss is minus the log of the probability given to the truth")
def test_naive_log_loss(sub):
    naive = has(sub, "naive_log_loss")
    close(naive(np.array([0.4]), np.array([1.0])), -np.log(0.4), atol=1e-12, label="p = 0.4, y = 1")
    close(naive(np.array([0.4]), np.array([0.0])), -np.log(0.6), atol=1e-12, label="p = 0.4, y = 0")
    p = np.array([0.6224593312018546, 0.18242552380635635])
    close(naive(p, EMAILS_Y), 0.6754902621628589, atol=1e-10, label="the two lesson emails")


@check("log_loss gives 0.6755 on the two lesson emails and agrees with naive_log_loss at moderate scores")
def test_log_loss_matches_naive(sub):
    log_loss = has(sub, "log_loss")
    naive = has(sub, "naive_log_loss")
    close(log_loss(EMAILS_W, EMAILS_X, EMAILS_Y), 0.6754902621628589, atol=1e-10, label="log_loss on the emails")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    w = np.array([0.7, -1.1, 0.4])
    y = (rng.uniform(size=50) < sigmoid_ref(X @ w)).astype(float)
    close(log_loss(w, X, y), naive(sigmoid_ref(X @ w), y), rtol=1e-10, label="log_loss versus the textbook formula")


@check("log_loss is finite at scores of +-800 and has the right value there")
def test_log_loss_extreme(sub):
    log_loss = has(sub, "log_loss")
    X = np.array([[1.0], [1.0]])
    w_big = np.array([800.0])
    close(log_loss(w_big, X, np.array([1.0, 1.0])), 0.0, atol=1e-9, label="right and certain at +800")
    close(log_loss(w_big, X, np.array([0.0, 0.0])), 1600.0, atol=1e-6, label="wrong and certain at +800")
    finite(log_loss(-w_big, X, np.array([1.0, 0.0])), label="log_loss at -800")
    close(log_loss(-w_big, X, np.array([1.0, 0.0])), 800.0, atol=1e-6, label="one wrong, one right at -800")


@check("log_loss at zero weights is n log 2, and the ridge fee adds (ridge / 2) ||w||^2")
def test_zero_and_ridge(sub):
    log_loss = has(sub, "log_loss")
    X = np.ones((10, 2))
    y = np.zeros(10)
    close(log_loss(np.zeros(2), X, y), 10 * np.log(2.0), atol=1e-9, label="loss at w = 0")
    close(log_loss(np.zeros(2), X, np.ones(10)), 10 * np.log(2.0), atol=1e-9, label="loss at w = 0, other labels")
    w = np.array([3.0, -4.0])
    gap = log_loss(w, X, y, ridge=2.0) - log_loss(w, X, y, ridge=0.0)
    close(gap, 25.0, atol=1e-8, label="ridge fee for weights 3 and -4 at ridge 2")
