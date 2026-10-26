"""Checks for the spam filter project."""

import numpy as np

from mlcheck import almost, check, close, equal, fail, has, shape


def _tiny_model():
    """Two classes, three words. Spam favours word 0, ham favours word 2."""
    X = np.array([[3.0, 1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 3.0], [0.0, 0.0, 2.0], [1.0, 1.0, 2.0]])
    y = np.array([1, 1, 0, 0, 0])
    return X, y


@check("fit_nb returns smoothed, finite log word probabilities that match a hand calculation")
def test_fit_nb(sub):
    X, y = _tiny_model()
    log_theta, log_priors = has(sub, "fit_nb")(X, y, 1.0)
    shape(log_theta, (2, 3), "log_theta")
    shape(log_priors, (2,), "log_priors")
    # spam word totals (5, 1, 0) + 1 -> (6, 2, 1) / 9; ham totals (1, 2, 7) + 1 -> (2, 3, 8) / 13
    close(np.exp(log_theta[1]), np.array([6, 2, 1]) / 9.0, atol=1e-9, label="spam word probabilities")
    close(np.exp(log_theta[0]), np.array([2, 3, 8]) / 13.0, atol=1e-9, label="ham word probabilities")
    close(np.exp(log_priors), np.array([0.6, 0.4]), atol=1e-9, label="priors (ham, spam)")
    X, y = _tiny_model()
    log_theta, _ = sub.fit_nb(X, y, 0.5)
    if not np.all(np.isfinite(log_theta)):
        fail(f"log_theta has non-finite entries with alpha=0.5: {log_theta}")
    big, _ = sub.fit_nb(X, y, 1000.0)
    spread = np.ptp(np.exp(big[1]))
    if spread > 0.05:
        fail(f"alpha=1000 should make the word probabilities nearly equal, but they span {spread:.3f}")



@check("predict_nb picks the larger log score and does not underflow on long emails")
def test_predict_nb(sub):
    X, y = _tiny_model()
    model = sub.fit_nb(X, y, 1.0)
    got = sub.predict_nb(model, np.array([[4.0, 0.0, 0.0], [0.0, 0.0, 4.0]]))
    equal(np.asarray(got).tolist(), [1, 0], "labels for a spammy and a hammy email")
    long_mail = np.array([[3000.0, 0.0, 0.0], [0.0, 0.0, 3000.0]])
    got = sub.predict_nb(model, long_mail)
    equal(np.asarray(got).tolist(), [1, 0], "labels for emails with 3000 words")


@check("top_spam_words ranks words by the log ratio of spam to ham probability")
def test_top_words(sub):
    log_theta = np.log(np.array([[0.5, 0.3, 0.1, 0.1], [0.1, 0.2, 0.3, 0.4]]))
    got = sub.top_spam_words((log_theta, np.log([0.5, 0.5])), ["a", "b", "c", "d"], 2)
    equal(list(got), ["d", "c"], "top two spam words, strongest first")


@check("k_fold_indices partitions the rows into k disjoint folds of nearly equal size")
def test_folds(sub):
    folds = has(sub, "k_fold_indices")(23, 5, np.random.default_rng(0))
    equal(len(folds), 5, "number of folds")
    everything = np.sort(np.concatenate(folds))
    equal(everything.tolist(), list(range(23)), "union of the folds")
    sizes = sorted(len(f) for f in folds)
    if sizes[-1] - sizes[0] > 1:
        fail(f"fold sizes {sizes} differ by more than one")


@check("cv_error keeps held-out rows out of training, averages the fold errors, and is repeatable")
def test_cv_no_leak(sub):
    n = 60
    ids = np.arange(n, dtype=float)[:, None]
    y = np.random.default_rng(1).integers(0, 2, size=n)

    def fit(X, t):
        return set(X[:, 0].astype(int).tolist())

    def predict(model, X):
        # A memoriser: right on rows it trained on, wrong on rows it did not.
        return np.array([y[int(i)] if int(i) in model else 1 - y[int(i)] for i in X[:, 0]])

    err = has(sub, "cv_error")(fit, predict, ids, y, 5)
    almost(err, 1.0, 1e-12, "error of a memoriser under honest cross-validation (every held-out row is unseen)")
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 1))
    y = (X[:, 0] > 0.3).astype(int)
    fit = lambda A, t: None
    pred_all_zero = lambda m, A: np.zeros(len(A), dtype=int)
    err = sub.cv_error(fit, pred_all_zero, X, y, 4)
    almost(err, y.mean(), 1e-12, "cv error of always predicting ham")
    equal(sub.cv_error(fit, pred_all_zero, X, y, 4), err, "second call")



@check("grid_search_cv scores every setting on the same folds and picks the lowest error")
def test_grid_search(sub):
    rng = np.random.default_rng(3)
    X = rng.uniform(-1, 1, size=(200, 1))
    y = (X[:, 0] > 0.2).astype(int)
    make_fit = lambda s: (lambda A, t: s)
    predict = lambda s, A: (A[:, 0] > s).astype(int)
    settings = [-0.5, 0.0, 0.2, 0.6]
    best, errors = has(sub, "grid_search_cv")(make_fit, predict, X, y, settings, 5)
    shape(np.asarray(errors), (4,), "errors")
    if best != 0.2:
        fail(f"expected the true threshold 0.2 to win, got {best}; errors were {np.round(errors, 3)}")
    close(errors[2], 0.0, atol=1e-12, label="error of the true threshold")


@check("standardise uses the training mean and standard deviation, not its own")
def test_standardise(sub):
    rng = np.random.default_rng(4)
    train = rng.normal([3.0, 0.1], [2.0, 0.02], size=(100, 2))
    other = rng.normal([5.0, 0.3], [2.0, 0.02], size=(30, 2))
    Z = has(sub, "standardise")(train, train)
    close(Z.mean(axis=0), np.zeros(2), atol=1e-9, label="training column means")
    close(Z.std(axis=0), np.ones(2), atol=1e-9, label="training column stds")
    W = sub.standardise(train, other)
    close(W, (other - train.mean(axis=0)) / train.std(axis=0), atol=1e-9, label="other rows")
    if abs(W.mean(axis=0)[0]) < 0.5:
        fail("the other rows were centred on their own mean; use the training statistics")


@check("The kernel classifier separates a ring from a blob even when the features have unequal units")
def test_krr(sub):
    rng = np.random.default_rng(5)

    def sample(n):
        y = rng.integers(0, 2, size=n)
        r = np.where(y == 1, rng.normal(2.0, 0.25, size=n), np.abs(rng.normal(0, 0.5, size=n)))
        a = rng.uniform(0, 2 * np.pi, size=n)
        # The second feature is squeezed into a tiny range, like a ratio next to a count.
        X = np.column_stack([3 + r * np.cos(a), 0.1 + 0.01 * r * np.sin(a)])
        return X, y

    Xtr, ytr = sample(200)
    Xte, yte = sample(300)
    model = has(sub, "fit_krr_classifier")(Xtr, ytr, 0.5, 0.1)
    pred = np.asarray(has(sub, "predict_krr")(model, Xte))
    if not set(np.unique(pred)).issubset({0, 1}):
        fail(f"predict_krr must return labels 0 and 1, got values {np.unique(pred)}")
    acc = np.mean(pred == yte)
    if acc < 0.9:
        fail(f"held-out accuracy {acc:.3f}, expected at least 0.90; standardise before the kernel so the small feature counts")


@check("learning_curve scores on rows training never sees, and more_data_would_help reads the curves")
def test_learning_curve(sub):
    n = 100
    ids = np.arange(n, dtype=float)[:, None]
    y = np.random.default_rng(6).integers(0, 2, size=n)
    seen = []

    def fit(X, t):
        seen.append(X[:, 0].astype(int).tolist())
        return None

    zero = lambda m, X: np.zeros(len(X), dtype=int)
    tr, va = has(sub, "learning_curve")(fit, zero, ids, y, [10, 30, 60], 20)
    shape(np.asarray(tr), (3,), "train errors")
    equal([len(s) for s in seen], [10, 30, 60], "training sizes")
    if max(max(s) for s in seen) >= 80:
        fail("training rows came from the last 20 rows, which are the validation set")
    close(va, np.full(3, y[80:].mean()), atol=1e-12, label="validation error of always predicting ham")
    close(tr, np.array([y[:10].mean(), y[:30].mean(), y[:60].mean()]), atol=1e-12, label="training errors")
    f = has(sub, "more_data_would_help")
    if f([0.02, 0.03, 0.04], [0.30, 0.20, 0.10]) is not True:
        fail("a 0.06 gap with validation error falling from 0.30 to 0.10 should say more data helps")
    if f([0.14, 0.14, 0.14], [0.16, 0.15, 0.15]) is not False:
        fail("a gap of 0.01 should say more data will not help")
    if f([0.02, 0.03, 0.04], [0.10, 0.12, 0.13]) is not False:
        fail("validation error that rises with size should not say more data helps")



@check("select_and_test keeps the test rows out of every selection step")
def test_no_leak(sub):
    data = sub.make_data(0)
    base = has(sub, "select_and_test")(data, 5)
    scrambled = dict(data)
    y = data["y"].copy()
    d = data["n_dev"]
    y[d:] = np.random.default_rng(9).permutation(y[d:])
    scrambled["y"] = y
    other = sub.select_and_test(scrambled, 5)
    for key in ("nb_alpha", "nb_cv", "krr_gamma", "krr_lam", "krr_cv", "winner", "best_cv"):
        if other[key] != base[key]:
            fail(f"changing only the test labels changed `{key}` from {base[key]} to {other[key]}; selection must not see the test set")
    if abs(other["test_error"] - base["test_error"]) < 0.05:
        fail("shuffling the test labels barely moved test_error, so it is not scored on the test rows")


@check("The chosen filter beats 0.11 test error on seeds 0 to 2 and the report is consistent")
def test_end_to_end(sub):
    for seed in range(3):
        data = sub.make_data(seed)
        r = has(sub, "select_and_test")(data, 5)
        if r["test_error"] > 0.11:
            fail(f"seed {seed}: test error {r['test_error']:.3f}, expected at most 0.11 (winner {r['winner']}, alpha {r['nb_alpha']})")
        if r["winner"] != "nb":
            fail(f"seed {seed}: the word model should beat the kernel model on this data, got winner {r['winner']} "
                 f"(nb cv {r['nb_cv']:.3f}, kernel cv {r['krr_cv']:.3f})")
        close(r["gap"], r["test_error"] - r["best_cv"], atol=1e-12, label="gap")
        if r["nb_alpha"] < 0.05:
            fail(f"seed {seed}: cross-validation should pick real smoothing, but chose alpha={r['nb_alpha']}")


@check("The top ten spam words from the naive Bayes log ratios match the planted ones")
def test_top_words_recovered(sub):
    for seed in range(3):
        data = sub.make_data(seed)
        d = data["n_dev"]
        r = has(sub, "select_and_test")(data, 5)
        model = sub.fit_nb(data["counts"][:d], data["y"][:d], r["nb_alpha"])
        top = list(sub.top_spam_words(model, data["vocab"], 10))
        planted = {data["vocab"][i] for i in data["spam_idx"]}
        hits = len(planted & set(top))
        if hits < 7:
            fail(f"seed {seed}: only {hits} of the top 10 words {top} are planted spam words {sorted(planted)}; expected at least 7")
