"""Project: build and choose a spam filter (reference solution).

You build a spam filter for a small mail provider and an honest estimate of its
accuracy. Two models compete: multinomial naive Bayes on word counts, and an RBF
kernel ridge classifier on two dense numeric features. Cross-validation tunes each
one, a learning curve says whether more mail would help, and a test set that
selection never touches gives the final score.

Conventions: labels are 0 for ham and 1 for spam. A `fit` has the signature
`model = fit(X_train, y_train)` and a `predict` has `labels = predict(model, X_test)`
with labels in {0, 1}.
"""

import numpy as np

WORDS_SPAM = ["free", "winner", "prize", "cash", "offer", "click", "urgent", "credit", "bonus", "deal"]
WORDS_HAM = ["meeting", "report", "lunch", "project", "schedule", "thanks", "team", "draft", "review", "agenda"]
V = 300


def make_data(seed: int = 0, n_dev: int = 600, n_test: int = 1400) -> dict:
    """Build a synthetic mailbox as word counts plus two numeric features.

    Planted structure:
    - The vocabulary has 300 words. Ten are spammy (`WORDS_SPAM`), ten are hammy
      (`WORDS_HAM`), and the other 280 are neutral words named `w0` ... `w279`.
      Word positions are shuffled, so the planted words sit at random columns.
    - Every email draws about 30 words (Poisson length). Neutral words follow a
      fixed skewed (Zipf-like) distribution. In spam, each spammy word is 4 times
      likelier than its neutral weight; in ham, each hammy word is 4 times likelier.
    - Two dense features per email: `links` (count-like, mean about 2 in ham) and
      `caps` (fraction of capital letters). In ham they form a blob; in spam they
      form a ring around that blob, so no straight line separates the classes.
    - 40% of emails are spam. The first `n_dev` rows are for development (cross-
      validation, learning curves); the last `n_test` rows are the held-out test set.

    Returns:
        A dict with `counts` (n, 300) float counts, `numeric` (n, 2) raw features
        in their natural, unequal units, `y` (n,) labels, `vocab` (list of 300
        strings), `spam_idx` and `ham_idx` (arrays of planted columns), and `n_dev`.
    """
    rng = np.random.default_rng(seed)
    n = n_dev + n_test
    names = WORDS_SPAM + WORDS_HAM + [f"w{i}" for i in range(V - 20)]
    perm = rng.permutation(V)
    vocab = [None] * V
    for name, col in zip(names, perm):
        vocab[col] = name
    spam_idx = np.sort(perm[:10])
    ham_idx = np.sort(perm[10:20])
    base = np.zeros(V)
    base[perm[20:]] = 1.0 / np.arange(1, V - 19) ** 0.8
    base[perm[:20]] = base[perm[20:]].mean() * 1.5
    base /= base.sum()
    p_spam = base.copy()
    p_spam[spam_idx] *= 4.0
    p_spam /= p_spam.sum()
    p_ham = base.copy()
    p_ham[ham_idx] *= 4.0
    p_ham /= p_ham.sum()

    y = (rng.uniform(size=n) < 0.4).astype(int)
    lengths = rng.poisson(30, size=n)
    counts = np.zeros((n, V))
    for i in range(n):
        counts[i] = rng.multinomial(lengths[i], p_spam if y[i] == 1 else p_ham)

    radius = np.where(y == 1, rng.normal(1.9, 0.6, size=n), np.abs(rng.normal(0.0, 0.9, size=n)))
    angle = rng.uniform(0, 2 * np.pi, size=n)
    u, v = radius * np.cos(angle), radius * np.sin(angle)
    links = np.maximum(2.0 + 1.2 * u, 0.0)  # about 0 to 5 links
    caps = np.clip(0.12 + 0.05 * v, 0.0, 1.0)  # about 0.02 to 0.22
    numeric = np.column_stack([links, caps])
    return {
        "counts": counts, "numeric": numeric, "y": y, "vocab": vocab,
        "spam_idx": spam_idx, "ham_idx": ham_idx, "n_dev": n_dev,
    }


def fit_nb(X: np.ndarray, y: np.ndarray, alpha: float):
    """Fit multinomial naive Bayes with Laplace (add-alpha) smoothing.

    Args:
        X: Word counts, shape (n, V).
        y: Labels in {0, 1}, shape (n,).
        alpha: Smoothing, added to every word count of each class.

    Returns:
        A tuple `(log_theta, log_priors)` with shapes (2, V) and (2,). Row k of
        `np.exp(log_theta)` sums to 1.

    Steps:
        1. For each class k, take the rows `X[y == k]`.
        2. The prior is the class's share of the rows; store its log.
        3. Sum the rows over documents, add `alpha` to every word, divide by the
           smoothed total, and store the log.
    """
    log_theta = np.zeros((2, X.shape[1]))
    log_priors = np.zeros(2)
    for k in (0, 1):
        rows = X[y == k]
        log_priors[k] = np.log(len(rows) / len(y))
        smoothed = rows.sum(axis=0) + alpha
        log_theta[k] = np.log(smoothed / smoothed.sum())
    return log_theta, log_priors


def predict_nb(model, X: np.ndarray) -> np.ndarray:
    """Return the more probable class of each row, as integer labels in {0, 1}.

    Work in log space: the score of class k is `X @ log_theta[k] + log_priors[k]`.
    No normalisation is needed to pick the larger one.

    Steps:
        1. Compute `X @ log_theta.T + log_priors`, shape (n, 2).
        2. Return the column index of the larger score in each row.
    """
    log_theta, log_priors = model
    scores = X @ log_theta.T + log_priors
    return np.argmax(scores, axis=1)


def top_spam_words(model, vocab, n_words: int) -> list:
    """Return the `n_words` words most indicative of spam, strongest first.

    A word's evidence is the log ratio `log_theta[1] - log_theta[0]`.

    Args:
        model: The tuple returned by `fit_nb`.
        vocab: List of V word strings.
        n_words: How many to return.

    Steps:
        1. Subtract the ham row of `log_theta` from the spam row.
        2. Sort the V differences from largest to smallest and keep `n_words`.
        3. Look the winners up in `vocab`.
    """
    ratio = model[0][1] - model[0][0]
    order = np.argsort(-ratio)[:n_words]
    return [vocab[i] for i in order]


def k_fold_indices(n: int, k: int, rng: np.random.Generator) -> list:
    """Shuffle 0..n-1 and cut into k disjoint folds of nearly equal size.

    Returns a list of k integer arrays whose union is `range(n)`.

    Steps:
        1. `rng.permutation(n)` shuffles the row numbers.
        2. `np.array_split(order, k)` cuts them into k nearly equal chunks.
    """
    order = rng.permutation(n)
    return [np.sort(chunk) for chunk in np.array_split(order, k)]


def error_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Given: return the fraction of labels that differ."""
    return float(np.mean(y_true != y_pred))


def cv_error(fit, predict, X: np.ndarray, y: np.ndarray, k: int, seed: int = 0) -> float:
    """Return the mean held-out error rate over k folds.

    Build the folds with `k_fold_indices` and `np.random.default_rng(seed)`, so
    equal inputs give equal folds. For each fold, fit on the other k - 1 folds and
    measure `error_rate` on the held-out one. Average the k fold errors.

    Args:
        fit: Callable `fit(X_train, y_train) -> model`.
        predict: Callable `predict(model, X_test) -> labels`.
        X: Features, shape (n, d).
        y: Labels, shape (n,).
        k: Number of folds.
        seed: Seed for the fold shuffle.

    Steps:
        1. Build the folds once, before the loop.
        2. For each held-out fold, the training rows are all the other rows.
        3. Fit on the training rows only, predict the held-out rows, and score.
    """
    folds = k_fold_indices(len(y), k, np.random.default_rng(seed))
    errors = []
    for held in folds:
        train = np.setdiff1d(np.arange(len(y)), held)
        model = fit(X[train], y[train])
        errors.append(error_rate(y[held], predict(model, X[held])))
    return float(np.mean(errors))


def grid_search_cv(make_fit, predict, X: np.ndarray, y: np.ndarray, settings: list, k: int):
    """Score every setting by cross-validation on the same folds and pick the best.

    Args:
        make_fit: Callable `make_fit(setting) -> fit`, where `fit` is as in `cv_error`.
        predict: Callable `predict(model, X_test) -> labels`.
        X: Features, shape (n, d).
        y: Labels, shape (n,).
        settings: List of settings, any objects `make_fit` accepts.
        k: Number of folds.

    Returns:
        A tuple `(best_setting, errors)`: the setting with the lowest cv error
        (the first one on ties) and an array of one error per setting.

    Steps:
        1. Call `cv_error(make_fit(s), ...)` for each setting `s`. Every call uses
           the same default seed, so every setting sees the same folds.
        2. Pick the setting at the lowest error with `np.argmin`.
    """
    errors = np.array([cv_error(make_fit(s), predict, X, y, k) for s in settings])
    return settings[int(np.argmin(errors))], errors


def standardise(X_train: np.ndarray, X_other: np.ndarray) -> np.ndarray:
    """Rescale `X_other` with the mean and standard deviation of `X_train`.

    Returns `(X_other - mean) / std`, column by column, where mean and std are
    computed on `X_train` only, shape (d,). Returns shape of `X_other`.

    Steps:
        1. `X_train.mean(axis=0)` and `X_train.std(axis=0)` have shape (d,).
        2. Subtract and divide; broadcasting handles the rows.
    """
    return (X_other - X_train.mean(axis=0)) / X_train.std(axis=0)


def rbf_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float) -> np.ndarray:
    """Given, from "Kernels: a similarity score that hides a bigger feature space"."""
    sq = (X1**2).sum(axis=1)[:, None] + (X2**2).sum(axis=1)[None, :] - 2.0 * X1 @ X2.T
    return np.exp(-gamma * np.maximum(sq, 0.0))


def kernel_ridge_fit(K: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Given, from "Kernel ridge regression: a weighted sum of similarities"."""
    L = np.linalg.cholesky(K + lam * np.eye(K.shape[0]))
    return np.linalg.solve(L.T, np.linalg.solve(L, y))


def fit_krr_classifier(X: np.ndarray, y: np.ndarray, gamma: float, lam: float):
    """Fit RBF kernel ridge regression on targets -1 (ham) and +1 (spam).

    Standardise the features with `standardise(X, X)` before building the kernel,
    so both numeric features count equally in the distance.

    Returns:
        A model of your choice, which `predict_krr` accepts. It must keep the raw
        training features so `predict_krr` can standardise queries the same way.

    Steps:
        1. Standardise, then build the Gram matrix with `rbf_kernel(Z, Z, gamma)`.
        2. Turn the labels {0, 1} into targets {-1, +1}.
        3. Solve for the weights with `kernel_ridge_fit(K, targets, lam)`.
    """
    Z = standardise(X, X)
    alpha = kernel_ridge_fit(rbf_kernel(Z, Z, gamma), 2.0 * y - 1.0, lam)
    return X, alpha, gamma


def predict_krr(model, X: np.ndarray) -> np.ndarray:
    """Return labels in {0, 1}: spam where the kernel ridge prediction is above 0.

    Standardise `X` with the training mean and standard deviation, not its own.

    Steps:
        1. Standardise the training features and the queries with the same
           training statistics.
        2. Build the cross-kernel between queries and training points, shape (m, n).
        3. Multiply by the weights and threshold at 0.
    """
    X_train, alpha, gamma = model
    K = rbf_kernel(standardise(X_train, X), standardise(X_train, X_train), gamma)
    return (K @ alpha > 0).astype(int)


def learning_curve(fit, predict, X: np.ndarray, y: np.ndarray, sizes, n_val: int):
    """Return training and validation error rates at growing training sizes.

    Hold out the last `n_val` rows as the validation set. For each size, fit on
    the first `size` of the remaining rows and record `error_rate` on those rows
    and on the validation rows.

    Returns:
        A tuple `(train_errors, val_errors)`, each of shape (len(sizes),).

    Steps:
        1. Split off the validation rows once.
        2. Loop over sizes; each fit uses a prefix of the pool, so training sets nest.
    """
    pool = len(y) - n_val
    Xv, yv = X[pool:], y[pool:]
    train_errors, val_errors = [], []
    for size in sizes:
        model = fit(X[:size], y[:size])
        train_errors.append(error_rate(y[:size], predict(model, X[:size])))
        val_errors.append(error_rate(yv, predict(model, Xv)))
    return np.array(train_errors), np.array(val_errors)


def more_data_would_help(train_errors, val_errors, gap_tol: float = 0.02) -> bool:
    """Return True when the curves say more mail should lower the error.

    Two conditions, both required: the gap `val_errors[-1] - train_errors[-1]`
    exceeds `gap_tol`, and the validation error at the largest size is below the
    validation error at the smallest size (the curve is still falling overall).

    Steps:
        1. Compute the final gap.
        2. Return the `and` of the two conditions as a Python bool.
    """
    gap = val_errors[-1] - train_errors[-1]
    return bool(gap > gap_tol and val_errors[-1] < val_errors[0])


def select_and_test(data: dict, k: int = 5) -> dict:
    """Run the whole pipeline and return an honest score.

    1. Use only the first `data["n_dev"]` rows for selection.
    2. Choose the NB smoothing `alpha` from [0.01, 0.1, 0.3, 1, 3, 10, 30] and the
       kernel pair `(gamma, lam)` from gamma in [0.1, 0.3, 1, 3] and lam in
       [0.01, 0.1, 1] by `grid_search_cv` with k folds (NB uses word counts,
       the kernel model uses the numeric features).
    3. The winner is the model with the lower best cv error (NB on ties). Refit
       it on all dev rows and score it once on the test rows.

    Returns:
        A dict with `nb_alpha`, `nb_cv`, `krr_gamma`, `krr_lam`, `krr_cv` (best
        cv errors as floats), `winner` ("nb" or "krr"), `best_cv` (the winner's
        cv error), `test_error` (float), and `gap` (`test_error - best_cv`).

    Steps:
        1. Slice `[:n_dev]` for every selection step and `[n_dev:]` for the test.
        2. `grid_search_cv` once for NB (a `make_fit` that closes over alpha) and
           once for the kernel model (a `make_fit` that closes over `(gamma, lam)`).
        3. Compare the two best cv errors, refit the winner on all dev rows, and
           touch the test rows exactly once.
    """
    d = data["n_dev"]
    Xw, Xn, y = data["counts"], data["numeric"], data["y"]
    alpha, nb_errs = grid_search_cv(
        lambda a: (lambda X, t: fit_nb(X, t, a)), predict_nb, Xw[:d], y[:d],
        [0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0], k)
    grid = [(g, l) for g in (0.1, 0.3, 1.0, 3.0) for l in (0.01, 0.1, 1.0)]
    (gamma, lam), krr_errs = grid_search_cv(
        lambda s: (lambda X, t: fit_krr_classifier(X, t, s[0], s[1])), predict_krr,
        Xn[:d], y[:d], grid, k)
    nb_cv, krr_cv = float(nb_errs.min()), float(krr_errs.min())
    if nb_cv <= krr_cv:
        winner, best_cv = "nb", nb_cv
        model = fit_nb(Xw[:d], y[:d], alpha)
        pred = predict_nb(model, Xw[d:])
    else:
        winner, best_cv = "krr", krr_cv
        model = fit_krr_classifier(Xn[:d], y[:d], gamma, lam)
        pred = predict_krr(model, Xn[d:])
    test_error = error_rate(y[d:], pred)
    return {
        "nb_alpha": float(alpha), "nb_cv": nb_cv, "krr_gamma": float(gamma), "krr_lam": float(lam),
        "krr_cv": krr_cv, "winner": winner, "best_cv": best_cv,
        "test_error": test_error, "gap": test_error - best_cv,
    }


if __name__ == "__main__":
    data = make_data(0)
    d = data["n_dev"]
    print("emails:", len(data["y"]), " dev:", d, " spam share:", round(float(data["y"].mean()), 3))
    result = select_and_test(data)
    for key, value in result.items():
        print(f"{key:>10}: {value}")
    model = fit_nb(data["counts"][:d], data["y"][:d], result["nb_alpha"])
    print("top spam words:", top_spam_words(model, data["vocab"], 10))
