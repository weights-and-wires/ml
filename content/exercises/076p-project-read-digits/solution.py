"""Reference solution: train a digit reader from scratch, in numpy only."""

import numpy as np

SIZE = 10
NUM_CLASSES = 10

# Ten stroke templates. Each digit is a list of (x, y) corners in a unit square,
# x to the right and y downward; the renderer joins the corners with line segments.
STROKES = {
    0: [(.25, .1), (.75, .1), (.9, .3), (.9, .7), (.75, .9), (.25, .9), (.1, .7), (.1, .3), (.25, .1)],
    1: [(.3, .28), (.55, .1), (.55, .9)],
    2: [(.15, .25), (.3, .1), (.7, .1), (.85, .25), (.8, .45), (.15, .9), (.9, .9)],
    3: [(.15, .15), (.8, .15), (.5, .45), (.85, .65), (.75, .9), (.15, .85)],
    4: [(.7, .9), (.7, .1), (.1, .65), (.9, .65)],
    5: [(.85, .1), (.2, .1), (.15, .45), (.7, .45), (.85, .65), (.7, .9), (.15, .85)],
    6: [(.8, .1), (.3, .3), (.15, .6), (.3, .9), (.7, .9), (.85, .7), (.7, .5), (.2, .55)],
    7: [(.1, .1), (.9, .1), (.4, .9)],
    8: [(.5, .1), (.2, .22), (.5, .48), (.8, .75), (.5, .9), (.2, .75), (.5, .48), (.8, .22), (.5, .1)],
    9: [(.85, .45), (.5, .55), (.2, .4), (.35, .12), (.7, .12), (.85, .3), (.8, .9)],
}


def _distance_to_polyline(px, py, corners):
    """Distance from every pixel centre (px, py) to the nearest segment of a polyline."""
    best = np.full(px.shape, np.inf)
    for (x0, y0), (x1, y1) in zip(corners[:-1], corners[1:]):
        dx, dy = x1 - x0, y1 - y0
        t = ((px - x0) * dx + (py - y0) * dy) / (dx * dx + dy * dy + 1e-12)
        t = np.clip(t, 0.0, 1.0)
        best = np.minimum(best, np.hypot(px - (x0 + t * dx), py - (y0 + t * dy)))
    return best


def render_digit(digit, rng, noise=0.25):
    """Draw one noisy 10 x 10 glyph of `digit` and return it as a flat array of 100 pixels."""
    centres = (np.arange(SIZE) + 0.5) / SIZE
    px, py = np.meshgrid(centres, centres)
    corners = [(x + rng.normal(scale=0.03), y + rng.normal(scale=0.03)) for x, y in STROKES[digit]]
    shift = rng.integers(-1, 2, size=2) / SIZE
    corners = [(x * 0.9 + 0.05 + shift[0], y * 0.9 + 0.05 + shift[1]) for x, y in corners]
    thickness = rng.uniform(0.07, 0.15)
    ink = np.clip(1.5 - _distance_to_polyline(px, py, corners) / thickness, 0.0, 1.0)
    return (ink + noise * rng.normal(size=ink.shape)).ravel()


def make_data(seed=0, n_train=1500, n_val=500, n_test=1000):
    """Return (X_train, y_train, X_val, y_val, X_test, y_test) of noisy digit glyphs."""
    rng = np.random.default_rng(seed)

    def split(n):
        y = np.arange(n) % NUM_CLASSES
        rng.shuffle(y)
        return np.stack([render_digit(d, rng) for d in y]), y

    X_train, y_train = split(n_train)
    X_val, y_val = split(n_val)
    X_test, y_test = split(n_test)
    return X_train, y_train, X_val, y_val, X_test, y_test


# Written in the chapter "Xavier and He initialisation".
def he_init(layer_sizes, rng):
    params = []
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        params.append({"W": rng.normal(scale=np.sqrt(2.0 / n_in), size=(n_in, n_out)),
                       "b": np.zeros(n_out)})
    return params


def forward(params, X, p_drop=0.0, rng=None, training=False):
    h = X
    cache = []
    last = len(params) - 1
    for i, layer in enumerate(params):
        z = h @ layer["W"] + layer["b"]
        if i == last:
            cache.append({"h_in": h, "z": z, "mask": None})
            return z, cache
        a = np.maximum(z, 0.0)
        if training and p_drop > 0.0:
            mask = (rng.random(a.shape) >= p_drop) / (1.0 - p_drop)
        else:
            mask = None
        h_out = a if mask is None else a * mask
        cache.append({"h_in": h, "z": z, "mask": mask})
        h = h_out


def smoothed_cross_entropy(logits, y, eps):
    n, k = logits.shape
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_p = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    target = (1.0 - eps) * np.eye(k)[y] + eps / k
    loss = float(-(target * log_p).sum(axis=1).mean())
    return loss, (np.exp(log_p) - target) / n


def backward(params, cache, dlogits):
    grads = [None] * len(params)
    d = dlogits
    for i in reversed(range(len(params))):
        c = cache[i]
        grads[i] = {"W": c["h_in"].T @ d, "b": d.sum(axis=0)}
        if i > 0:
            d = d @ params[i]["W"].T
            prev = cache[i - 1]
            if prev["mask"] is not None:
                d = d * prev["mask"]
            d = d * (prev["z"] > 0.0)
    return grads


def loss_and_grads(params, X, y, eps=0.0, p_drop=0.0, rng=None, training=True):
    logits, cache = forward(params, X, p_drop, rng, training)
    loss, dlogits = smoothed_cross_entropy(logits, y, eps)
    return loss, backward(params, cache, dlogits)


def gradient_check(params, X, y, eps=0.0, grads=None, h=1e-5):
    if grads is None:
        _, grads = loss_and_grads(params, X, y, eps, training=False)
    worst = 0.0
    for i, layer in enumerate(params):
        for key in ("W", "b"):
            array = layer[key]
            numeric = np.zeros_like(array)
            for index in np.ndindex(*array.shape):
                original = array[index]
                array[index] = original + h
                high = loss_and_grads(params, X, y, eps, training=False)[0]
                array[index] = original - h
                low = loss_and_grads(params, X, y, eps, training=False)[0]
                array[index] = original
                numeric[index] = (high - low) / (2.0 * h)
            analytic = grads[i][key]
            scale = np.linalg.norm(analytic) + np.linalg.norm(numeric)
            gap = np.linalg.norm(analytic - numeric)
            worst = max(worst, float(gap / scale) if scale > 0 else 0.0)
    return worst


# Written in the chapter "AdamW: decaying weights the right way".
def adamw_step(w, grad, state, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
    t = state.get("t", 0) + 1
    m = beta1 * state.get("m", np.zeros_like(w)) + (1.0 - beta1) * grad
    v = beta2 * state.get("v", np.zeros_like(w)) + (1.0 - beta2) * grad**2
    m_hat = m / (1.0 - beta1**t)
    v_hat = v / (1.0 - beta2**t)
    new_w = w - lr * m_hat / (np.sqrt(v_hat) + eps) - lr * weight_decay * w
    return new_w, {"m": m, "v": v, "t": t}


def adamw_update(params, grads, states, lr, weight_decay=0.01):
    new_params, new_states = [], []
    for layer, grad, state in zip(params, grads, states):
        W, state_w = adamw_step(layer["W"], grad["W"], state["W"], lr, weight_decay=weight_decay)
        b, state_b = adamw_step(layer["b"], grad["b"], state["b"], lr, weight_decay=0.0)
        new_params.append({"W": W, "b": b})
        new_states.append({"W": state_w, "b": state_b})
    return new_params, new_states


def lr_schedule(step, warmup, total, peak):
    if step < warmup:
        return peak * step / warmup
    progress = min((step - warmup) / max(total - warmup, 1), 1.0)
    return peak * 0.5 * (1.0 + np.cos(np.pi * progress))


def predict(params, X):
    logits, _ = forward(params, X, training=False)
    return logits.argmax(axis=1)


def train(data, hidden=(64, 32), epochs=60, batch_size=64, peak_lr=3e-3, weight_decay=0.01,
          p_drop=0.2, smoothing=0.1, patience=10, seed=0):
    X_train, y_train, X_val, y_val = data[:4]
    rng = np.random.default_rng(seed)
    params = he_init([X_train.shape[1], *hidden, NUM_CLASSES], rng)
    states = [{"W": {}, "b": {}} for _ in params]
    n = len(y_train)
    steps_per_epoch = int(np.ceil(n / batch_size))
    total, warmup = epochs * steps_per_epoch, max(1, int(0.05 * epochs * steps_per_epoch))
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val, best_epoch, best_params, step = np.inf, 0, params, 0
    for epoch in range(epochs):
        order = rng.permutation(n)
        running = 0.0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            lr = lr_schedule(step, warmup, total, peak_lr)
            loss, grads = loss_and_grads(params, X_train[idx], y_train[idx], smoothing, p_drop, rng)
            params, states = adamw_update(params, grads, states, lr, weight_decay)
            running += loss * len(idx)
            step += 1
        logits, _ = forward(params, X_val, training=False)
        val_loss, _ = smoothed_cross_entropy(logits, y_val, 0.0)
        history["train_loss"].append(running / n)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(float((logits.argmax(axis=1) == y_val).mean()))
        history["lr"].append(lr)
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            best_params = [{"W": p["W"].copy(), "b": p["b"].copy()} for p in params]
        elif epoch - best_epoch >= patience:
            break
    history["best_epoch"] = best_epoch
    return best_params, history


if __name__ == "__main__":
    import time

    data = make_data(0)
    X_train, y_train, X_val, y_val, X_test, y_test = data
    print("data:", X_train.shape, X_val.shape, X_test.shape)
    tiny = he_init([100, 8, 10], np.random.default_rng(0))
    print("gradient check relative error:", gradient_check(tiny, X_train[:20], y_train[:20], eps=0.1))
    start = time.time()
    params, history = train(data)
    print(f"epochs run {len(history['val_loss'])}, best epoch {history['best_epoch']}")
    print(f"MLP test accuracy {(predict(params, X_test) == y_test).mean():.3f} in {time.time() - start:.1f}s")
    linear, _ = train(data, hidden=(), p_drop=0.0)
    print(f"linear baseline {(predict(linear, X_test) == y_test).mean():.3f}")
