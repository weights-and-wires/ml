"""Reference solution: temperature, filtering, and sampling one token."""

import numpy as np


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(logits, dtype=float) / temperature
    exp = np.exp(scaled - scaled.max())
    return exp / exp.sum()


def _top_k(probs: np.ndarray, k: int) -> np.ndarray:
    keep = np.argsort(-probs, kind="stable")[:k]
    out = np.zeros_like(probs)
    out[keep] = probs[keep]
    return out / out.sum()


def _nucleus(probs: np.ndarray, p: float) -> np.ndarray:
    order = np.argsort(-probs, kind="stable")
    cumulative = np.cumsum(probs[order])
    # The first position whose running total reaches p is the last one kept.
    # Rounding can leave the final total just under 1.0, so clamp the index.
    cutoff = min(int(np.searchsorted(cumulative, p, side="left")), len(probs) - 1)
    out = np.zeros_like(probs)
    keep = order[: cutoff + 1]
    out[keep] = probs[keep]
    return out / out.sum()


def filter_probs(
    probs: np.ndarray, top_k: int | None = None, top_p: float | None = None
) -> np.ndarray:
    out = np.asarray(probs, dtype=float).copy()
    if top_k is not None:
        out = _top_k(out, top_k)
    if top_p is not None:
        out = _nucleus(out, top_p)
    return out


def sample_token(
    logits: np.ndarray,
    rng: np.random.Generator,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> int:
    if temperature == 0:
        return int(np.argmax(logits))
    probs = filter_probs(apply_temperature(logits, temperature), top_k, top_p)
    return int(rng.choice(len(probs), p=probs))


if __name__ == "__main__":
    logits = np.array([0.5, -2.0, 3.5, 3.4, -1.0])
    rng = np.random.default_rng(0)

    for tau in (0.5, 1.0, 2.0):
        print(f"temperature {tau}: {np.round(apply_temperature(logits, tau), 4)}")

    probs = apply_temperature(logits, 2.0)
    print(f"top_p 0.9 at temperature 2: {np.round(filter_probs(probs, top_p=0.9), 4)}")

    draws = [sample_token(logits, rng, temperature=2.0, top_p=0.9) for _ in range(5000)]
    print(f"frequencies of 5000 draws: {np.round(np.bincount(draws, minlength=5) / 5000, 3)}")
    print(f"greedy choice: {sample_token(logits, rng, temperature=0)}")
