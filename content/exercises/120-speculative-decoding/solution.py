"""Reference solution: the accept-or-replace rule of speculative decoding."""

import numpy as np


def expected_accept_rate(p: np.ndarray, q: np.ndarray) -> float:
    # q[x] * min(1, p[x] / q[x]) is min(p[x], q[x]) wherever q[x] > 0.
    return float(np.minimum(np.asarray(p, float), np.asarray(q, float)).sum())


def residual_distribution(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    leftover = np.maximum(p - np.asarray(q, dtype=float), 0.0)
    total = leftover.sum()
    if total <= 0.0:
        return p.copy()
    return leftover / total


def speculative_accept(
    p: np.ndarray, q: np.ndarray, draft_token: int, rng: np.random.Generator
) -> int:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if rng.uniform() < min(1.0, p[draft_token] / q[draft_token]):
        return int(draft_token)
    residual = residual_distribution(p, q)
    return int(rng.choice(len(p), p=residual))


if __name__ == "__main__":
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.1, 0.1, 0.8])
    rng = np.random.default_rng(0)

    print(f"acceptance rate: {expected_accept_rate(p, q):.3f}")
    print(f"residual: {np.round(residual_distribution(p, q), 4)}")

    draws = [speculative_accept(p, q, int(rng.choice(3, p=q)), rng) for _ in range(20_000)]
    print(f"output frequencies: {np.round(np.bincount(draws, minlength=3) / len(draws), 3)}")
    print(f"target p:           {p}")
