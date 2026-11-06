"""Reference solution: the KL leash and the best policy."""

import numpy as np


def kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    # Replace p by 1 where it is 0 so the log is never taken of 0; the where
    # then sets those terms to 0, which is the right value for 0 * log 0.
    safe_p = np.where(p > 0, p, 1.0)
    terms = np.where(p > 0, p * np.log(safe_p / q), 0.0)
    return np.sum(terms, axis=-1)


def optimal_policy(ref_logprobs: np.ndarray, rewards: np.ndarray, beta: float) -> np.ndarray:
    logits = np.asarray(ref_logprobs, dtype=float) + np.asarray(rewards, dtype=float) / beta
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    unnormalised = np.exp(logits)
    return unnormalised / np.sum(unnormalised, axis=-1, keepdims=True)


def penalised_objective(
    probs: np.ndarray, ref_probs: np.ndarray, rewards: np.ndarray, beta: float
) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    expected = np.sum(probs * np.asarray(rewards, dtype=float), axis=-1)
    return expected - beta * kl_divergence(probs, ref_probs)


if __name__ == "__main__":
    ref = np.array([0.5, 0.3, 0.2])
    rewards = np.array([1.0, 0.0, 0.0])
    beta = 1.0

    best = optimal_policy(np.log(ref), rewards, beta)
    print("best policy:", best.round(4))
    for name, p in [
        ("reference", ref),
        ("[0.7, 0.2, 0.1]", np.array([0.7, 0.2, 0.1])),
        ("best", best),
        ("all on helpful", np.array([1.0, 0.0, 0.0])),
    ]:
        print(f"{name:>16}: KL {kl_divergence(p, ref):.4f}, score {penalised_objective(p, ref, rewards, beta):.4f}")
    print("beta * log Z:", round(beta * np.log(np.sum(ref * np.exp(rewards / beta))), 4))
