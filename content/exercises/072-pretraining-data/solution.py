"""Reference solution: dedup, filter, and mix pretraining text."""

from __future__ import annotations

import hashlib

import numpy as np

PRIME = 2_147_483_647  # 2**31 - 1, so a * x + b stays inside int64


# ---- Provided helpers (identical in the starter) ---------------------------


def normalise(text: str) -> str:
    """Lowercase the text and collapse every run of whitespace to one space."""
    return " ".join(text.lower().split())


def stable_id(shingle: str) -> int:
    """Map a shingle to an integer in [0, PRIME). Python's hash() changes between runs, so use blake2b."""
    digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % PRIME


def hash_params(n_hashes: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (a, b), two int64 arrays of length n_hashes. Hash i is x -> (a[i] * x + b[i]) % PRIME."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, PRIME, size=n_hashes, dtype=np.int64)
    b = rng.integers(0, PRIME, size=n_hashes, dtype=np.int64)
    return a, b


# ---- Exact duplicates and shingles -----------------------------------------


def shingles(text: str, k: int) -> set[str]:
    words = normalise(text).split()
    if not words:
        return set()
    if len(words) < k:
        return {" ".join(words)}
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def jaccard(a: set, b: set) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def exact_dedup(docs: list[str]) -> list[int]:
    seen: set[str] = set()
    kept = []
    for i, doc in enumerate(docs):
        digest = hashlib.sha256(normalise(doc).encode("utf-8")).hexdigest()
        if digest not in seen:
            seen.add(digest)
            kept.append(i)
    return kept


# ---- MinHash and LSH --------------------------------------------------------


def minhash_signature(shingle_set: set[str], n_hashes: int, seed: int) -> np.ndarray:
    a, b = hash_params(n_hashes, seed)
    if not shingle_set:
        return np.full(n_hashes, PRIME, dtype=np.int64)
    ids = np.array([stable_id(s) for s in shingle_set], dtype=np.int64)
    hashed = (a[:, None] * ids[None, :] + b[:, None]) % PRIME  # (n_hashes, n_shingles)
    return hashed.min(axis=1)


def estimate_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    return float(np.mean(sig_a == sig_b))


def lsh_candidate_prob(s, r: int, b: int):
    s = np.asarray(s, dtype=float)
    return 1.0 - (1.0 - s**r) ** b


def lsh_candidates(sigs: np.ndarray, b: int, r: int) -> set[tuple[int, int]]:
    n = sigs.shape[0]
    pairs: set[tuple[int, int]] = set()
    for band in range(b):
        buckets: dict[tuple, list[int]] = {}
        for i in range(n):
            key = tuple(sigs[i, band * r : (band + 1) * r].tolist())
            buckets.setdefault(key, []).append(i)
        for members in buckets.values():
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    pairs.add((members[x], members[y]))
    return pairs


def dedup(
    docs: list[str],
    threshold: float = 0.7,
    k: int = 3,
    n_hashes: int = 128,
    b: int = 32,
    seed: int = 0,
) -> list[int]:
    r = n_hashes // b
    sigs = np.stack([minhash_signature(shingles(d, k), n_hashes, seed) for d in docs])
    parent = list(range(len(docs)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in sorted(lsh_candidates(sigs, b, r)):
        if estimate_jaccard(sigs[i], sigs[j]) >= threshold:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)  # the root is the earliest document
    return [i for i in range(len(docs)) if find(i) == i]


# ---- Mixing -----------------------------------------------------------------


def mix_weights(sizes, T: float) -> np.ndarray:
    n = np.asarray(sizes, dtype=float)
    w = n ** (1.0 / T)
    return w / w.sum()


def epochs_per_domain(sizes, weights, budget: float) -> np.ndarray:
    return budget * np.asarray(weights, dtype=float) / np.asarray(sizes, dtype=float)


# ---- Quality ----------------------------------------------------------------

ALPHABET = 128


def _codes(text: str) -> np.ndarray:
    return np.array([min(ord(c), ALPHABET - 1) for c in text], dtype=np.int64)


def train_char_bigram(texts: list[str], alpha: float = 1.0) -> np.ndarray:
    counts = np.zeros((ALPHABET, ALPHABET))
    for text in texts:
        codes = _codes(text)
        np.add.at(counts, (codes[:-1], codes[1:]), 1.0)
    smoothed = counts + alpha
    return np.log(smoothed / smoothed.sum(axis=1, keepdims=True))


def quality_score(text: str, logp: np.ndarray) -> float:
    codes = _codes(text)
    if len(codes) < 2:
        return float(logp.min())
    return float(logp[codes[:-1], codes[1:]].mean())


if __name__ == "__main__":
    a = shingles("the cat sat on the mat", 2)
    b = shingles("the cat sat on the rug", 2)
    print("jaccard:", jaccard(a, b))
    print("estimate:", estimate_jaccard(minhash_signature(a, 256, 0), minhash_signature(b, 256, 0)))
    print("mix_weights:", mix_weights([900, 90, 10], 2.0))
