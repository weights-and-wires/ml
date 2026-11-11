"""Pretraining data: exact dedup, MinHash, LSH, a quality score, and domain mixing.

You write twelve functions. `shingles`, `jaccard`, and `exact_dedup` find
exact and measurable overlap. `minhash_signature` and `estimate_jaccard` shrink
a set to a short signature whose matches estimate the overlap.
`lsh_candidate_prob` and `lsh_candidates` find likely near-duplicate pairs
without comparing every pair, and `dedup` ties the pieces together. `mix_weights`
and `epochs_per_domain` turn domain sizes into sampling shares and repeat counts.
`train_char_bigram` and `quality_score` separate clean text from noise.

`normalise`, `stable_id`, and `hash_params` are provided. Hash number i of a
MinHash is x -> (a[i] * x + b[i]) % PRIME, applied to `stable_id` of each
shingle, and `hash_params(n_hashes, seed)` returns the arrays a and b.
"""

from __future__ import annotations

import hashlib

import numpy as np

PRIME = 2_147_483_647  # 2**31 - 1, so a * x + b stays inside int64


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


def shingles(text: str, k: int) -> set[str]:
    """Return the set of word k-grams of the normalised text.

    Each shingle is k consecutive words joined by one space. A text with fewer
    than k words gives one shingle holding all its words. An empty text gives
    the empty set.
    """
    raise NotImplementedError("write shingles")


def jaccard(a: set, b: set) -> float:
    """Return |a & b| / |a | b|. Two empty sets give 0.0."""
    raise NotImplementedError("write jaccard")


def exact_dedup(docs: list[str]) -> list[int]:
    """Return the indices to keep, in order: the first document of each distinct normalised text.

    Hash the normalised text and remember the hashes you have seen.
    """
    raise NotImplementedError("write exact_dedup")


def minhash_signature(shingle_set: set[str], n_hashes: int, seed: int) -> np.ndarray:
    """Return an int64 array of shape (n_hashes,).

    Entry i is the minimum over the set of hash i of stable_id(shingle). An
    empty set returns an array filled with PRIME.
    """
    raise NotImplementedError("write minhash_signature")


def estimate_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Return the fraction of positions where the two signatures agree."""
    raise NotImplementedError("write estimate_jaccard")


def lsh_candidate_prob(s, r: int, b: int):
    """Return the probability that two documents with similarity s share a bucket in at least one of b bands of r rows.

    Accepts a float or an array of similarities and returns the same shape.
    """
    raise NotImplementedError("write lsh_candidate_prob")


def lsh_candidates(sigs: np.ndarray, b: int, r: int) -> set[tuple[int, int]]:
    """Return every pair (i, j), i < j, whose signatures agree on all r rows of at least one band.

    sigs has shape (n_docs, b * r). Band number q uses columns q * r to (q + 1) * r - 1.
    """
    raise NotImplementedError("write lsh_candidates")


def dedup(
    docs: list[str],
    threshold: float = 0.7,
    k: int = 3,
    n_hashes: int = 128,
    b: int = 32,
    seed: int = 0,
) -> list[int]:
    """Return the sorted indices of documents to keep.

    Shingle every document, sign it, find candidate pairs with b bands of
    n_hashes // b rows, and link a pair when its estimated Jaccard similarity is
    at least threshold. Linked documents form a cluster, even through a chain.
    Keep the earliest document of each cluster.
    """
    raise NotImplementedError("write dedup")


def mix_weights(sizes, T: float) -> np.ndarray:
    """Return sampling shares p_i proportional to sizes[i] ** (1 / T), summing to 1.

    T = 1 is proportional to size. A large T approaches equal shares.
    """
    raise NotImplementedError("write mix_weights")


def epochs_per_domain(sizes, weights, budget: float) -> np.ndarray:
    """Return how many passes over each domain a run of `budget` tokens makes.

    Domain i supplies budget * weights[i] tokens out of sizes[i] available.
    """
    raise NotImplementedError("write epochs_per_domain")


ALPHABET = 128  # ASCII; any character above 127 maps to 127


def train_char_bigram(texts: list[str], alpha: float = 1.0) -> np.ndarray:
    """Return a (128, 128) table of log probabilities, logp[i, j] = log P(next = j | current = i).

    Count every adjacent character pair across the texts (map each character to
    min(ord(c), 127)), add alpha to every count, and normalise each row to sum to 1.
    """
    raise NotImplementedError("write train_char_bigram")


def quality_score(text: str, logp: np.ndarray) -> float:
    """Return the mean log probability per adjacent character pair of the text.

    A text with fewer than two characters returns logp.min().
    """
    raise NotImplementedError("write quality_score")


if __name__ == "__main__":
    a = shingles("the cat sat on the mat", 2)
    b = shingles("the cat sat on the rug", 2)
    print("jaccard:", jaccard(a, b))
    print("estimate:", estimate_jaccard(minhash_signature(a, 256, 0), minhash_signature(b, 256, 0)))
    print("mix_weights:", mix_weights([900, 90, 10], 2.0))
