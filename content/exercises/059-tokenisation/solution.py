"""Reference solution for the tokenisation exercise."""

import re

_PRETOKEN_PATTERN = re.compile(r" ?[^\s]+|\s+")


def _pretokenize(text: str) -> list[str]:
    """Split text into pieces that join back into the exact input."""
    return _PRETOKEN_PATTERN.findall(text)


# ---------------------------------------------------------------------------
# The functions the checks call.
# ---------------------------------------------------------------------------


def text_to_bytes(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def unknown_rate(text: str, word_vocab: set[str]) -> float:
    words = text.split()
    if not words:
        return 0.0
    missing = sum(1 for w in words if w not in word_vocab)
    return missing / len(words)


def get_pair_counts(token_lists: list[list[int]]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for tokens in token_lists:
        for a, b in zip(tokens, tokens[1:]):
            counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts


def merge_pair(
    token_lists: list[list[int]], pair: tuple[int, int], new_id: int
) -> list[list[int]]:
    result = []
    for tokens in token_lists:
        merged = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                merged.append(new_id)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        result.append(merged)
    return result


def train_bpe(corpus: str, vocab_size: int) -> tuple[list[tuple[int, int]], dict[int, bytes]]:
    token_lists = [list(piece.encode("utf-8")) for piece in _pretokenize(corpus)]
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[int, int]] = []

    while 256 + len(merges) < vocab_size:
        counts = get_pair_counts(token_lists)
        if not counts:
            break
        best = min(counts, key=lambda p: (-counts[p], p))
        new_id = 256 + len(merges)
        merges.append(best)
        vocab[new_id] = vocab[best[0]] + vocab[best[1]]
        token_lists = merge_pair(token_lists, best, new_id)
    return merges, vocab


def apply_merges_to_word(word_bytes: list[int], merges: list[tuple[int, int]]) -> list[int]:
    ids = list(word_bytes)
    for i, pair in enumerate(merges):
        ids = merge_pair([ids], pair, 256 + i)[0]
    return ids


def encode(text: str, merges: list[tuple[int, int]]) -> list[int]:
    ids: list[int] = []
    for piece in _pretokenize(text):
        ids.extend(apply_merges_to_word(list(piece.encode("utf-8")), merges))
    return ids


def decode(ids: list[int], vocab: dict[int, bytes]) -> str:
    return b"".join(vocab[i] for i in ids).decode("utf-8")


# ---------------------------------------------------------------------------
# Provided for the measurements: the bytes of every id.
# ---------------------------------------------------------------------------


def build_vocab(merges: list[tuple[int, int]]) -> dict[int, bytes]:
    """Return the bytes of every id: the 256 byte values, then one entry per merge."""
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for i, (a, b) in enumerate(merges):
        vocab[256 + i] = vocab[a] + vocab[b]
    return vocab


def fertility(text: str, merges: list[tuple[int, int]]) -> float:
    return len(encode(text, merges)) / len(text.split())


def tokens_per_char(text: str, merges: list[tuple[int, int]]) -> float:
    return len(encode(text, merges)) / len(text)


def digit_token_lengths(text: str, merges: list[tuple[int, int]]) -> list[int]:
    vocab = build_vocab(merges)
    lengths = []
    for i in encode(text, merges):
        digits = sum(1 for byte in vocab[i] if 48 <= byte <= 57)
        if digits > 0:
            lengths.append(digits)
    return lengths


if __name__ == "__main__":
    sample = "héllo 世界 🎉"
    ids = text_to_bytes(sample)
    print(f"{sample!r}: {len(sample)} characters, {len(ids)} bytes")
    word_vocab = {"the", "cat", "sat"}
    text = "the cat sat on the mat"
    print(f"unknown rate of a 3-word list on {text!r}: {unknown_rate(text, word_vocab):.2f}")
    n = 5700
    print(f"attention pairs for {n} bytes: {n * n:,}; for 1000 words: {1000 * 1000:,}")

    merges, vocab = train_bpe("aaabdaaabac", 259)
    print("\nmerges:", merges)
    for i, pair in enumerate(merges):
        print(f"  id {256 + i} = {pair} = {vocab[256 + i]!r}")
    print("ids:", encode("aaabdaaabac", merges), "(expected [258, 100, 258, 97, 99])")
    print("reversed merge order:", encode("aaabdaaabac", list(reversed(merges))))

    english = (
        "the cat sat on the mat. the dog sat on the log. the cat and the dog are friends. "
        "a cat likes to sit in the sun, and a dog likes to run in the park. "
        "the children are playing in the garden, and the teacher is reading a book. "
    )
    en_sentence = "the cat sat on the mat."
    hi_sentence = "बिल्ली चटाई पर बैठी थी।"
    merges, vocab = train_bpe(english, 256 + 60)
    for text in (en_sentence, "héllo 世界 🎉"):
        ids = encode(text, merges)
        print(f"{text!r}: {len(text.encode('utf-8'))} bytes -> {len(ids)} ids,",
              "round trip ok:", decode(ids, vocab) == text)

    print("\nfertility of 'the quick fox' with no merges:", round(fertility("the quick fox", []), 2))
    for name, text in (("English", en_sentence), ("Hindi", hi_sentence)):
        print(f"{name:8s} tokens per character: {tokens_per_char(text, merges):.2f}")
    digits_merges, _ = train_bpe("1234\n" * 20, 259)
    print("cuts of 1234 after training on 1234:", digit_token_lengths("1234", digits_merges))
    print("cuts of 1234 with no merges:", digit_token_lengths("1234", []))
