"""Reference solution for the byte-pair encoding exercise."""

import re

_PRETOKEN_PATTERN = re.compile(r" ?[^\s]+|\s+")


def _pretokenize(text: str) -> list[str]:
    return _PRETOKEN_PATTERN.findall(text)


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
        n = len(tokens)
        while i < n:
            if i < n - 1 and (tokens[i], tokens[i + 1]) == pair:
                merged.append(new_id)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        result.append(merged)
    return result


def train_bpe(corpus: str, vocab_size: int) -> tuple[list[tuple[int, int]], dict[int, bytes]]:
    words = _pretokenize(corpus)
    token_lists = [list(w.encode("utf-8")) for w in words]

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[int, int]] = []
    next_id = 256

    while next_id < vocab_size:
        counts = get_pair_counts(token_lists)
        if not counts:
            break
        best_pair = min(counts, key=lambda p: (-counts[p], p))
        merges.append(best_pair)
        vocab[next_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        token_lists = merge_pair(token_lists, best_pair, next_id)
        next_id += 1

    return merges, vocab


def encode(text: str, merges: list[tuple[int, int]]) -> list[int]:
    words = _pretokenize(text)
    token_lists = [list(w.encode("utf-8")) for w in words]

    for i, pair in enumerate(merges):
        token_lists = merge_pair(token_lists, pair, 256 + i)

    ids: list[int] = []
    for tokens in token_lists:
        ids.extend(tokens)
    return ids


def decode(ids: list[int], vocab: dict[int, bytes]) -> str:
    return b"".join(vocab[i] for i in ids).decode("utf-8")


def fertility(text: str, merges: list[tuple[int, int]]) -> float:
    return len(encode(text, merges)) / len(text.split())


if __name__ == "__main__":
    corpus = (
        "the quick brown fox jumps over the lazy dog. "
        "the quick brown fox runs from the lazy dog. "
    ) * 20

    for vocab_size in (260, 280, 320):
        merges, vocab = train_bpe(corpus, vocab_size)
        sample = "the quick brown fox jumps over a very lazy dog"
        ids = encode(sample, merges)
        back = decode(ids, vocab)
        assert back == sample, "decode(encode(text)) must reproduce text exactly"
        print(
            f"vocab_size={vocab_size:4d}  merges={len(merges):3d}  "
            f"tokens for sample={len(ids):3d}  fertility={fertility(sample, merges):.2f}"
        )

    # Byte-level tokenisation never fails on text the corpus never saw,
    # including scripts and symbols entirely absent from training.
    unseen = "héllo — 世界 🎉"
    merges, vocab = train_bpe(corpus, 270)
    assert decode(encode(unseen, merges), vocab) == unseen
    print(f"round-tripped text the training corpus never saw: {unseen!r}")
