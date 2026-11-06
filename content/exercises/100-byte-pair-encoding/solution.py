"""Reference solution: train byte-pair encoding, then encode and decode with it."""

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


if __name__ == "__main__":
    merges, vocab = train_bpe("aaabdaaabac", 259)
    print("merges:", merges)
    for i, pair in enumerate(merges):
        print(f"  id {256 + i} = {pair} = {vocab[256 + i]!r}")

    ids = encode("aaabdaaabac", merges)
    print("ids:", ids, "(expected [258, 100, 258, 97, 99])")
    print("reversed merge order:", encode("aaabdaaabac", list(reversed(merges))))

    corpus = "the cat sat on the mat. the dog sat on the log. " * 4
    merges, vocab = train_bpe(corpus, 290)
    shown = [vocab[256 + i].decode() for i in range(len(merges))]
    print("English merges:", shown)
    for text in ("the cat sat", "héllo 世界 🎉"):
        ids = encode(text, merges)
        print(f"{text!r}: {len(text.encode('utf-8'))} bytes -> {len(ids)} ids,",
              "round trip ok:", decode(ids, vocab) == text)
