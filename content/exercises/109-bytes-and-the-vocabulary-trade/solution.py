"""Reference solution: bytes as a vocabulary, and the unknown-word rate."""


def text_to_bytes(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def bytes_to_text(ids: list[int]) -> str:
    return bytes(ids).decode("utf-8")


def unknown_rate(text: str, word_vocab: set[str]) -> float:
    words = text.split()
    if not words:
        return 0.0
    missing = sum(1 for w in words if w not in word_vocab)
    return missing / len(words)


if __name__ == "__main__":
    sample = "héllo 世界 🎉"
    ids = text_to_bytes(sample)
    print(f"{sample!r}: {len(sample)} characters, {len(ids)} bytes")
    print("ids:", ids)
    print("round trip ok:", bytes_to_text(ids) == sample)

    word_vocab = {"the", "cat", "sat"}
    text = "the cat sat on the mat"
    print(f"unknown rate of a 3-word list on {text!r}: {unknown_rate(text, word_vocab):.2f}")
    n = 5700
    print(f"attention pairs for {n} bytes: {n * n:,}; for 1000 words: {1000 * 1000:,}")
