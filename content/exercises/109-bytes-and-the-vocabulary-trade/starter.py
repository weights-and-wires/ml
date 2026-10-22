"""Bytes as a vocabulary, and the unknown-word rate.

Fill in the three functions below. The Run button executes the block at the
bottom, which fails with NotImplementedError until the first one is written.
"""


def text_to_bytes(text: str) -> list[int]:
    """Return the UTF-8 bytes of `text` as a list of ints.

    Args:
        text: Any string.

    Returns:
        A list of ints, each between 0 and 255. Its length is at least
        len(text), because one character can take up to four bytes.
    """
    raise NotImplementedError("text_to_bytes")


def bytes_to_text(ids: list[int]) -> str:
    """Turn a list of byte values back into the string they spell.

    Args:
        ids: A list of ints between 0 and 255, a whole UTF-8 byte string.

    Returns:
        The decoded string. Decode the whole list in one call.
    """
    raise NotImplementedError("bytes_to_text")


def unknown_rate(text: str, word_vocab: set[str]) -> float:
    """Return the share of whitespace-separated words missing from `word_vocab`.

    The match is exact and case-sensitive, so "The" is not "the".

    Args:
        text: The text to check.
        word_vocab: A set of known words.

    Returns:
        missing words divided by total words, a float from 0.0 to 1.0.
        Return 0.0 when `text` has no words.
    """
    raise NotImplementedError("unknown_rate")


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
