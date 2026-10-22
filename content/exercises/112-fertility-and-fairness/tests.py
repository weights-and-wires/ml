"""Checks for the fertility and fairness exercise."""

import numpy as np

from mlcheck import check, equal, fail, has

ENGLISH = (
    "the cat sat on the mat. the dog sat on the log. the cat and the dog are friends. "
    "a cat likes to sit in the sun, and a dog likes to run in the park. "
    "the children are playing in the garden, and the teacher is reading a book. "
    "we are learning about the world, and the world is full of interesting things. "
    "the morning was quiet, and the street was empty. nothing was moving in the town. "
    "she was reading, he was writing, and they were singing in the evening. "
    "the running water is shining, and the singing birds are flying over the hills. "
)
HINDI = (
    "बिल्ली चटाई पर बैठी थी। मुझे चाय पसंद है। आज मौसम बहुत अच्छा है। "
    "वह रोज़ सुबह स्कूल जाता है। हम किताबें पढ़ना पसंद करते हैं। कुत्ता बगीचे में दौड़ रहा है। "
    "माँ रसोई में खाना बना रही हैं। बच्चे मैदान में खेल रहे हैं। "
)
EN_SENTENCE = "the cat sat on the mat."
HI_SENTENCE = "बिल्ली चटाई पर बैठी थी।"
SAMPLE = "the dog and the cat were singing in the garden"


def learned(sub, corpus, vocab_size):
    """Train with the reader's module and return only the merge list."""
    return has(sub, "train_bpe")(corpus, vocab_size)[0]


@check("fertility with no merges is bytes per word, counted by whitespace")
def test_fertility_raw(sub):
    f = has(sub, "fertility")
    got = f("the quick fox", [])
    if abs(got - 13 / 3) > 1e-9:
        fail(f"the quick fox is 13 bytes in 3 words, expected 4.3333, got {got}")
    got = f("hello  world", [])
    if abs(got - 6.0) > 1e-9:
        fail(f"hello  world is 12 bytes in 2 words, expected 6.0, got {got}")
    if abs(f("a", []) - 1.0) > 1e-9:
        fail("a single letter is 1 token in 1 word, so fertility 1.0")


@check("fertility falls as the merge list grows, and unseen words do not shrink")
def test_fertility_falls(sub):
    f = has(sub, "fertility")
    merges = learned(sub, ENGLISH, 256 + 120)
    values = [f(SAMPLE, merges[:k]) for k in (0, 20, 80)]
    if not (values[0] > values[1] > values[2]):
        fail(f"fertility should fall strictly with 0, 20, 80 merges, got {values}")
    if values[2] > 2.0:
        fail(f"80 merges on this corpus should bring fertility under 2.0, got {values[2]}")
    unseen = f("zzzz xxxx", merges[:80])
    if abs(unseen - 4.5) > 1e-9:
        fail(f"words made of letters never seen cannot shrink: expected 4.5, got {unseen}")


@check("tokens_per_char with no merges equals the UTF-8 bytes per character")
def test_tpc_raw(sub):
    f = has(sub, "tokens_per_char")
    for text, want in (("abc", 1.0), ("é", 2.0), ("世界", 3.0), ("🎉", 4.0), ("a世", 2.0)):
        got = f(text, [])
        if abs(got - want) > 1e-9:
            fail(f"tokens_per_char({text!r}, []) should be {want}, got {got}")


@check("An English-trained tokeniser charges Hindi several times more per character")
def test_fairness_gap(sub):
    f = has(sub, "tokens_per_char")
    merges = learned(sub, ENGLISH, 256 + 80)
    en = f(EN_SENTENCE, merges)
    hi = f(HI_SENTENCE, merges)
    if en > 0.5:
        fail(f"English should cost under 0.5 tokens per character after 80 merges, got {en:.3f}")
    if hi < 2.5:
        fail(f"Hindi has no merges of its own and should cost about 2.65 per character, got {hi:.3f}")
    if hi / en < 4.0:
        fail(f"the gap hi / en should be above 4, got {hi / en:.2f}")


@check("Training on both languages cuts the Hindi cost and raises the English cost")
def test_shared_budget(sub):
    f = has(sub, "tokens_per_char")
    only_en = learned(sub, ENGLISH, 256 + 80)
    both = learned(sub, ENGLISH + HINDI, 256 + 80)
    if not f(HI_SENTENCE, both) < f(HI_SENTENCE, only_en) / 1.5:
        fail("training on Hindi too should cut its cost per character by a lot")
    if not f(EN_SENTENCE, both) > f(EN_SENTENCE, only_en):
        fail("with a fixed budget, English should cost more once merges go to Hindi")


@check("digit_token_lengths gives the three cuts of 1234 from the chapter")
def test_digit_cuts(sub):
    f = has(sub, "digit_token_lengths")
    whole = learned(sub, "1234\n" * 20, 259)
    apart = learned(sub, "12\n34\n" * 20, 258)
    english = learned(sub, ENGLISH, 256 + 50)
    equal(list(f("1234", whole)), [4], label="trained on 1234")
    equal(list(f("1234", apart)), [2, 2], label="trained on 12 and 34 apart")
    equal(list(f("1234", english)), [1, 1, 1, 1], label="trained on English")
    equal(list(f("no digits here", whole)), [], label="a text without digits")


@check("Digit cuts follow the tokens, not the place value, and always add up to the digits")
def test_digit_sums(sub):
    f = has(sub, "digit_token_lengths")
    whole = learned(sub, "1234\n" * 20, 259)
    equal(list(f("11234", whole)), [1, 4], label="11234")
    equal(list(f("12345", whole)), [4, 1], label="12345")
    equal(list(f("price 1234 and 56", whole)), [4, 1, 1], label="price 1234 and 56")
    rng = np.random.default_rng(0)
    alphabet = list("0123456789 ab.\n")
    for _ in range(20):
        text = "".join(rng.choice(alphabet, size=int(rng.integers(1, 30))))
        got = f(text, whole)
        want = sum(ch.isdigit() for ch in text)
        if sum(got) != want:
            fail(f"digit lengths {got} sum to {sum(got)}, but {text!r} has {want} digits")
