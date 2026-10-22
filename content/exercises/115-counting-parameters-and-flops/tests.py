"""Checks for the counting-parameters-and-flops exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, has


@check("transformer_params matches the chapter's hand count for the toy model")
def test_toy(sub):
    got = has(sub, "transformer_params")(layers=2, d_model=8, d_ff=32, vocab=100, seq_len=16)
    want = {"attention": 512, "feedforward": 1024, "embedding": 928, "non_embedding": 1536, "total": 2464}
    for key, value in want.items():
        if key not in got:
            fail(f"transformer_params output has no {key!r} key")
        if got[key] != value:
            fail(f"transformer_params()[{key!r}] is {got[key]:,}, expected {value:,} for the toy model")
        if not isinstance(got[key], (int, np.integer)):
            fail(f"transformer_params()[{key!r}] should be an integer, got {type(got[key]).__name__}")


@check("Embeddings are counted once, not once per block")
def test_embeddings_once(sub):
    tp = has(sub, "transformer_params")
    two = tp(layers=2, d_model=64, d_ff=256, vocab=1000, seq_len=128)
    six = tp(layers=6, d_model=64, d_ff=256, vocab=1000, seq_len=128)
    if two["embedding"] != six["embedding"]:
        fail(f"embedding count changed with layers: {two['embedding']:,} versus {six['embedding']:,}")
    almost(six["non_embedding"] / two["non_embedding"], 3.0, tol=1e-12, label="non_embedding ratio for 6 blocks versus 2")


@check("Doubling d_model and d_ff quadruples the weights that multiply; with d_ff = 4 d_model a block holds 12 d_model**2")
def test_scaling(sub):
    tp = has(sub, "transformer_params")
    small = tp(layers=6, d_model=32, d_ff=128, vocab=500, seq_len=64)
    large = tp(layers=6, d_model=64, d_ff=256, vocab=500, seq_len=64)
    almost(large["non_embedding"] / small["non_embedding"], 4.0, tol=1e-12, label="ratio under doubled width")
    for layers, d in [(3, 16), (12, 96)]:
        got = tp(layers=layers, d_model=d, d_ff=4 * d, vocab=1000, seq_len=100)["non_embedding"]
        if got != 12 * layers * d * d:
            fail(f"with d_ff = 4 * d_model, non_embedding should be 12 * layers * d_model**2 = {12 * layers * d * d:,}, got {got:,}")


@check("training_flops is 6 * params * tokens")
def test_flops(sub):
    tf = has(sub, "training_flops")
    almost(tf(1_000_000, 500_000), 3.0e12, tol=1.0, label="training_flops(1e6, 5e5)")
    almost(tf(1.5e9, 3e11) / 2.7e21, 1.0, tol=1e-12, label="training_flops(1.5e9, 3e11) relative to 2.7e21")


@check("training_flops does not overflow on numpy int64 inputs")
def test_flops_overflow(sub):
    tf = has(sub, "training_flops")
    got = tf(np.int64(7_000_000_000), np.int64(300_000_000_000))
    want = 6.0 * 7e9 * 3e11
    if not np.isfinite(got) or got <= 0 or abs(got - want) > 1e-9 * want:
        fail(f"training_flops(int64 7e9, int64 3e11) returned {got!r}, expected {want:.4e}. Convert to float before multiplying.")


@check("training_days: 1e20 flops take about 5.79 days at the default 2e14 flops per second")
def test_days(sub):
    td = has(sub, "training_days")
    almost(td(1e20), 5.787037, tol=1e-5, label="days for 1e20 flops")
    almost(td(1e20, 1e15), 1.157407, tol=1e-5, label="days at 1e15 flops per second")
    almost(td(2e20) / td(1e20), 2.0, tol=1e-12, label="time doubles with flops")


@check("A billion-parameter-scale run: config to flops to days")
def test_pipeline(sub):
    tp, tf, td = has(sub, "transformer_params"), has(sub, "training_flops"), has(sub, "training_days")
    b = tp(48, 1600, 6400, 50257, 1024)
    if b["non_embedding"] != 1_474_560_000:
        fail(f"non_embedding for 48 blocks, width 1600, d_ff 6400 is {b['non_embedding']:,}, expected 1,474,560,000")
    flops = tf(b["non_embedding"], 300_000_000_000)
    almost(flops / 2.6542e21, 1.0, tol=1e-4, label="flops for 300B tokens, relative to 2.6542e21")
    almost(td(flops), 153.6, tol=0.1, label="days on one accelerator")
