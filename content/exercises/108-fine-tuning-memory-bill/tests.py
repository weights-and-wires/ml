"""Checks for the fine-tuning memory bill exercise."""

import numpy as np

from mlcheck import check, equal, fail, has


@check("adam_bytes_breakdown lists the five pieces, and they add up to 16 bytes")
def test_breakdown(sub):
    got = has(sub, "adam_bytes_breakdown")()
    want = {
        "weight": 2,
        "gradient": 2,
        "master_weight": 4,
        "first_average": 4,
        "second_average": 4,
    }
    if set(got) != set(want):
        fail(f"expected keys {sorted(want)}, got {sorted(got)}")
    for key, nbytes in want.items():
        if got[key] != nbytes:
            fail(f"{key!r} should be {nbytes} bytes, got {got[key]}")
    equal(sum(got.values()), 16, label="bytes per trained parameter")


@check("optimiser_memory_bytes gives 112 GB for 7 billion parameters, in both modes")
def test_seven_billion(sub):
    memory = has(sub, "optimiser_memory_bytes")
    for mode in ("full", "lora"):
        got = memory(7_000_000_000, mode)
        if got != 112_000_000_000:
            fail(f"mode {mode!r}: expected 112,000,000,000 bytes for 7e9 parameters, got {got:,}")


@check("optimiser_memory_bytes is linear in the parameter count, with the same rate in both modes")
def test_linear(sub):
    memory = has(sub, "optimiser_memory_bytes")
    rng = np.random.default_rng(0)
    for params in rng.integers(1, 10**9, size=6):
        params = int(params)
        for mode in ("full", "lora"):
            got = memory(params, mode)
            if got != 16 * params:
                fail(f"memory({params}, {mode!r}) is {got}, expected {16 * params}")


@check("optimiser_memory_bytes rejects a mode that is not 'full' or 'lora'")
def test_bad_mode(sub):
    memory = has(sub, "optimiser_memory_bytes")
    for bad in ("half", "", "FULL"):
        try:
            memory(1000, bad)
        except ValueError:
            continue
        fail(f"mode {bad!r} should raise ValueError, but it returned a value")


@check("trainable_fraction_memory charges 16 bytes per trained and 2 per frozen parameter")
def test_fraction(sub):
    got = has(sub, "trainable_fraction_memory")(7_000_000_000, 70_000_000)
    want = 16 * 70_000_000 + 2 * 6_930_000_000
    if got != want:
        fail(f"expected {want:,} bytes (about 14.98 GB) for 1% of 7e9 trained, got {got:,}")


@check("trainable_fraction_memory agrees with the full bill at the two ends")
def test_fraction_ends(sub):
    fraction = has(sub, "trainable_fraction_memory")
    memory = has(sub, "optimiser_memory_bytes")
    rng = np.random.default_rng(1)
    for total in rng.integers(1, 10**9, size=4):
        total = int(total)
        equal(fraction(total, total), memory(total, "full"), label="train everything")
        equal(fraction(total, 0), 2 * total, label="train nothing")


@check("trainable_fraction_memory rejects more trained parameters than exist")
def test_fraction_error(sub):
    fraction = has(sub, "trainable_fraction_memory")
    try:
        fraction(100, 101)
    except ValueError:
        return
    fail("trained_params larger than total_params should raise ValueError")
