"""The memory bill for fine-tuning.

Fill in the three functions below. Each trained parameter carries five pieces
(see Figure 1 of the chapter), and a frozen parameter carries only its weight.
"""


def adam_bytes_breakdown() -> dict:
    """Return the bytes each trained parameter carries, piece by piece.

    Returns:
        A dict with exactly these keys, each mapped to an int byte count:
        "weight" (the 16-bit working copy), "gradient" (16-bit),
        "master_weight" (the 32-bit copy), "first_average" (Adam's m, 32-bit),
        and "second_average" (Adam's v, 32-bit).
    """
    raise NotImplementedError("adam_bytes_breakdown")


def optimiser_memory_bytes(params: int, mode: str) -> int:
    """Return the bytes needed to train `params` parameters with Adam.

    Both modes use the same per-parameter rate: `mode` says which kind of
    parameters are being counted, not how much each one costs.

    Args:
        params: Number of trained parameters.
        mode: "full" or "lora".

    Returns:
        The bytes needed, as an int.

    Raises:
        ValueError: If `mode` is not "full" or "lora".
    """
    raise NotImplementedError("optimiser_memory_bytes")


def trainable_fraction_memory(total_params: int, trained_params: int) -> int:
    """Return the bytes when only some parameters are trained.

    A trained parameter costs 16 bytes. A frozen parameter costs 2 bytes,
    because its weight must still sit in memory.

    Args:
        total_params: All parameters in the model.
        trained_params: How many of them are trained.

    Returns:
        The bytes needed, as an int.

    Raises:
        ValueError: If trained_params is larger than total_params.
    """
    raise NotImplementedError("trainable_fraction_memory")


if __name__ == "__main__":
    pieces = adam_bytes_breakdown()
    print("bytes per trained parameter:", pieces, "->", sum(pieces.values()))

    n = 7_000_000_000
    full = optimiser_memory_bytes(n, "full")
    print(f"full fine-tune of {n:,} parameters: {full / 1e9:.2f} GB")

    one_percent = n // 100
    frozen = trainable_fraction_memory(n, one_percent)
    print(f"train 1% and freeze the rest: {frozen / 1e9:.2f} GB")
    print("fits a 24 GB GPU before activations:", frozen < 24e9)
