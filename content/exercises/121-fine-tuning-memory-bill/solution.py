"""Reference solution for the fine-tuning memory bill exercise."""


def adam_bytes_breakdown() -> dict:
    return {
        "weight": 2,
        "gradient": 2,
        "master_weight": 4,
        "first_average": 4,
        "second_average": 4,
    }


def optimiser_memory_bytes(params: int, mode: str) -> int:
    if mode not in ("full", "lora"):
        raise ValueError(f"mode must be 'full' or 'lora', got {mode!r}")
    return 16 * params


def trainable_fraction_memory(total_params: int, trained_params: int) -> int:
    if trained_params > total_params:
        raise ValueError("trained_params cannot exceed total_params")
    return 16 * trained_params + 2 * (total_params - trained_params)


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
