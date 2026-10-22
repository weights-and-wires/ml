"""Counting weights, and turning them into a training bill.

Fill in the three functions below. `transformer_params` counts the weights
that multiply separately from the embedding tables. `training_flops` applies
the six-flops-per-weight-per-token rule. `training_days` converts flops into
days on one accelerator.
"""


def transformer_params(
    layers: int, d_model: int, d_ff: int, vocab: int, seq_len: int
) -> dict:
    """Count a transformer's parameters, broken down by component.

    Assumes four square attention matrices per block (Q, K, V, and the output
    matrix, each d_model by d_model), a feed-forward network with two matrices
    (d_model by d_ff, then d_ff by d_model), a token table of shape
    (vocab, d_model), a position table of shape (seq_len, d_model), and an
    output layer that reuses the token table (no extra parameters). Biases and
    layer-norm parameters are left out.

    Args:
        layers: Number of blocks.
        d_model: Model width.
        d_ff: Feed-forward hidden width.
        vocab: Vocabulary size.
        seq_len: Number of positions in the position table.

    Returns:
        A dict of integers with keys "attention", "feedforward", "embedding",
        "non_embedding" (attention plus feedforward, summed over all blocks),
        and "total" (non_embedding plus embedding).
    """
    raise NotImplementedError("transformer_params")


def training_flops(params, tokens) -> float:
    """Estimate training compute as 6 * params * tokens.

    Convert to float before multiplying, so that numpy integer inputs do not
    wrap around.

    Args:
        params: Non-embedding parameter count.
        tokens: Number of training tokens.

    Returns:
        The estimated training compute, in flops, as a float.
    """
    raise NotImplementedError("training_flops")


def training_days(flops: float, flops_per_second: float = 2e14) -> float:
    """Convert a flop count into days on one accelerator.

    Args:
        flops: Total flops to run.
        flops_per_second: Sustained speed of the accelerator.

    Returns:
        The time in days (86,400 seconds each), as a float.
    """
    raise NotImplementedError("training_days")


if __name__ == "__main__":
    toy = transformer_params(layers=2, d_model=8, d_ff=32, vocab=100, seq_len=16)
    print("toy model:", toy)

    for name, cfg in [
        ("12 blocks, width 768", (12, 768, 3072, 50257, 1024)),
        ("48 blocks, width 1600", (48, 1600, 6400, 50257, 1024)),
    ]:
        b = transformer_params(*cfg)
        share = b["embedding"] / b["total"]
        print(f"{name}: N = {b['non_embedding']:,}, embeddings are {share:.1%} of all parameters")

    big = transformer_params(48, 1600, 6400, 50257, 1024)
    flops = training_flops(big["non_embedding"], 300_000_000_000)
    print(f"\n1.5B-parameter model on 300B tokens: {flops:.3e} flops")
    print(f"on one accelerator at 2e14 flops/s: {training_days(flops):.0f} days")
    print(f"a 1e20-flop run: {training_days(1e20):.2f} days")
