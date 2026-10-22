"""Reference solution for the counting-parameters-and-flops exercise."""


def transformer_params(
    layers: int, d_model: int, d_ff: int, vocab: int, seq_len: int
) -> dict:
    attention = layers * 4 * d_model * d_model
    feedforward = layers * 2 * d_model * d_ff
    embedding = vocab * d_model + seq_len * d_model
    non_embedding = attention + feedforward
    return {
        "attention": attention,
        "feedforward": feedforward,
        "embedding": embedding,
        "non_embedding": non_embedding,
        "total": non_embedding + embedding,
    }


def training_flops(params, tokens) -> float:
    return 6.0 * float(params) * float(tokens)


def training_days(flops: float, flops_per_second: float = 2e14) -> float:
    return float(flops) / flops_per_second / 86400.0


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
