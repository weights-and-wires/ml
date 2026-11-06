"""Project: serve a small language model.

You inherit a tiny decoder-only transformer with fixed seeded weights, and you
ship it: a tokeniser, fast generation with a KV cache, sampling, speculative
decoding, and a customer's LoRA adapter.

Given, so you do not write them: `make_data`, `forward` (the full-sequence
transformer), `KVCache`, `new_cache`, `plain_linear`, `rmsnorm`, `softmax`,
`make_prob_fn`, `_pretokenize`, `get_pair_counts`, and `merge_pair`. You write
every function from `train_bpe` down.

Conventions. Token ids are Python ints. Ids 0 to 255 are byte values, and the
merge at position i of a merge list creates id 256 + i. A model is a dict of
numpy arrays (see `make_model`). Every linear layer calls
`linear(x, W, key)`, where `key` is `(layer_index, name)`, so an adapter can
hook in without changing the transformer. A weight matrix has shape
(d_out, d_in) and is applied as `x @ W.T`.
"""

import copy
import re

import numpy as np

VOCAB_SIZE = 400
N_LAYER, N_HEAD, D_MODEL, D_FF, MAX_LEN = 2, 2, 32, 64, 256

_PRETOKEN_PATTERN = re.compile(r" ?[^\s]+|\s+")


def _pretokenize(text: str) -> list[str]:
    """Given. Split text into pieces that join back into the exact input."""
    return _PRETOKEN_PATTERN.findall(text)


# ---------------------------------------------------------------------------
# Given: seeded data, the transformer, and the cache container.
# ---------------------------------------------------------------------------

_SUBJECTS = ["the customer", "the driver", "our team", "the warehouse", "a nurse", "the clerk"]
_VERBS = ["shipped", "returned", "checked", "ordered", "packed", "signed"]
_OBJECTS = ["the parcel", "a refund", "the invoice", "three boxes", "the report", "a ticket"]
_TAILS = ["on monday", "in the morning", "before lunch", "without delay", "at the station"]


def make_corpus(seed: int, n_sentences: int = 500) -> str:
    """Given. 500 short sentences from a small shop-and-delivery grammar, about 24,000 characters.

    Words repeat a lot, which is what lets byte-pair encoding find good merges.
    About a third of the sentences end with an order number of four digits.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_sentences):
        s = f"{rng.choice(_SUBJECTS)} {rng.choice(_VERBS)} {rng.choice(_OBJECTS)} {rng.choice(_TAILS)}"
        if rng.uniform() < 0.3:
            s += f" under order {int(rng.integers(1000, 9999))}"
        out.append(s + ".")
    return " ".join(out)


def _init_layer(rng):
    def w(d_out, d_in):
        return rng.normal(size=(d_out, d_in)) / np.sqrt(d_in)

    return {
        "Wq": w(D_MODEL, D_MODEL), "Wk": w(D_MODEL, D_MODEL),
        "Wv": w(D_MODEL, D_MODEL), "Wo": w(D_MODEL, D_MODEL),
        "W1": w(D_FF, D_MODEL), "W2": w(D_MODEL, D_FF),
    }


def make_model(seed: int = 0) -> dict:
    """Given. The target model: 2 layers, 2 heads, width 32, a vocabulary of 400, untied output head.

    The weights are random and fixed by the seed, so the text it writes is not
    fluent. That is deliberate: every check compares your serving code with this
    model, never with good English.
    """
    rng = np.random.default_rng(seed)
    layers = [_init_layer(rng) for _ in range(N_LAYER)]
    for name in ("Wo", "W2"):
        layers[1][name] = 0.4 * layers[1][name]  # the second layer refines the first
    unemb = rng.normal(size=(VOCAB_SIZE, D_MODEL))
    ids = np.arange(VOCAB_SIZE)
    unemb[(ids < 32) | ((ids >= 127) & (ids < 256))] *= 0.2  # bytes that are not printable ASCII
    return {
        "emb": rng.normal(size=(VOCAB_SIZE, D_MODEL)),
        "unemb": unemb,
        "pos": 0.5 * rng.normal(size=(MAX_LEN, D_MODEL)),
        "layers": layers,
        "n_head": N_HEAD,
        "logit_scale": 0.3,
    }


def make_draft(params: dict) -> dict:
    """Given. A smaller draft model: the target with only its first layer. Same vocabulary."""
    draft = dict(params)
    draft["layers"] = params["layers"][:1]
    return draft


def make_adapter(seed: int = 0, r: int = 4, alpha: float = 8.0) -> dict:
    """Given. A customer's LoRA adapter on Wq and Wv of both layers, with trained-looking nonzero B.

    Returns {"alpha", "r", "weights": {(layer, name): (A, B)}}.
    """
    rng = np.random.default_rng(seed)
    weights = {}
    for li in range(N_LAYER):
        for name in ("Wq", "Wv"):
            A = rng.normal(size=(r, D_MODEL)) / np.sqrt(D_MODEL)
            B = 0.2 * rng.normal(size=(D_MODEL, r))
            weights[(li, name)] = (A, B)
    return {"alpha": alpha, "r": r, "weights": weights}


def make_data(seed: int = 0) -> dict:
    """Given. Everything the project starts from, fixed by `seed`.

    Returns a dict with "corpus" (training text for the tokeniser), "params" (the
    target model), "draft" (the one-layer draft), and "adapter" (the LoRA adapter).
    Planted: the corpus has a tiny vocabulary of words, so merges such as
    ' the' are worth learning, and the adapter changes the model's logits visibly.
    """
    return {
        "corpus": make_corpus(seed),
        "params": make_model(seed),
        "draft": make_draft(make_model(seed)),
        "adapter": make_adapter(seed + 100),
    }


def rmsnorm(x: np.ndarray) -> np.ndarray:
    return x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + 1e-6)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def plain_linear(x: np.ndarray, W: np.ndarray, key: tuple) -> np.ndarray:
    return x @ W.T


def forward(params: dict, tokens, linear=plain_linear) -> np.ndarray:
    """Given. Full-sequence logits of shape (len(tokens), VOCAB_SIZE) with a causal mask.

    Row i holds the scores for the token after tokens[: i + 1]. Position
    embeddings are added by absolute position, so tokens must number at most MAX_LEN.
    """
    tokens = np.asarray(tokens)
    T = len(tokens)
    H = params["n_head"]
    dh = D_MODEL // H
    x = params["emb"][tokens] + params["pos"][:T]
    mask = np.triu(np.ones((T, T), dtype=bool), 1)
    for li, layer in enumerate(params["layers"]):
        h = rmsnorm(x)
        q = linear(h, layer["Wq"], (li, "Wq")).reshape(T, H, dh)
        k = linear(h, layer["Wk"], (li, "Wk")).reshape(T, H, dh)
        v = linear(h, layer["Wv"], (li, "Wv")).reshape(T, H, dh)
        scores = np.einsum("thd,shd->hts", q, k) / np.sqrt(dh)
        scores[:, mask] = -np.inf
        att = np.einsum("hts,shd->thd", softmax(scores), v).reshape(T, D_MODEL)
        x = x + linear(att, layer["Wo"], (li, "Wo"))
        h = rmsnorm(x)
        x = x + linear(np.maximum(linear(h, layer["W1"], (li, "W1")), 0.0), layer["W2"], (li, "W2"))
    return rmsnorm(x) @ params["unemb"].T * params["logit_scale"]


class KVCache:
    """Given (from "The KV cache"). Per-layer lists of keys and values.

    `append(layer, k, v)` takes arrays of shape (n_head, d_head). `get(layer)`
    returns (K, V), each of shape (t, n_head, d_head). `length` is the number of
    tokens stored in layer 0, so read it before you append for a new token.
    """

    def __init__(self, n_layer: int, n_head: int, d_head: int):
        self.n_head = n_head
        self.d_head = d_head
        self._keys = [[] for _ in range(n_layer)]
        self._values = [[] for _ in range(n_layer)]

    @property
    def length(self) -> int:
        return len(self._keys[0])

    def append(self, layer: int, k: np.ndarray, v: np.ndarray) -> None:
        self._keys[layer].append(np.asarray(k, dtype=float))
        self._values[layer].append(np.asarray(v, dtype=float))

    def get(self, layer: int) -> tuple:
        return np.stack(self._keys[layer], axis=0), np.stack(self._values[layer], axis=0)


def new_cache(params: dict) -> KVCache:
    """Given. An empty cache sized for `params`."""
    return KVCache(len(params["layers"]), params["n_head"], D_MODEL // params["n_head"])


# ---------------------------------------------------------------------------
# Given: helpers from earlier chapters, and the probability wrapper.
# ---------------------------------------------------------------------------


def get_pair_counts(token_lists: list[list[int]]) -> dict[tuple[int, int], int]:
    """Given (from "Byte-pair encoding: train it, then use it"). Count adjacent pairs inside each list."""
    counts: dict[tuple[int, int], int] = {}
    for tokens in token_lists:
        for a, b in zip(tokens, tokens[1:]):
            counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts


def merge_pair(
    token_lists: list[list[int]], pair: tuple[int, int], new_id: int
) -> list[list[int]]:
    """Given. Replace every non-overlapping occurrence of `pair` with `new_id`, from the left."""
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


def make_prob_fn(params, temperature=1.0, top_k=None, top_p=None, linear=plain_linear):
    """Given. Wrap a model as `fn(tokens) -> probs`, an array of shape (len(tokens), VOCAB_SIZE).

    Row i is the model's next-token distribution after tokens[: i + 1], after
    your `next_token_probs` has applied temperature, top-k, and top-p. One call
    runs one forward pass, so asking for every position costs one model pass.
    """

    def prob_fn(tokens):
        logits = forward(params, tokens, linear)
        return np.stack([next_token_probs(row, temperature, top_k, top_p) for row in logits])

    return prob_fn


# ---------------------------------------------------------------------------
# Stage 1: the tokeniser. Chapters: training BPE, encode and decode, fertility.
# ---------------------------------------------------------------------------


def train_bpe(corpus: str, vocab_size: int) -> tuple[list[tuple[int, int]], dict[int, bytes]]:
    """Learn an ordered merge list and the bytes behind every id.

    Pre-tokenise the corpus with `_pretokenize`, turn each piece into a list of
    byte values, and merge until the vocabulary has `vocab_size` ids. Counts are
    taken inside pieces, so a merge never spans a space boundary. Break ties by
    the smaller pair: pick the pair with the largest count, and among equal
    counts the lexicographically smallest.

    Args:
        corpus: Training text.
        vocab_size: Total ids wanted, 256 bytes plus the merges.

    Returns:
        merges: list of (id_a, id_b); merge i creates id 256 + i.
        vocab: dict from every id to its bytes. Ids 0 to 255 map to single bytes.

    Steps:
        1. token_lists = [list(piece.encode("utf-8")) for piece in _pretokenize(corpus)].
        2. Loop while 256 + len(merges) < vocab_size: count pairs, stop if there are none.
        3. best = min(counts, key=lambda p: (-counts[p], p)); record it and its bytes.
        4. Replace the pair everywhere with merge_pair.
    """
    raise NotImplementedError("train_bpe")


def encode(text: str, merges: list[tuple[int, int]]) -> list[int]:
    """Turn text into ids by replaying the merges in the order they were learned.

    Each piece from `_pretokenize` is encoded on its own: its UTF-8 bytes, then
    merge 0, merge 1, and so on. The result for a text is the pieces' ids joined.

    Returns:
        A list of ints. `encode("", merges)` is `[]`.

    Steps:
        1. For every piece, start from list(piece.encode("utf-8")).
        2. For i, pair in enumerate(merges): merge_pair([ids], pair, 256 + i)[0].
        3. Extend the output with the piece's ids.
    """
    raise NotImplementedError("encode")


def decode(ids: list[int], vocab: dict[int, bytes]) -> str:
    """Turn ids back into text.

    A model can write any id in any order, so the bytes may not be valid UTF-8.
    Replace bad sequences with the replacement character instead of raising.
    For ids that came from `encode`, `decode(encode(s)) == s` exactly.

    Steps:
        1. Join vocab[i] for every id into one bytes object.
        2. Decode it as UTF-8 with errors="replace".
    """
    raise NotImplementedError("decode")


def fertility(text: str, merges: list[tuple[int, int]]) -> float:
    """Tokens per whitespace-separated word: len(encode(text)) / len(text.split())."""
    raise NotImplementedError("fertility")


# ---------------------------------------------------------------------------
# Stage 2: incremental decoding. Chapter: decoding and the KV cache.
# ---------------------------------------------------------------------------


def decode_step(params: dict, token: int, cache: KVCache, linear=plain_linear) -> np.ndarray:
    """Run the transformer on one new token, reading earlier tokens from the cache.

    The token sits at position `cache.length`. In every layer, compute the new
    token's query, key, and value, append the key and value to the cache, and
    attend over everything in the cache. The result must match
    `forward(params, tokens, linear)[-1]` for the same history.

    Args:
        params: A model dict.
        token: The new token id.
        cache: A `KVCache` from `new_cache(params)`. This function appends to it.
        linear: A hook `linear(x, W, key)`, used for every weight matrix.

    Returns:
        Logits of shape (VOCAB_SIZE,) for the token after `token`.

    Steps:
        1. x = params["emb"][token] + params["pos"][cache.length], read before appending.
        2. For each layer li: h = rmsnorm(x); q, k, v = linear(h, W, (li, name)) for
           Wq, Wk, Wv, each reshaped to (n_head, 16).
        3. cache.append(li, k, v); K, V = cache.get(li) has shape (t, n_head, 16).
        4. weights = softmax(einsum("hd,thd->ht", q, K) / sqrt(16)); no mask is
           needed, because the cache holds only the past and the present.
        5. att = einsum("ht,thd->hd", weights, V).reshape(32); x = x + linear(att, Wo).
        6. h = rmsnorm(x); x = x + linear(relu(linear(h, W1)), W2). Copy `forward`.
        7. Return rmsnorm(x) @ params["unemb"].T * params["logit_scale"].
    """
    raise NotImplementedError("decode_step")


# ---------------------------------------------------------------------------
# Stage 3: sampling. Chapter: temperature and sampling.
# ---------------------------------------------------------------------------


def next_token_probs(logits, temperature: float = 1.0, top_k=None, top_p=None) -> np.ndarray:
    """Turn a logit vector of shape (V,) into the distribution a sampler draws from.

    Temperature 0 means greedy: return a one-hot vector at the argmax. Otherwise
    divide by the temperature, softmax (subtract the maximum first), keep the
    `top_k` largest if given, then keep the smallest set of tokens whose
    probability reaches `top_p` if given (the token that crosses `top_p` stays),
    and renormalise so the result sums to 1.

    Returns:
        A float array of shape (V,).
    """
    raise NotImplementedError("next_token_probs")


def sample_token(logits, rng: np.random.Generator, temperature=1.0, top_k=None, top_p=None) -> int:
    """Draw one token id. Temperature 0 returns the argmax and does not touch `rng`.

    Use `rng.choice(len(probs), p=probs)` on the output of `next_token_probs`,
    and return a plain int.
    """
    raise NotImplementedError("sample_token")


def generate(
    params: dict,
    prompt: list[int],
    n_new: int,
    rng: np.random.Generator,
    temperature: float = 0.0,
    top_k=None,
    top_p=None,
    use_cache: bool = True,
    linear=plain_linear,
) -> list[int]:
    """Write `n_new` tokens after `prompt` and return only the new ones.

    With `use_cache=True`, feed the prompt through `decode_step` one token at a
    time, then loop: sample from the latest logits, feed the sampled token back.
    With `use_cache=False`, recompute `forward(params, tokens, linear)[-1]` over
    the whole sequence at every step. Both paths must produce the same tokens
    for the same seed.

    Steps:
        1. Keep `tokens`, the prompt plus everything written so far.
        2. Cached: build `cache = new_cache(params)` and keep the logits of the last token.
        3. Each step: sample_token(logits, rng, temperature, top_k, top_p), then update the logits.
    """
    raise NotImplementedError("generate")


# ---------------------------------------------------------------------------
# Stage 4: speculative decoding. Chapter: speculative decoding.
# ---------------------------------------------------------------------------


def residual_distribution(p, q) -> np.ndarray:
    """The replacement distribution after a rejection: max(p - q, 0), renormalised.

    When the total is 0 (p equals q), return a copy of `p`.
    """
    raise NotImplementedError("residual_distribution")


def speculative_round(target_fn, draft_fn, context: list[int], k: int, rng: np.random.Generator):
    """One round: the draft writes k tokens, the target checks them in one pass.

    `target_fn(tokens)` and `draft_fn(tokens)` return arrays of shape
    (len(tokens), V); row i is the next-token distribution after tokens[: i + 1].

    1. The draft proposes k tokens, one after another, each drawn from the last
       row of `draft_fn(context + drafted)`. Remember each row as q.
    2. Call `target_fn(context + drafted)` once. Row len(context) - 1 + i is the
       target's p for draft token i, and row len(context) - 1 + k is for the bonus.
    3. For each draft token x in order: accept it when rng.uniform() < min(1, p[x] / q[x]).
       On the first rejection, draw a replacement from residual_distribution(p, q)
       and stop. If all k are accepted, draw one bonus token from the last row.

    Returns:
        (new_tokens, n_accepted, n_checked): the tokens this round adds (between 1
        and k + 1), how many draft tokens were accepted, and how many draft
        tokens the target examined (the accepted ones plus the rejected one).
    """
    raise NotImplementedError("speculative_round")


def speculative_generate(target_fn, draft_fn, prompt: list[int], n_new: int, k: int, rng: np.random.Generator):
    """Run rounds until `n_new` tokens exist, and report how well the draft did.

    A round can overshoot, so cut the result to exactly `n_new` tokens.

    Returns:
        (new_tokens, stats), where stats is a dict with the keys "accepted",
        "checked", "target_passes" (rounds), and "acceptance_rate"
        (accepted / checked, or 0.0 when nothing was checked).
    """
    raise NotImplementedError("speculative_generate")


# ---------------------------------------------------------------------------
# Stage 5: a customer's LoRA adapter. Chapter: LoRA, train a low-rank update then merge it.
# An adapter is {"alpha": float, "r": int, "weights": {(layer, name): (A, B)}},
# with A of shape (r, d_in) and B of shape (d_out, r).
# ---------------------------------------------------------------------------


def lora_linear(x, W, A, B, alpha: float, r: int) -> np.ndarray:
    """Apply W plus a low-rank update without forming it: x @ W.T + (alpha / r) * (x @ A.T) @ B.T.

    `x` can have shape (d_in,) or (n, d_in).
    """
    raise NotImplementedError("lora_linear")


def merge_lora(W, A, B, alpha: float, r: int) -> np.ndarray:
    """Fold the update into the weight: W + (alpha / r) * (B @ A), shape (d_out, d_in)."""
    raise NotImplementedError("merge_lora")


def make_lora_linear(adapter: dict):
    """Return a `linear(x, W, key)` hook that applies the adapter on the fly.

    When `key` is in `adapter["weights"]`, call `lora_linear` with that key's A and
    B and the adapter's alpha and r. Otherwise return x @ W.T.
    """
    raise NotImplementedError("make_lora_linear")


def merge_adapter(params: dict, adapter: dict) -> dict:
    """Return a new model whose adapted weights are merged, leaving `params` untouched.

    Use `copy.deepcopy(params)`, then replace each adapted
    `layer[name]` with `merge_lora` of the original weight. The merged model runs
    with the default `plain_linear` and has no extra cost per token.
    """
    raise NotImplementedError("merge_adapter")


# ---------------------------------------------------------------------------
# Stage 6: the whole service.
# ---------------------------------------------------------------------------


def complete(
    prompt: str,
    merges: list[tuple[int, int]],
    vocab: dict[int, bytes],
    params: dict,
    n_new: int,
    rng: np.random.Generator,
    temperature: float = 0.0,
    top_k=None,
    top_p=None,
    adapter=None,
    merge: bool = False,
) -> str:
    """Encode the prompt, write `n_new` tokens with the KV cache, and decode prompt plus new text.

    With an adapter and `merge=True`, serve the merged model with `plain_linear`.
    With an adapter and `merge=False`, serve the base model with
    `make_lora_linear(adapter)`. Without an adapter, serve the base model.
    The returned text starts with `prompt`.
    """
    raise NotImplementedError("complete")


if __name__ == "__main__":
    data = make_data(0)
    merges, vocab = train_bpe(data["corpus"], VOCAB_SIZE)
    text = "the driver returned a refund on monday."
    ids = encode(text, merges)
    print(f"{len(text.encode())} bytes -> {len(ids)} tokens; round trip ok:", decode(ids, vocab) == text)
    print("fertility, in domain:", round(fertility(text, merges), 2),
          "| German:", round(fertility("Der Fahrer gab die Rückerstattung zurück.", merges), 2))

    params, adapter = data["params"], data["adapter"]
    cache = new_cache(params)
    step = np.stack([decode_step(params, t, cache) for t in ids])
    print("cache against full recompute, largest gap:", np.abs(step - forward(params, ids)).max())

    rng = np.random.default_rng(0)
    print("greedy:", repr(complete(text, merges, vocab, params, 12, rng)))
    print("sampled:", repr(complete(text, merges, vocab, params, 12, rng, temperature=0.8, top_k=40)))
    print("with adapter, on the fly:", repr(complete(text, merges, vocab, params, 12, rng, adapter=adapter)))
    print("with adapter, merged:    ", repr(complete(text, merges, vocab, params, 12, rng, adapter=adapter, merge=True)))

    target = make_prob_fn(params, 1.0, 5)
    draft = make_prob_fn(data["draft"], 1.0, 5)
    for k in (2, 4):
        _, stats = speculative_generate(target, draft, ids, 100, k, np.random.default_rng(0))
        print(f"speculative k={k}: acceptance {stats['acceptance_rate']:.2f}, "
              f"{100 / stats['target_passes']:.2f} tokens per target pass")
