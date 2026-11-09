"""Reference solution: serve a small language model.

The pipeline is a tokeniser (byte-pair encoding), a decoder-only transformer
that is given, a KV cache, a sampler, speculative decoding, and a LoRA adapter.
`make_data`, the transformer, and the helpers marked "given" are not the
reader's work.
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
    draft = dict(params)
    draft["layers"] = params["layers"][:1]
    return draft


def make_adapter(seed: int = 0, r: int = 4, alpha: float = 8.0) -> dict:
    rng = np.random.default_rng(seed)
    weights = {}
    for li in range(N_LAYER):
        for name in ("Wq", "Wv"):
            A = rng.normal(size=(r, D_MODEL)) / np.sqrt(D_MODEL)
            B = 0.2 * rng.normal(size=(D_MODEL, r))
            weights[(li, name)] = (A, B)
    return {"alpha": alpha, "r": r, "weights": weights}


def make_data(seed: int = 0) -> dict:
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
    return KVCache(len(params["layers"]), params["n_head"], D_MODEL // params["n_head"])


# ---------------------------------------------------------------------------
# Stage 1: the tokeniser.
# ---------------------------------------------------------------------------


def get_pair_counts(token_lists):
    counts: dict[tuple[int, int], int] = {}
    for tokens in token_lists:
        for a, b in zip(tokens, tokens[1:]):
            counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts


def merge_pair(token_lists, pair, new_id):
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


def train_bpe(corpus: str, vocab_size: int) -> tuple[list[tuple[int, int]], dict[int, bytes]]:
    token_lists = [list(piece.encode("utf-8")) for piece in _pretokenize(corpus)]
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[int, int]] = []
    while 256 + len(merges) < vocab_size:
        counts = get_pair_counts(token_lists)
        if not counts:
            break
        best = min(counts, key=lambda p: (-counts[p], p))
        new_id = 256 + len(merges)
        merges.append(best)
        vocab[new_id] = vocab[best[0]] + vocab[best[1]]
        token_lists = merge_pair(token_lists, best, new_id)
    return merges, vocab


def encode(text: str, merges: list[tuple[int, int]]) -> list[int]:
    ids: list[int] = []
    for piece in _pretokenize(text):
        piece_ids = list(piece.encode("utf-8"))
        for i, pair in enumerate(merges):
            piece_ids = merge_pair([piece_ids], pair, 256 + i)[0]
        ids.extend(piece_ids)
    return ids


def decode(ids: list[int], vocab: dict[int, bytes]) -> str:
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


def fertility(text: str, merges: list[tuple[int, int]]) -> float:
    return len(encode(text, merges)) / len(text.split())


# ---------------------------------------------------------------------------
# Stage 2: incremental decoding with a KV cache.
# ---------------------------------------------------------------------------


def decode_step(params: dict, token: int, cache: KVCache, linear=plain_linear) -> np.ndarray:
    H = params["n_head"]
    dh = D_MODEL // H
    x = params["emb"][token] + params["pos"][cache.length]
    for li, layer in enumerate(params["layers"]):
        h = rmsnorm(x)
        q = linear(h, layer["Wq"], (li, "Wq")).reshape(H, dh)
        k = linear(h, layer["Wk"], (li, "Wk")).reshape(H, dh)
        v = linear(h, layer["Wv"], (li, "Wv")).reshape(H, dh)
        cache.append(li, k, v)
        K, V = cache.get(li)
        weights = softmax(np.einsum("hd,thd->ht", q, K) / np.sqrt(dh))
        att = np.einsum("ht,thd->hd", weights, V).reshape(D_MODEL)
        x = x + linear(att, layer["Wo"], (li, "Wo"))
        h = rmsnorm(x)
        x = x + linear(np.maximum(linear(h, layer["W1"], (li, "W1")), 0.0), layer["W2"], (li, "W2"))
    return rmsnorm(x) @ params["unemb"].T * params["logit_scale"]


# ---------------------------------------------------------------------------
# Stage 3: sampling.
# ---------------------------------------------------------------------------


def next_token_probs(logits, temperature: float = 1.0, top_k=None, top_p=None) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    if temperature == 0:
        probs = np.zeros_like(logits)
        probs[int(np.argmax(logits))] = 1.0
        return probs
    probs = softmax(logits / temperature)
    order = np.argsort(-probs, kind="stable")
    if top_k is not None:
        keep = np.zeros_like(probs)
        keep[order[:top_k]] = probs[order[:top_k]]
        probs = keep / keep.sum()
    if top_p is not None:
        cumulative = np.cumsum(probs[order])
        cutoff = min(int(np.searchsorted(cumulative, top_p, side="left")), len(probs) - 1)
        keep = np.zeros_like(probs)
        keep[order[: cutoff + 1]] = probs[order[: cutoff + 1]]
        probs = keep / keep.sum()
    return probs


def sample_token(logits, rng, temperature=1.0, top_k=None, top_p=None) -> int:
    if temperature == 0:
        return int(np.argmax(logits))
    probs = next_token_probs(logits, temperature, top_k, top_p)
    return int(rng.choice(len(probs), p=probs))


def generate(params, prompt, n_new, rng, temperature=0.0, top_k=None, top_p=None,
             use_cache=True, linear=plain_linear) -> list[int]:
    tokens = [int(t) for t in prompt]
    new: list[int] = []
    cache = new_cache(params) if use_cache else None
    if use_cache:
        for t in tokens:
            logits = decode_step(params, t, cache, linear)
    for _ in range(n_new):
        if not use_cache:
            logits = forward(params, tokens, linear)[-1]
        nxt = sample_token(logits, rng, temperature, top_k, top_p)
        tokens.append(nxt)
        new.append(nxt)
        if use_cache:
            logits = decode_step(params, nxt, cache, linear)
    return new


# ---------------------------------------------------------------------------
# Stage 4: speculative decoding.
# ---------------------------------------------------------------------------


def make_prob_fn(params, temperature=1.0, top_k=None, top_p=None, linear=plain_linear):
    def prob_fn(tokens):
        logits = forward(params, tokens, linear)
        return np.stack([next_token_probs(row, temperature, top_k, top_p) for row in logits])

    return prob_fn


def residual_distribution(p, q) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    left = np.maximum(p - q, 0.0)
    total = left.sum()
    if total <= 0:
        return p.copy()
    return left / total


def speculative_round(target_fn, draft_fn, context, k, rng):
    context = [int(t) for t in context]
    drafted, q_rows = [], []
    for _ in range(k):
        q = draft_fn(context + drafted)[-1]
        drafted.append(int(rng.choice(len(q), p=q)))
        q_rows.append(q)
    p_rows = target_fn(context + drafted)[len(context) - 1:]
    out, n_accepted, n_checked = [], 0, 0
    for i, x in enumerate(drafted):
        p, q = p_rows[i], q_rows[i]
        n_checked += 1
        if rng.uniform() < min(1.0, p[x] / q[x]):
            out.append(x)
            n_accepted += 1
        else:
            out.append(int(rng.choice(len(p), p=residual_distribution(p, q))))
            return out, n_accepted, n_checked
    out.append(int(rng.choice(len(p_rows[k]), p=p_rows[k])))
    return out, n_accepted, n_checked


def speculative_generate(target_fn, draft_fn, prompt, n_new, k, rng):
    tokens = [int(t) for t in prompt]
    accepted = checked = rounds = 0
    while len(tokens) - len(prompt) < n_new:
        out, a, c = speculative_round(target_fn, draft_fn, tokens, k, rng)
        tokens.extend(out)
        accepted += a
        checked += c
        rounds += 1
    stats = {
        "accepted": accepted,
        "checked": checked,
        "target_passes": rounds,
        "acceptance_rate": accepted / checked if checked else 0.0,
    }
    return tokens[len(prompt): len(prompt) + n_new], stats


# ---------------------------------------------------------------------------
# Stage 5: a LoRA adapter, on the fly and merged.
# ---------------------------------------------------------------------------


def lora_linear(x, W, A, B, alpha, r):
    return x @ W.T + (alpha / r) * ((x @ A.T) @ B.T)


def merge_lora(W, A, B, alpha, r):
    return W + (alpha / r) * (B @ A)


def make_lora_linear(adapter):
    weights = adapter["weights"]

    def linear(x, W, key):
        if key in weights:
            A, B = weights[key]
            return lora_linear(x, W, A, B, adapter["alpha"], adapter["r"])
        return x @ W.T

    return linear


def merge_adapter(params, adapter):
    merged = copy.deepcopy(params)
    for (li, name), (A, B) in adapter["weights"].items():
        W = params["layers"][li][name]
        merged["layers"][li][name] = merge_lora(W, A, B, adapter["alpha"], adapter["r"])
    return merged


# ---------------------------------------------------------------------------
# Stage 6: the whole service.
# ---------------------------------------------------------------------------


def complete(prompt, merges, vocab, params, n_new, rng, temperature=0.0, top_k=None,
             top_p=None, adapter=None, merge=False) -> str:
    if adapter is not None and merge:
        params, linear = merge_adapter(params, adapter), plain_linear
    elif adapter is not None:
        linear = make_lora_linear(adapter)
    else:
        linear = plain_linear
    ids = encode(prompt, merges)
    new = generate(params, ids, n_new, rng, temperature, top_k, top_p, True, linear)
    return decode(ids + new, vocab)


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
