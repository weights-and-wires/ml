"""Checks for the tiny transformer project."""

import math

import torch

from mlcheck import check, close, equal, fail, has, shape

CFG = {"vocab": 11, "d_model": 16, "n_heads": 2, "n_layers": 2, "d_ff": 24}
_CACHE = {}


def _params(cfg=CFG, seed=0):
    return has(_SUB[0], "init_params")(cfg, seed)


_SUB = [None]


def _rand(*s, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*s, generator=g)


@check("sinusoidal_positions gives sin in even columns and cos in odd ones")
def test_positions(sub):
    _SUB[0] = sub
    pe = sub.sinusoidal_positions(5, 8)
    shape(pe.detach(), (5, 8), "positions")
    close(pe[0].detach(), torch.tensor([0.0, 1.0] * 4), atol=1e-6, label="row 0 (angle 0)")
    # Column pair 0 has speed 1; column pair 1 has speed 10000 ** (-2/8) = 0.1.
    close(float(pe[3, 0]), math.sin(3.0), atol=1e-5, label="sin(3) in column 0")
    close(float(pe[3, 1]), math.cos(3.0), atol=1e-5, label="cos(3) in column 1")
    close(float(pe[4, 2]), math.sin(4 * 0.1), atol=1e-5, label="sin(0.4) in column 2")


@check("causal_mask is 0 on and below the diagonal and -inf above it")
def test_mask(sub):
    m = sub.causal_mask(4)
    shape(m.detach(), (4, 4), "mask")
    for i in range(4):
        for j in range(4):
            want_blocked = j > i
            got_blocked = bool(torch.isinf(m[i, j]) and m[i, j] < 0)
            if want_blocked != got_blocked or (not want_blocked and float(m[i, j]) != 0.0):
                fail(f"mask[{i}, {j}] is {float(m[i, j])}; expected {'-inf' if want_blocked else '0'}")


def _reference_attention(x, Wq, Wk, Wv, Wo, H):
    B, T, d = x.shape
    dh = d // H
    outs = []
    for b in range(B):
        heads = []
        for h in range(H):
            sl = slice(h * dh, (h + 1) * dh)
            q, k, v = (x[b] @ Wq)[:, sl], (x[b] @ Wk)[:, sl], (x[b] @ Wv)[:, sl]
            s = q @ k.T / math.sqrt(dh)
            for i in range(T):
                s[i, i + 1:] = -1e30
            w = torch.softmax(s, dim=-1)
            heads.append(w @ v)
        outs.append(torch.cat(heads, dim=-1) @ Wo)
    return torch.stack(outs)


@check("causal_self_attention matches a head-by-head reference")
def test_attention_value(sub):
    d, H = 8, 2
    x = _rand(2, 5, d, seed=1)
    Ws = [_rand(d, d, seed=2 + i) * 0.5 for i in range(4)]
    got = sub.causal_self_attention(x, *Ws, H)
    shape(got.detach(), (2, 5, d), "attention output")
    close(got.detach(), _reference_attention(x, *Ws, H), atol=1e-5, rtol=1e-4,
          label="attention output (check the 1/sqrt(d_head) scale and the head split)")


@check("Attention output at position t does not change when later tokens change")
def test_attention_causal(sub):
    d, H = 8, 2
    x = _rand(1, 6, d, seed=3)
    Ws = [_rand(d, d, seed=4 + i) * 0.5 for i in range(4)]
    base = sub.causal_self_attention(x, *Ws, H)
    x2 = x.clone()
    x2[:, 4:] = _rand(1, 2, d, seed=9)
    changed = sub.causal_self_attention(x2, *Ws, H)
    close(changed[:, :4].detach(), base[:, :4].detach(), atol=1e-6,
          label="positions 0-3 after rewriting positions 4-5 (the mask is leaking the future)")
    if torch.allclose(changed[:, 4:], base[:, 4:], atol=1e-6):
        fail("positions 4-5 did not change when their own inputs changed")


@check("transformer_block is a pre-norm residual: zero output matrices return x")
def test_block(sub):
    p = _params()
    x = _rand(2, 5, 16, seed=5)
    zeroed = {k: v.detach().clone() for k, v in p.items()}
    for k in ("l0.Wo", "l0.Wd"):
        zeroed[k] = torch.zeros_like(zeroed[k])
    close(sub.transformer_block(x, zeroed, "l0.", 2).detach(), x, atol=1e-6,
          label="block output with Wo = 0 and Wd = 0 (the residual stream must pass through)")
    out = sub.transformer_block(x, p, "l0.", 2)
    shape(out.detach(), (2, 5, 16), "block output")
    if torch.allclose(out, x, atol=1e-4):
        fail("block with real weights returned x unchanged; the branches add nothing")


@check("forward returns logits of shape (B, T, vocab) and never looks ahead")
def test_forward_causal(sub):
    p = _params()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2, 1]])
    logits = sub.forward(p, ids, CFG)
    shape(logits.detach(), (2, 7, 11), "logits")
    ids2 = ids.clone()
    ids2[:, 4:] = torch.tensor([[9, 9, 9], [0, 0, 0]])
    logits2 = sub.forward(p, ids2, CFG)
    close(logits2[:, :4].detach(), logits[:, :4].detach(), atol=1e-5,
          label="logits at positions 0-3 after changing tokens 4-6 (a token saw its future)")


@check("forward uses position: swapping two earlier tokens changes the last logits")
def test_forward_position(sub):
    p = _params()
    a = torch.tensor([[1, 2, 3, 4, 5]])
    b = torch.tensor([[2, 1, 3, 4, 5]])
    la, lb = sub.forward(p, a, CFG)[0, -1], sub.forward(p, b, CFG)[0, -1]
    if torch.allclose(la, lb, atol=1e-5):
        fail("the same tokens in a different order gave identical last-position logits; "
             "add sinusoidal_positions to the embeddings")


@check("answer_loss scores only the masked positions, in value and in gradient")
def test_answer_loss(sub):
    logits = _rand(2, 3, 4, seed=6).requires_grad_()
    targets = torch.tensor([[0, 1, 2], [3, 0, 1]])
    mask = torch.tensor([[False, True, True], [False, False, True]])
    got = sub.answer_loss(logits, targets, mask)
    logp = torch.log_softmax(logits.detach(), dim=-1)
    picked = [-logp[0, 1, 1], -logp[0, 2, 2], -logp[1, 2, 1]]
    close(float(got), float(sum(picked) / 3), atol=1e-6,
          label="answer_loss (the mean over the 3 scored positions, not over all 6)")
    got.backward()
    if float(logits.grad[0, 0].abs().sum()) != 0.0 or float(logits.grad[1, :2].abs().sum()) != 0.0:
        fail("positions outside the mask received a gradient; they must contribute nothing")


@check("parameter_count and training_flops follow their formulas")
def test_counts(sub):
    for cfg in (CFG, {"vocab": 11, "d_model": 32, "n_heads": 4, "n_layers": 3, "d_ff": 64}):
        p = _params(cfg)
        want = sum(t.numel() for t in p.values())
        equal(int(sub.parameter_count(cfg)), want, f"parameter_count for d={cfg['d_model']}, L={cfg['n_layers']}")
    tiny = {"vocab": 3, "d_model": 2, "n_heads": 1, "n_layers": 1, "d_ff": 4}
    # 3*2 embed + (4*4 + 3*2*4 + 4*2) layer + 2*2 final norm = 6 + 48 + 4
    equal(int(sub.parameter_count(tiny)), 58, "parameter_count for the tiny hand example")
    close(float(sub.training_flops(tiny, 1000)), 6 * 58 * 1000.0, label="training_flops")


def _trained(sub, seed=0):
    key = (id(sub), seed)
    if key not in _CACHE:
        torch.manual_seed(seed)
        data = sub.make_data(seed)
        cfg = {"vocab": 11, "d_model": 32, "n_heads": 4, "n_layers": 2, "d_ff": 64}
        p = sub.init_params(cfg, seed)
        tcfg = dict(sub.DEFAULT_TRAIN, seed=seed)
        p, losses = sub.train(p, data["train"], cfg, tcfg)
        _CACHE[key] = (p, losses, data, cfg)
    return _CACHE[key]


@check("train lowers the answer loss: last-20-step average below 0.05")
def test_train_loss(sub):
    p, losses, data, cfg = _trained(sub)
    equal(len(losses), sub.DEFAULT_TRAIN["steps"], "number of recorded losses")
    first = sum(losses[:5]) / 5
    last = sum(losses[-20:]) / 20
    if not math.isfinite(last) or last > 0.05:
        fail(f"average loss over the last 20 steps is {last:.3f}; expected below 0.05 "
             f"(it started near {first:.2f}, about ln(10) = 2.3 for a guess among 10 digits)")
    if not all(isinstance(v, torch.Tensor) and v.requires_grad for v in p.values()):
        fail("train must return every weight as a tensor with requires_grad=True")


@check("greedy_decode keeps the prompt and the trained model reverses held-out strings")
def test_exact_match(sub):
    p, losses, data, cfg = _trained(sub)
    test = data["test"]
    out = sub.greedy_decode(p, test[:5, :7], 6, cfg)
    shape(out.detach(), (5, 13), "decoded sequences")
    equal(out[:, :7].detach(), test[:5, :7], "the first 7 tokens (the prompt) of the decoded sequences")
    acc = sub.exact_match(p, test, cfg)
    if acc < 0.95:
        bad = sub.greedy_decode(p, test[:3, :7], 6, cfg)
        fail(f"exact match on {len(test)} held-out strings is {acc:.3f}; expected at least 0.95. "
             f"First rows decoded: {bad.tolist()}, wanted {test[:3].tolist()}")
