# Content contract

Every lesson in this course is two things on disk: one Markdown file that the
reader reads, and one exercise directory that the reader's code is checked
against. This document is the exact contract both must satisfy. The loader in
`backend/app/curriculum.py` and the harness in `backend/app/runner/harness.py`
depend on it.

## Audience

Readers hold a working background in calculus, linear algebra, and probability.
They know what a derivative, a matrix, and a probability distribution are. They
read a lesson to learn what a method does, why it works, and when it fails --
not to reconstruct its proofs. Write for them:

- Lead with the result and what it means. State the gradient, the bound, or the
  update rule, then say in one sentence what it tells you.
- Keep the mathematics the exercise needs, and cut the rest. Every formula the
  reader implements has to appear in the lesson. A derivation that no check
  depends on belongs in a linked reference, or nowhere.
- Replace a long derivation with the one step that carries the idea. When a
  result takes six lines to prove, give the line that explains why it holds and
  state the rest as fact. Say where the full derivation can be found.
- Define every symbol you use, including ones a graduate course would assume.
- Prefer the concrete to the general. A worked two-by-two example beats an
  index expression that covers every case.
- State assumptions precisely: when a result needs convexity, full rank, i.i.d.
  sampling, or a bounded second moment, say so. Reducing the mathematics does
  not mean hiding the conditions a result depends on.
- Connect each result to what it costs to compute: flops, memory, and how the
  cost scales.

Do not pad. Cutting a derivation is not licence to fill the space with prose.

## Prose style

Follow the Google developer documentation style guide:

- Address the reader as "you". Use active voice and present tense.
- One idea per sentence. Split any sentence that needs a second comma-spliced
  clause to land. Long subordinate chains are the main thing that made these
  lessons hard to read.
- Sentence case for every heading. Start task headings with a bare infinitive.
- Serial commas. Em dashes with no surrounding spaces.
- Code, filenames, and identifiers in code font. No `simply`, `just`, `easily`,
  `obviously`, `note that`, `e.g.`, `i.e.`, `etc.`
- Don't write timelessly-rotting words: `currently`, `now`, `new`, `latest`,
  `recently`, `state of the art`.
- Introduce every list, table, and code block with a complete sentence.

## Lesson file

Path: `content/lessons/NN-slug.md`, where `NN` is the two-digit order prefix and
`slug` matches the `slug` field.

The file starts with YAML front matter:

```yaml
---
title: Logistic regression
slug: 05-logistic-regression
part: "Part 1 — Foundations"
summary: One sentence, under 140 characters, that says what the lesson delivers.
minutes: 55
gpu: false
objectives:
  - Derive the gradient and Hessian of the logistic negative log-likelihood.
  - Prove the loss is convex, and say when it is strictly convex.
  - Explain why separable data sends the MLE to infinity.
prereqs:
  - 02-linear-regression
  - 03-gradient-descent
exercise: 05-logistic-regression
---
```

Every value that holds a colon must be wrapped in double quotes, or YAML reads
the item as a mapping rather than a string. A mapping that reaches the client
renders as nothing at all, so `scripts/check_content.py` rejects one.

Every field is required except `prereqs`, which can be omitted. `gpu` is `true`
only when the exercise genuinely needs a GPU. `minutes` is an honest reading and
working estimate.

The body is Markdown with these conventions:

- One `#` heading, matching `title`, as the first line after the front matter.
- `##` for sections, `###` for subsections. Never skip a level.
- Math in KaTeX: `$inline$` and `$$display$$`. The renderer loads
  `remark-math` and `rehype-katex`, so use KaTeX-supported commands only.
  `\operatorname`, `\mathbb`, `\mathcal`, `\begin{bmatrix}`, `\begin{aligned}`,
  `\top`, `\lVert`, and `\argmin` via `\operatorname*{arg\,min}` all work.
- Fenced code blocks tagged with a language. Python blocks show the shape of an
  implementation, not a full solution to the exercise.
- Tables for genuinely two-dimensional data.

Every lesson body includes, in this order:

1. An opening that states the problem the lesson solves and why the previous
   lesson is not enough. Two or three paragraphs. No heading.
2. `## What you already know` — a short list of the results the lesson builds on,
   stated as facts rather than explained. Keep it to five bullets.
3. The substance, in as many `##` sections as the material needs. State each
   result, explain what it means, and show only the mathematics the reader
   needs to implement it.
4. `## Cost and scaling` — flops, memory, and how both scale with the relevant
   dimensions. A table works well.
5. `## Where this breaks` — the failure modes, and what practitioners do about
   them.
6. `## The exercise` — two or three paragraphs describing what the reader
   implements and what the checks verify. Name the functions by their exact
   identifiers.

Aim for 1000 to 1800 words of body, plus math.

## Exercise directory

Path: `content/exercises/<id>/`, where `<id>` matches the lesson's `exercise`
field. It holds exactly four files.

### `meta.json`

```json
{
  "id": "05-logistic-regression",
  "title": "Fit a logistic regression two ways",
  "brief": "Implement the gradient and the Newton step, and watch them disagree on iteration count.",
  "needs_gpu": false,
  "timeout": 60,
  "packages": ["numpy"],
  "hints": [
    "The gradient is X^T (sigmoid(Xw) - y); no sum over samples is needed if you keep it as a matrix product.",
    "Newton's step solves (X^T D X + eps I) delta = grad. Add the ridge term or the solve fails on separable data."
  ]
}
```

`packages` is documentation only; the runtime image already carries `numpy`,
`scipy`, and `torch` on the CPU. `timeout` is in seconds and must leave headroom:
set it to about four times the reference solution's runtime, and never above 120
for a CPU exercise.

### `starter.py`

What the reader sees when the lesson opens. It must:

- Import only from the standard library, `numpy`, and, when the lesson needs it,
  `torch`.
- Define every function the checks call, with the exact signature, a docstring
  that states the contract including array shapes, and a body that raises
  `NotImplementedError`.
- Carry a `if __name__ == "__main__":` block that builds a small example and
  prints something informative. The **Run** button executes this block, so it must
  work once the reader fills in the functions, and it must fail with a clear
  `NotImplementedError` before that.
- Never contain the answer, in a comment or anywhere else.

### `solution.py`

A correct, idiomatic reference implementation with the same public surface as
`starter.py`. It must pass every check in `tests.py`. Keep it clean enough to read
as the worked answer, because the reader can reveal it.

### `tests.py`

The checks. Import the harness helpers and register each check with the
`@check` decorator:

```python
"""Checks for the logistic regression exercise."""

import numpy as np

from mlcheck import check, close, fail


@check("The sigmoid is numerically stable at large magnitudes")
def test_sigmoid_stability(sub):
    z = np.array([-800.0, 0.0, 800.0])
    out = sub.sigmoid(z)
    if not np.all(np.isfinite(out)):
        fail(f"sigmoid returned non-finite values: {out}")
    close(out, np.array([0.0, 0.5, 1.0]), atol=1e-12)


@check("Newton converges in under 10 iterations on a well-conditioned problem")
def test_newton(sub):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    w_true = np.array([1.5, -2.0, 0.5, 0.0])
    y = (rng.uniform(size=200) < sub.sigmoid(X @ w_true)).astype(float)
    w, history = sub.fit_newton(X, y, max_iter=25)
    if len(history) > 10:
        fail(f"took {len(history)} iterations, expected fewer than 10")
    close(sub.sigmoid(X @ w).mean(), y.mean(), atol=0.05)
```

The `mlcheck` module is on the path at run time. Its API:

| Name | Signature | Behaviour |
|---|---|---|
| `check` | `@check("description")` | Registers a check. The description is what the reader sees. |
| `close` | `close(got, want, rtol=1e-5, atol=1e-8, label=None)` | Fails with a readable diff when the arrays differ. |
| `equal` | `equal(got, want, label=None)` | Exact equality, for shapes, dtypes, and integers. |
| `fail` | `fail("message")` | Fails the check with that message. |
| `shape` | `shape(array, (3, 4), label=None)` | Asserts a shape. |
| `almost` | `almost(got, want, tol, label=None)` | Scalar comparison within `tol`. |
| `timed` | `with timed("matmul") as t:` | Records elapsed seconds, reported to the reader. |

Rules for checks:

- Each check receives `sub`, the reader's module, as its only argument.
- Seed every random number generator. `np.random.default_rng(0)` for numpy,
  `torch.manual_seed(0)` for torch. A check that passes intermittently is a bug.
- Write six to ten checks per exercise, ordered from the smallest piece to the
  whole. The first check must pass as soon as the reader gets one function right,
  so the pane shows progress early.
- Check the mathematics, not the implementation. Verify a gradient against a
  finite-difference approximation rather than against a particular loop.
- Never import `solution` from `tests.py`.
- Keep the whole suite under the `timeout` in `meta.json`, with margin. On CPU
  that means matrices no larger than about 2000 by 200.
- Failure messages must say what was expected and what arrived. The reader cannot
  see your code.

## Mathematical accuracy

Every result, gradient, and complexity claim in a lesson must be correct. A
lesson that states a result without deriving it carries the same burden as one
that derives it: if it is on the page, it is right. When you write a gradient in
prose, the check that verifies it in `tests.py` must use finite differences, so
the prose and the code confirm each other. If those two disagree, the lesson is
wrong, not the reader.

Cutting a derivation must never change what a result says. Shortening the path
to a formula is editing; changing the formula is breaking the lesson.
