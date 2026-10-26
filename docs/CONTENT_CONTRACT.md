# Content contract

Every lesson in this course is two things on disk: one Markdown file that the
reader reads, and one exercise directory that the reader's code is checked
against. This document is the exact contract both must satisfy. The loader in
`backend/app/curriculum.py` and the harness in `backend/app/runner/harness.py`
depend on it.

## Audience

Readers are undergraduates who are new to machine learning. They have taken a
first course in calculus, linear algebra, and probability, but they are rusty,
and dense notation stops them cold. They can write Python and have used numpy a
little. They read a chapter to understand an idea well enough to build it, and
they learn by building.

Students said an earlier version of this course was "easy to read but not easy
to understand", and that it threw things at them instead of building intuition.
Short sentences were not the problem. The problem was facts in a row with no
reason connecting them. Every rule below serves one goal: the reader should be
able to say *why* each step happens before they see *how* to compute it.

## Build intuition first

- One chapter, one idea. A chapter teaches a single idea well enough that the
  reader could explain it to a friend without notes. When a draft needs a
  second big idea, that idea is the next chapter.
- Start from a question the reader already cares about, with a concrete
  situation: a handful of houses and prices, a spam folder, a blurry photo.
  Say what goes wrong without the idea before you introduce it.
- Climb the ladder in order: picture, then words, then a tiny example with real
  numbers, then the formula, then code. The formula comes last, as a compact
  summary of something the reader already believes.
- Try the obvious approach first and show where it fails. "Why not just…?" is
  the question the reader is asking. Answer it before they have to.
- Explain the reason, not only the step. Connect sentences with "because",
  "so", and "which means". A paragraph is a short argument, not a list of
  facts. Prefer three connected sentences to five disconnected ones.
- Use one running example through the chapter, and reuse it across chapters
  of the same part where it fits, so the reader is not decoding new data every
  page.
- Read each formula aloud. After every display equation, say in plain English
  what each piece is and what the whole thing computes, then show it as one or
  two lines of numpy.
- Work one small example by hand. Use tiny numbers, two or three data points or
  a two-by-two matrix, and show every intermediate value, so the reader can
  check their code against it.
- Define every symbol and every term the first time it appears, including ones
  that feel basic, such as "gradient", "i.i.d.", or "norm". Prefer an ordinary
  word to jargon when both mean the same thing.
- Avoid heavy notation. No `\mathcal` sets, no measure-theoretic language, no
  proofs. Prefer `x`, `w`, and `y` with plain subscripts.
- Keep the conditions honest. When a result needs an assumption, say so in
  plain words and say what goes wrong without it.
- Close the loop. End the substance with a short "what you now know" paragraph
  in a `[!KEY]` callout that ties the idea back to the opening question.

Every result on the page must still be correct. Simpler wording must never
change what a result says.

## Figures

Every chapter has at least one figure, and most have two or three. A figure
earns its place when it shows something words struggle with: a shape (a loss
bowl, a decision boundary), a motion (a step downhill, a vector rotating), a
flow (data through layers, a gradient flowing back), or a comparison (before
and after, good and bad). Put the main figure near the top, where it carries
the chapter's central picture, and refer to it in the text.

A figure is a fenced block tagged `figure`. It holds one inline SVG, then a
caption on the lines after `</svg>`:

````markdown
```figure
<svg viewBox="0 0 560 220" role="img" aria-label="Three points and the best line">
  <line x1="40" y1="190" x2="520" y2="190" class="stroke-muted"/>
  <circle cx="120" cy="150" r="5" class="fill-a"/>
  <line x1="80" y1="170" x2="500" y2="40" class="stroke-b"/>
  <text x="430" y="40">best line</text>
</svg>
Figure 1. The best line passes between the points, not through them.
```
````

Rules for figures:

- Colour only with these classes, never with `fill="#..."`, `stroke="red"`, or
  a `style` colour, so the figure follows the light and dark themes and prints
  in the PDF book. Shapes draw nothing until a class colours them.
  - Fills: `fill-ink`, `fill-muted`, `fill-paper`, `fill-sunk`, `fill-a`,
    `fill-b`, `fill-c`, `fill-d`, `fill-bad`, and the pale area fills
    `fill-soft-a`, `fill-soft-b`, `fill-soft-c`, `fill-soft-d`.
  - Strokes: `stroke-ink`, `stroke-muted`, `stroke-grid`, `stroke-a`,
    `stroke-b`, `stroke-c`, `stroke-d`, `stroke-bad`, plus `dashed`.
  - Text is ink-coloured at 13 px by default. Add `fig-small`, `fig-bold`, or
    `fig-mono` as needed. `a` is blue, `b` is orange, `c` is green, `d` is
    purple. Use them consistently: `a` for data, `b` for the model's output.
- Set a `viewBox` about 560 to 640 units wide and no `width` or `height`. The
  figure scales to the column. Keep text at least 11 units high.
- Give the `svg` an `aria-label` that says what it shows.
- Draw from real numbers. If the figure plots the chapter's worked example,
  the points must be the example's points, computed, not eyeballed.
- Text inside SVG is plain text. KaTeX does not render there, so write
  `w1`, `x²`, or `σ` instead of `$w_1$`.
- Arrowheads need a `<marker>` in `<defs>` whose path carries a fill class.
  Give each marker an `id` prefixed with the chapter number, such as
  `c07-arrow`, because every figure on a page shares one document.
- Start the caption with `Figure N.` and say what the reader should notice,
  not only what is drawn.

## Prose style

Follow the Google developer documentation style guide:

- Address the reader as "you". Use active voice and present tense.
- Keep sentences short, but never at the cost of the connection between them.
  A sentence that says why something happens is worth its length.
- Sentence case for every heading. Start task headings with a bare infinitive.
- Serial commas. Em dashes with no surrounding spaces.
- Code, filenames, and identifiers in code font. No `simply`, `just`, `easily`,
  `obviously`, `note that`, `e.g.`, `i.e.`, `etc.`
- Don't write timelessly-rotting words: `currently`, `now`, `new`, `latest`,
  `recently`, `state of the art`.
- Introduce every list, table, and code block with a complete sentence.

## Lesson file

Path: `content/lessons/NNN-slug.md`, where `NNN` is the three-digit order prefix and
`slug` matches the `slug` field.

The file starts with YAML front matter:

```yaml
---
title: From a score to a probability
slug: 017-sigmoid-and-probability
part: "Part 1 — Foundations"
summary: One sentence, under 140 characters, that says what the lesson delivers.
minutes: 35
gpu: false
objectives:
  - Explain why a straight-line score makes a poor probability.
  - Turn a score into a probability with the sigmoid, and compute it without overflow.
  - Read a probability back as odds and log-odds.
prereqs:
  - 012-momentum
exercise: 017-sigmoid-and-probability
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
- Math in KaTeX: `$inline$` and `$$display$$`. Put each `$$` on its own line
  above and below the equation. A one-line `$$...$$` parses as inline math and
  renders at text size, so `scripts/check_content.py` rejects it. The renderer loads
  `remark-math` and `rehype-katex`, so use KaTeX-supported commands only.
  `\operatorname`, `\mathbb`, `\mathcal`, `\begin{bmatrix}`, `\begin{aligned}`,
  `\top`, `\lVert`, and `\argmin` via `\operatorname*{arg\,min}` all work.
- Fenced code blocks tagged with a language. Python blocks show the shape of an
  implementation, not a full solution to the exercise.
- Tables for genuinely two-dimensional data.
- Callouts, written as a blockquote whose first line is a type marker. The
  reader renders each type as a coloured box with its own label:

  ```markdown
  > [!KEY]
  > A trained model's error on its own training data is lower than its true
  > error, on average.
  ```

  | Marker | Label | Use it for |
  |---|---|---|
  | `[!KEY]` | Key idea | The one sentence to remember from a section. Two to four per lesson. |
  | `[!TIP]` | Tip | A practical shortcut or habit. At most two per lesson. |
  | `[!WARNING]` | Watch out | A trap the reader is about to fall into. At most two per lesson. |
  | `[!RECAP]` | Before you build | Three to five bullets, placed right before `## Build it step by step`. One per lesson. |

  `scripts/check_content.py` rejects any other marker, and a second recap.

Every lesson body includes, in this order:

1. An opening that says, in plain words, what problem the chapter solves and
   why it matters, starting from a concrete situation. Two or three short
   paragraphs. No heading, and no formulas.
2. `## What you already know` — a short list of the ideas the lesson builds on,
   each in one plain sentence. Keep it to five bullets.
3. The substance, in as many `##` sections as the material needs, following
   the ladder in "Build intuition first": picture, words, a tiny example, the
   formula, code. Include at least one small example worked by hand with real
   numbers, and at least one figure.
4. `## Build it step by step` — a numbered walkthrough of the exercise, one
   step per function in the order the checks test them. Each step says what the
   function takes and returns, the numpy operations it needs, and the mistake
   people most often make. It gives the shape of the code, never the full
   answer.
5. `## Cost and scaling` — optional. How the running time and memory grow as
   the data or the model gets bigger, in plain words. Include it when cost is
   part of the idea, and leave it out of a chapter where it is filler.
6. `## Where this breaks` — optional. The common ways the method fails in
   practice, and what people do about each one.
7. `## The exercise` — one or two paragraphs that say what the reader builds
   and what the checks verify. Name the functions by their exact identifiers.

Aim for 1000 to 1800 words of body, plus math and figures. A chapter is
a sitting of about 30 to 45 minutes, reading and building together. When a
draft runs past 2000 words, it holds two ideas: split it. `scripts/check_content.py`
rejects a chapter under 800 or over 2600 words, and one with no figure.

## Exercise directory

Path: `content/exercises/<id>/`, where `<id>` matches the lesson's `exercise`
field. It holds exactly four files.

### `meta.json`

```json
{
  "id": "017-sigmoid-and-probability",
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
`scipy`, `pandas`, `matplotlib`, and `torch` on the CPU. `timeout` is in seconds and must leave headroom:
set it to about four times the reference solution's runtime, and never above 120
for a CPU exercise.

### `starter.py`

What the reader sees when the lesson opens. It must:

- Import only from the standard library, `numpy`, and, when the lesson needs it,
  `torch`, `pandas`, or `matplotlib`. When **Run** calls `plt.show()`, each open
  figure appears in the console as an image, in order with the printed output.
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
| `finite` | `finite(array, label=None)` | Fails when any element is NaN or infinite. |
| `has` | `has(sub, "name")` | Returns `sub.name`, failing readably when the reader has not defined it. |
| `finite_difference` | `finite_difference(f, x, eps=1e-6)` | Central-difference gradient of a scalar function, for checking gradients. |
| `almost` | `almost(got, want, tol, label=None)` | Scalar comparison within `tol`. |
| `timed` | `with timed("matmul") as t:` | Records elapsed seconds, reported to the reader. |

Rules for checks:

- Each check receives `sub`, the reader's module, as its only argument.
- Seed every random number generator. `np.random.default_rng(0)` for numpy,
  `torch.manual_seed(0)` for torch. A check that passes intermittently is a bug.
- Write four to eight checks per exercise, ordered from the smallest piece to the
  whole. The first check must pass as soon as the reader gets one function right,
  so the pane shows progress early.
- Check the mathematics, not the implementation. Verify a gradient against a
  finite-difference approximation rather than against a particular loop.
- Never import `solution` from `tests.py`.
- Keep the whole suite under the `timeout` in `meta.json`, with margin. On CPU
  that means matrices no larger than about 2000 by 200.
- Failure messages must say what was expected and what arrived. The reader cannot
  see your code.

## Projects

The course has a set of projects, listed in their own section after the
chapters. A chapter teaches one idea; a project asks the
reader to combine the ideas of a whole part to answer a realistic question, the
way they would on the job. A project is a lesson file and an exercise directory,
like a chapter, and everything above applies unless this section says
otherwise.

### Where a project lives

Path: `content/lessons/NNNp-project-slug.md`, where `NNN` is the prefix of the
last chapter the project needs, usually the last chapter of its part. The app
and the PDF book list every project in one Projects section after the
chapters, in that order, and tell the reader "After chapter 23" for
`023p-project-price-houses`. The exercise directory has the same name.

One project needs only the first four chapters:
`004p-project-explore-before-you-model`, because looking at the data comes
before every model in the rest of the course. It teaches its own skills, loading, cleaning, plotting, and
summarising data, so unlike the other projects it explains each one rather
than only linking a chapter.

The front matter adds one field, and lists the chapters the project draws on:

```yaml
kind: project
prereqs:
  - 008-qr-and-svd
  - 014-ridge-svd-path-and-degrees-of-freedom
  - 019-gradient-descent-for-logistic
```

`part` is the part whose chapters the project draws on most; the reader shows
it beside the project's title. List at least three chapters in `prereqs`, every
one a real slug from that part or an earlier one. Set
`minutes` honestly; a project is two to four hours of work.

### What a project body holds

A project gives the reader a job to do, not a lecture. It reminds them of the
ideas they need and points to the chapter that taught each one, rather than
teaching them again. The body includes, in this order:

1. An opening that sets the scene: who needs what, with what data, and why
   the obvious approach is not good enough. Two or three short paragraphs.
2. `## The brief` — the deliverables as a short list, and how the work is
   judged. Say what a good answer looks like in numbers.
3. `## What you will use` — one bullet per idea the project draws on, each a
   plain sentence that links the chapter, such as
   `[ridge for every lambda](/lessons/014-ridge-svd-path-and-degrees-of-freedom)`.
4. `## Meet the data` — what each column or input means, how it was made,
   and a figure that shows it. The data is synthetic and generated by a seeded
   function in the starter, so no file is downloaded. Say plainly which
   pattern is planted in it, because the reader's result is judged against it.
5. `## Plan the build` — the pipeline as a figure (data in, each stage, the
   answer out), with a sentence on why the stages come in that order.
6. `## Build it step by step` — one numbered step per function, in the order
   the checks test them, with the same rules as a chapter: what goes in and
   out, which earlier chapter it reuses, and the mistake people most often
   make. Give the shape of the code, never the answer. Put the one `[!RECAP]`
   callout right before this section.
7. `## Read your results` — what numbers the reader should see when the
   pipeline works, what each one means for the person who asked, and what a
   wrong number usually points to.
8. `## Take it further` — three to five ungraded extensions, each one
   sentence of task and one sentence of what it will teach.
9. `## The exercise` — what the checks verify, naming each function by its
   identifier.

Aim for 1800 to 3200 words of body, plus math and figures, with at least two
figures. `scripts/check_content.py` rejects a project under 1200 or over 4000
words, or with fewer than two figures.

### The project exercise

- Write eight to twelve checks. Order them from the first stage of the
  pipeline to the last, and finish with one or two end-to-end checks that run
  the whole pipeline and judge the answer: a test error under a threshold, the
  planted clusters recovered, a generated string that follows the planted
  grammar.
- The starter ships `make_data`, or a function like it, already written. It
  is seeded, takes a `seed` argument, and builds the synthetic data in well
  under a second. It is the one function the reader does not write, so its
  docstring says what it plants.
- Every other public function raises `NotImplementedError` in the starter.
  Helpers the reader already wrote in a chapter, such as a stable sigmoid,
  may be given in full when re-deriving them adds nothing; say so in the
  docstring.
- Set thresholds in the end-to-end checks with margin: the reference solution
  must clear each one comfortably across seeds 0 to 4, and a plausible but
  wrong pipeline, such as one that skips the honest split or the
  standardisation, must miss it.
- Keep the whole suite within `timeout`, which may go up to 120 seconds on the
  CPU. Use numpy. Use torch only when the part is about networks large enough
  that writing every backward pass by hand would bury the point. Use pandas
  and matplotlib only where loading, reshaping, and plotting data is the
  point, as in the data-exploration project.

## Mathematical accuracy

Every result, gradient, and complexity claim in a lesson must be correct. A
lesson that states a result without deriving it carries the same burden as one
that derives it: if it is on the page, it is right. When you write a gradient in
prose, the check that verifies it in `tests.py` must use finite differences, so
the prose and the code confirm each other. If those two disagree, the lesson is
wrong, not the reader.

Cutting a derivation must never change what a result says. Shortening the path
to a formula is editing; changing the formula is breaking the lesson.
