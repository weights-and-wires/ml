# Machine learning

A hands-on course that teaches machine learning to undergraduates by having them
build it. Eighty-seven chapters run from scoring a guess to a diffusion model. Each one
teaches a single topic, two or three closely related ideas in one arc: it starts from a concrete
question, shows a picture, works a small example by hand, and only then gives
the formula and walks you through implementing it, in an editor beside the
text, against checks that verify the results rather than the shape of your
code.

Eight projects, on a page of their own linked from the header, put that work
to use on realistic problems. The first teaches you to load, clean, plot, and
question a dataset before you model it, and the others each draw on a whole
part. Every project says which chapter to finish before you start it.

The reading pane is on the left and the editor is on the right, the way a
programming-problem site arranges them. Fold the editor away with the button in
its header, or with Cmd or Ctrl and the period key, to read at full width. The
**PDF book** button in the header downloads the whole course as one book. **Run** executes your file and streams
its output. **Check** runs the exercise's checks and reports them one at a time
as they finish.

## Who it is for

You have taken a first course in calculus, linear algebra, and probability, even
if you remember little of it, and you can write Python. You don't need any
machine learning background. The lessons define every term and symbol when it
first appears, skip proofs, and show each formula next to the numpy code that
computes it. Each lesson ends with a step-by-step guide to its exercise.

## What is in here

| Path | Contents |
|---|---|
| `content/lessons/` | Eighty-seven chapters and eight projects, in Markdown with YAML front matter and inline SVG figures. |
| `content/renames.json` | Each slug from earlier editions and the chapters that replaced it, so progress carries over. |
| `content/exercises/` | One directory per exercise: a starter file, checks, and a worked solution. |
| `backend/` | FastAPI: auth, content, progress, drafts, and the run stream. |
| `backend/app/runner/` | The harness, the assertion library the checks use, and the two execution backends. |
| `frontend/` | React and Vite: the reader, the editor, and the check console. |
| `frontend/src/book/` | The print layout for the PDF book, and `frontend/scripts/build-book.mjs`, which prints it. |
| `gpu/` | The RunPod serverless worker, for the lessons that want a GPU. |
| `docs/` | The content contract and the curriculum. |
| `scripts/` | Content validation, the book's content dump, and the deploy script. |

## The curriculum

**Part 1 — Foundations.** Scoring a guess, honest test scores, bias and
variance, least squares and why it loses digits, gradient descent and momentum,
ridge and lasso, logistic regression, exponential families, and the Bayesian reading of ridge and lasso.

**Part 2 — Classical models.** Cross-validation, feature pipelines and
leakage, learning curves, learning theory, generative classifiers, convex
duality, support vector machines, kernels, and Gaussian processes.

**Part 3 — Trees, ensembles, and unsupervised learning.** Decision trees,
bagging and boosting, PCA and the SVD, ICA and t-SNE, k-means, EM for Gaussian
mixtures, and nearest neighbours with matrix factorisation.

**Part 4 — Neural networks.** Why layers need a bend, how many units a curve
costs, backpropagation, SGD and Adam, initialisation and normalisation,
regularisation, debugging and tuning a network, and uncertainty with conformal
prediction before the digit project.

**Part 5 — Vision and sequences.** The map from your own tensor onto PyTorch,
then convolutions, residual networks, augmentation, transfer learning, patch
embedding, detection and segmentation, contrastive learning, recurrent networks,
LSTMs and GRUs, attention for sequence to sequence, and graph neural networks.

**Part 6 — Transformers.** Self-attention, online softmax and tiled attention,
position encodings, the transformer block, modern design choices, encoders and
cross-attention, state-space models and linear attention, a count model and word
vectors, and training one, with gradient accumulation and activation
checkpointing to fit it in memory.

**Part 7 — Large language models.** Tokenisation, pretraining data, scaling
laws, optimiser scaling rules, inference and decoding, evaluation, text embeddings
and retrieval, LoRA, distillation, alignment, mixtures of experts, and interpretability.

**Part 8 — Generative models.** A linear autoencoder keeps the first principal
component, and a variational autoencoder draws a row from a standard normal code.
A GAN trains a generator against a discriminator, and FID, precision, and
recall score the result. A diffusion model noises that cloud on a schedule and walks a standard normal back onto it. A diffusion language model hides tokens behind a mask and unmasks the sure ones first.

[The full list, with what each exercise delivers](docs/CURRICULUM.md).

## Run it locally

You need Python 3.12 or later and Node 20 or later.

1. Copy the environment file and configure the shared SSO proxy secret:

   ```bash
   cp .env.example .env
   python3 -c "import secrets; print(secrets.token_urlsafe(48))"
   ```

   Set the generated value as `SSO_PROXY_SECRET` in `.env` and as
   `ML_SSO_PROXY_SECRET` in Caddy's environment. They must match.
2. Install and start the backend:

   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -r backend/requirements.txt
   .venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
   .venv/bin/uvicorn backend.app.main:app --reload --port 8000
   ```

3. In another terminal, start the frontend:

   ```bash
   cd frontend && npm install && npm run dev
   ```

Open `http://localhost:5173`. The Vite dev server proxies `/api` to the backend
on port 8000. Authenticated development requires routing the site through the
trusted Caddy/Authelia proxy; a direct backend request has no trusted identity
and private endpoints return 401. Do not simulate identity with browser-supplied
headers.

## Write a lesson

`docs/CONTENT_CONTRACT.md` is the binding specification: the file layout, the
front matter, the required sections, the prose style, and the `mlcheck` API the
checks are written against.

Validate what you write:

```bash
.venv/bin/python scripts/check_content.py 008-sigmoid-and-log-loss
```

```bash
node scripts/check_math.mjs content/lessons/008-sigmoid-and-log-loss.md
```

That checks the front matter, the required sections, the word count, and the
figures; runs
every check against `solution.py`, which must pass all of them; and runs them
against `starter.py`, which must fail at least one. Omit the slug to validate
everything. The same script runs in CI on every push.

To look at a chapter's figures without starting the site, render them to PNG:

```bash
node scripts/render_figures.mjs /tmp/figs content/lessons/003-least-squares-and-projection.md
```

`check_math.mjs` compiles every math segment with the frontend's own KaTeX, so
math that fails only at render time fails here instead. Run `npm install` in
`frontend/` first.

## How a run works

Pressing **Run** or **Check** posts your code to `/api/run`, which streams
server-sent events back until the run finishes. The event types are `begin`,
`status`, `plan`, `stdout`, `stderr`, `check`, `error`, and `done`.

The submission executes in a subprocess in its own process group, started with
`python -I`, under a CPU-time limit and a wall clock, with the container's
`mem_limit` bounding memory. When the wall clock runs out the whole process
group is killed. The harness wraps the reader's own prints as events, so a print
that happens to look like JSON cannot be confused with a harness event.

This is a resource boundary, not a security boundary. Anyone who can sign in can
execute code in the container. The app is for two named people on a private
host; do not open it wider without putting the runner in a sandbox that isolates
the filesystem and the network.

## Authentication

The site uses the central Authelia portal at `https://auth.ifkash.dev/`. When a
visitor is not authenticated, the app redirects there with the exact current
URL as the `rd` return destination. Caddy runs Authelia `forward_auth` before
proxying requests, strips incoming identity headers, and forwards the vetted
`Remote-User` identity plus `X-SSO-Proxy`. Set the same unpredictable
`SSO_PROXY_SECRET` in the app environment and `ML_SSO_PROXY_SECRET` in Caddy;
without it, the app rejects all identities. Private APIs and the PDF remain
protected by the identity dependency. `SSO_USERS` can map portal usernames to
existing progress identities, for example `ifkash:enderboi,ananya:michu`.

## The GPU backend

One chapter, `069-training-loop-and-cost`, is marked `gpu: true`. Its checks run on the CPU like every other exercise, and
the GPU is there for the larger run in the starter file's `__main__` block.

`backend/app/runner/runpod.py` sends a submission to a RunPod serverless
endpoint and streams the worker's events back. The worker in `gpu/` carries the
same harness and the same exercise directories, so a lesson behaves identically
on either backend.

With `RUNPOD_ENDPOINT_ID` unset, the GPU lessons fall back to the local CPU
runner and everything still works. That is how the deployment starts: the worker
image is built and `RUNPOD_API_KEY` is set, and one step is left.

1. The publish workflow already builds and pushes the worker. It runs on every
   push that touches `gpu/`, `content/exercises/`, or the harness, and tags
   `ghcr.io/weights-and-wires/ml-gpu-worker:latest`.

2. **Give RunPod read access to that image.** A package published from Actions
   starts private, and RunPod pulls anonymously. Make the package public under
   the organisation's **Packages** settings, the same way
   `kashifulhaque/learn-inference-worker` is public. The alternative is a
   container registry credential in the RunPod console holding a GitHub token
   with `read:packages`, which hands that token to RunPod.

3. Create the endpoint and read back its ID:

   ```bash
   RUNPOD_API_KEY=... python3 scripts/runpod_endpoint.py
   ```

   The script refuses to run while the image is private, and says so. Pass
   `--skip-pull-check` when you took the registry-credential route instead.

4. Put `RUNPOD_ENDPOINT_ID` in the deployed `.env` and restart:

   ```bash
   ssh ifkash@vm.ifkash.dev "cd ~/docs/courses/ml && docker compose up -d"
   ```

`/api/compute` reports what the endpoint has: whether it is reachable, how many
workers are warm, and how many jobs are queued.

## Deploy

The VM runs the app from a container image, behind the shared Caddy on the
external `edge` network. This bundle never publishes 80 or 443 itself.

A deploy has three steps:

1. Merge to `main`.
2. GitHub Actions builds the image and pushes it to
   `ghcr.io/weights-and-wires/ml`.
3. On the VM, run `deploy ml`. It pulls the new image and restarts the
   container.

The VM never builds from source and holds no copy of the tree. `.env` on the VM
holds the SSO proxy secret and RunPod key, and a deploy never touches it.
`Caddyfile.example` holds the vhost block, including the Authelia forward-auth
and streaming settings. `docker-compose.yml` is an example of the service
definition.

The image also prints the PDF book, in a build stage that holds Chromium and
the print layout's dependencies. The final image carries only the PDF and its
manifest. The manifest's `commit` is `unknown` in the image, because the book
stage doesn't receive `GIT_SHA`. For more information, see
[The PDF book](#the-pdf-book).

To reach the app before its DNS record resolves, tunnel the container's
localhost port:

```bash
ssh -L 8093:127.0.0.1:8093 ifkash@vm.ifkash.dev
```

## The PDF book

The **PDF** button in the header downloads the whole course as one book: every
lesson, every exercise with its starter file and hints, and the solutions in an
appendix. The book has a contents page with page numbers, a bookmark for every
lesson and section, and running heads. Like the lessons, the download requires
an authenticated Authelia identity through Caddy.

To build the book, run the following command:

```bash
node frontend/scripts/build-book.mjs
```

The build needs Google Chrome or Chromium, found through `CHROME_PATH` or in
the usual install locations. It dumps the lessons with the backend's own loader
(`scripts/dump_book.py`), renders the print layout in `frontend/src/book/` with
the site's Markdown pipeline, and prints it with headless Chrome. It prints
more than once, so that the contents carry real page numbers. The PDF and its
`manifest.json` land in `book/`, which Git ignores.

The build fails on any formula KaTeX can't render and on anything wider than
the page, because Chrome shrinks every page to fit the widest element. It skips
itself when no lesson or layout file has changed; to rebuild anyway, pass
`--force`. The image build prints the book, so a deploy needs no local build.
CI also builds the book on every push and keeps it as a workflow artifact for
seven days.

The lockfile must come from npm 10, because the image's `npm ci` rejects a
lockfile that npm 11 writes. After you change a dependency, run
`npx npm@10 install` in `frontend/`.

## Licence

MIT. See [LICENSE](LICENSE).
