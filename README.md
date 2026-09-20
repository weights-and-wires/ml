# Machine learning

A course that teaches machine learning to people who already have the
mathematics. Thirty-three lessons run from least squares to mixtures of experts.
Each one derives its result and then asks you to implement it, in an editor
beside the text, against checks that verify the mathematics rather than the
shape of your code.

The reading pane is on the left and the editor is on the right, the way a
programming-problem site arranges them. **Run** executes your file and streams
its output. **Check** runs the exercise's checks and reports them one at a time
as they finish.

## Who it is for

You know what a gradient, a Jacobian, an eigendecomposition, a convex function,
and a maximum likelihood estimator are. The lessons do not explain those. They
spend their space on derivations, on the assumptions a result needs, and on what
each result costs in flops and memory.

## What is in here

| Path | Contents |
|---|---|
| `content/lessons/` | Thirty-three lessons, in Markdown with YAML front matter. |
| `content/exercises/` | One directory per exercise: a starter file, checks, and a worked solution. |
| `backend/` | FastAPI: auth, content, progress, drafts, and the run stream. |
| `backend/app/runner/` | The harness, the assertion library the checks use, and the two execution backends. |
| `frontend/` | React and Vite: the reader, the editor, and the check console. |
| `gpu/` | The RunPod serverless worker, for the lessons that want a GPU. |
| `docs/` | The content contract and the curriculum. |
| `scripts/` | Content validation and the deploy script. |

## The curriculum

**Part 1 — Foundations.** The learning problem, linear regression, gradient
descent, ridge and lasso, logistic regression, exponential families and GLMs.

**Part 2 — Classical machine learning.** Generalisation and model selection,
generative classifiers, support vector machines, kernels, decision trees,
ensembles, PCA and the SVD, clustering and expectation maximisation.

**Part 3 — Neural networks.** Multilayer perceptrons, backpropagation as
reverse-mode automatic differentiation, optimisers, initialisation and
normalisation, regularisation.

**Part 4 — Vision and sequences.** Convolutions, residual networks, recurrent
networks and backpropagation through time, gated recurrence, sequence to
sequence with attention.

**Part 5 — Transformers.** Self-attention, the transformer block, training a
transformer.

**Part 6 — Large language models.** Tokenisation, scaling laws, inference and
decoding, parameter-efficient fine-tuning, alignment, mixtures of experts.

[The full list, with what each exercise delivers](docs/CURRICULUM.md).

## Run it locally

You need Python 3.12 or later and Node 20 or later.

1. Copy the environment file and set a session secret:

   ```bash
   cp .env.example .env
   python3 -c "import secrets; print(secrets.token_urlsafe(48))"
   ```

   Put that value in `SESSION_SECRET`.

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

Open `http://localhost:5173`. The Vite dev server proxies `/api` to the backend.
Sign in with one of the names in `APP_USERS` and the password in `APP_PASSWORD`.

## Write a lesson

`docs/CONTENT_CONTRACT.md` is the binding specification: the file layout, the
front matter, the required sections, the prose style, and the `mlcheck` API the
checks are written against.

Validate what you write:

```bash
.venv/bin/python scripts/check_content.py 05-logistic-regression
```

That checks the front matter, the required sections, and the word count; runs
every check against `solution.py`, which must pass all of them; and runs them
against `starter.py`, which must fail at least one. Omit the slug to validate
everything. The same script runs in CI on every push.

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

Two people share one password, set in `APP_PASSWORD`, and sign in under one of
the names in `APP_USERS`. The username is not a secret: it scopes progress,
drafts, and run history so that two readers keep separate state. A signed,
HTTP-only cookie carries the session. Set `COOKIE_SECURE=true` once the app is
served over HTTPS.

## The GPU backend

Two lessons — `21-residual-networks` and `27-training-a-transformer` — are
marked `gpu: true`. Their checks run on the CPU like every other exercise, and
the GPU is there for the larger run in the starter file's `__main__` block.

`backend/app/runner/runpod.py` sends a submission to a RunPod serverless
endpoint and streams the worker's events back. The worker in `gpu/` carries the
same harness and the same exercise directories, so a lesson behaves identically
on either backend.

With `RUNPOD_ENDPOINT_ID` unset, the GPU lessons fall back to the local CPU
runner and everything still works. To turn the GPU on:

1. Let the publish workflow build and push the worker. It runs on every push
   that touches `gpu/`, `content/exercises/`, or the harness, and tags
   `ghcr.io/weights-and-wires/ml-gpu-worker:latest`.

2. Give RunPod read access to that image. Either make the package public in the
   repository's package settings, or add a container registry credential in the
   RunPod console holding a GitHub token with `read:packages`.

3. In the RunPod console, create a serverless endpoint from that image on the
   GPU you want, and copy its ID.

4. Set `RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID` in the deployed `.env`, then
   `docker compose up -d`.

## Deploy

The VM holds a clone at `~/docs/ml` and runs it behind the shared Caddy on the
external `edge` network. This bundle never publishes 80 or 443 itself.

```bash
./scripts/deploy.sh
```

That pushes the current commit, fetches and rebuilds on the VM, and waits until
`/api/health` reports the commit it sent. `Caddyfile.example` holds the vhost
block, including the buffering and timeout settings the run stream needs.

To reach the app before its DNS record resolves, tunnel the container's
localhost port:

```bash
ssh -L 8093:127.0.0.1:8093 ifkash@vm.ifkash.dev
```

## Licence

MIT. See [LICENSE](LICENSE).
