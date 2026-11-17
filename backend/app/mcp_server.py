"""An MCP server that gives AI agents the whole course.

The server speaks the MCP Streamable HTTP transport at `/mcp`, statelessly and
with plain JSON responses, so each request stands alone and needs no session.
It exposes lessons, projects, exercises, solutions, and the course documents as
tools, lessons and exercises as resources, and the caller's progress, drafts,
runs, and compute status.

Every request needs `Authorization: Bearer <token>`, where the token comes
from `mcp_tokens`. The token alone decides who the caller is. The SSO headers
that `auth.current_user` trusts are never read here, because Caddy routes
`/mcp` without Authelia and so can't vouch for them.
"""

from __future__ import annotations

import functools
import json
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Literal

import anyio
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import Receive, Scope, Send

from . import course, curriculum, db
from .config import REPO_ROOT, get_settings
from .mcp_tokens import user_for_token
from .runner import runpod as runpod_backend
from .runs import start_run

# The ASGI scope key that carries the authenticated reader to the tools.
USER_SCOPE_KEY = "ml.mcp_user"

COURSE_TITLE = "Machine learning"
COURSE_DESCRIPTION = (
    "A hands-on course that teaches machine learning to undergraduates by having "
    "them build it. Chapters run from scoring a guess to a diffusion model: "
    "classical models, trees and ensembles, unsupervised learning, neural "
    "networks, vision and sequences, transformers, large language models, and "
    "generative models. Each chapter teaches one topic, works a small example by "
    "hand, and links an auto-graded Python exercise whose checks verify the "
    "mathematics. Projects close the course's parts and apply several chapters to "
    "a realistic problem."
)

INSTRUCTIONS = (
    f"{COURSE_DESCRIPTION}\n\n"
    "Start with `course_overview` for the syllabus and the reader's progress. Use "
    "`search` to find where a topic is taught, `get_lesson` to read a chapter or a "
    "project, and `get_exercise` for its brief, starter code, and checks. "
    "`run_exercise` runs code as the reader, saves it as their draft, and on a "
    "passing check marks the lesson solved. Only call `get_solution` when the "
    "reader asks for the answer."
)

# Documents an agent may read with `get_document`. Nothing else under docs/ is served.
DOCUMENTS = {
    "CURRICULUM.md": "The curriculum: every part and chapter, and why they are in that order.",
    "CONTENT_CONTRACT.md": "The format every lesson, exercise, and project follows.",
}

# Bounds that keep a tool result readable.
MAX_CODE_LENGTH = 200_000
MAX_STREAM_CHARS = 16_000
MAX_CHECK_TEXT = 4_000
MAX_SEARCH_RESULTS = 50
MAX_RUNS = 100

mcp = FastMCP(
    name="ml-course",
    instructions=INSTRUCTIONS,
    stateless_http=True,
    json_response=True,
    # The bearer token is the protection here. DNS-rebinding checks guard
    # unauthenticated servers on localhost, and would reject `ml.ifkash.dev`.
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


# Plumbing ---------------------------------------------------------------


def _tool(*, writes: bool = False, destructive: bool = False) -> Callable[[Callable[..., Any]], Any]:
    """Register a tool that returns JSON text only.

    Structured output would send every result twice, as text and as an object,
    which doubles a lesson's size for no gain.
    """
    return mcp.tool(
        structured_output=False,
        annotations=ToolAnnotations(
            readOnlyHint=not writes, destructiveHint=destructive, openWorldHint=False
        ),
    )


def _threaded(fn: Callable[..., Any]) -> Callable[..., Awaitable[Any]]:
    """Run a blocking tool body (SQLite, file reads) off the event loop."""

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        return await anyio.to_thread.run_sync(functools.partial(fn, *args, **kwargs))

    return wrapper


def _user(ctx: Context) -> str:
    request = ctx.request_context.request
    user = request.scope.get(USER_SCOPE_KEY) if request is not None else None
    if not user:
        # The endpoint authenticates before the server runs, so this means a
        # transport that bypassed it.
        raise ToolError("Not authenticated.")
    return str(user)


def _exercise_or_error(exercise_id: str) -> dict[str, Any]:
    exercise = curriculum.get_exercise(exercise_id)
    if exercise is None:
        raise ToolError(
            f"No exercise with ID {exercise_id!r}. Call `list_exercises` for the IDs."
        )
    return exercise


def _lessons_for(exercise_id: str) -> list[str]:
    return [
        lesson["slug"]
        for lesson in curriculum.lesson_list()
        if lesson.get("exercise") == exercise_id
    ]


def _clip(text: str, limit: int) -> str:
    """Keep the start and the end of long output, where the useful parts are."""
    if len(text) <= limit:
        return text
    head = limit // 4
    tail = limit - head
    omitted = len(text) - head - tail
    return f"{text[:head]}\n... [{omitted} characters omitted] ...\n{text[-tail:]}"


def _numbers() -> dict[str, str]:
    """The number the site shows for each lesson: 1, 2, 3 for chapters and P1, P2 for projects."""
    numbers: dict[str, str] = {}
    chapters = projects = 0
    for lesson in curriculum.lesson_list():
        if lesson["kind"] == "project":
            projects += 1
            numbers[lesson["slug"]] = f"P{projects}"
        else:
            chapters += 1
            numbers[lesson["slug"]] = str(chapters)
    return numbers


def _after_chapter(slug: str) -> int | None:
    """The chapter a project follows, from its slug: `010p-...` follows chapter 10."""
    match = re.match(r"^(\d+)p-", slug)
    return int(match.group(1)) if match else None


def _lesson_summary(
    lesson: dict[str, Any], progress: dict[str, Any], numbers: dict[str, str]
) -> dict[str, Any]:
    return {
        "slug": lesson["slug"],
        "number": numbers.get(lesson["slug"]),
        "kind": lesson["kind"],
        "title": lesson["title"],
        "part": lesson["part"],
        "summary": lesson["summary"],
        "minutes": lesson.get("minutes"),
        "exercise": lesson.get("exercise"),
        "gpu": lesson.get("gpu", False),
        "status": progress.get(lesson["slug"], {}).get("status"),
    }


# Course and lessons -----------------------------------------------------


@_tool()
@_threaded
def course_overview(ctx: Context) -> dict[str, Any]:
    """Get the whole syllabus and the reader's progress in one call.

    Returns the course title and description; every part with its lessons
    (slug, number, kind, title, summary, minutes, linked exercise ID, and the
    reader's status: null, "started", or "solved"); counts of chapters,
    projects, and exercises; the PDF book's details; and a progress summary.
    The projects form the last part, named "Projects". Call this first to
    orient yourself, then `get_lesson` with a slug to read a chapter.
    """
    user = _user(ctx)
    progress = db.get_progress(user)
    numbers = _numbers()
    parts = [
        {
            "part": part["part"],
            "lessons": [_lesson_summary(lesson, progress, numbers) for lesson in part["lessons"]],
        }
        for part in curriculum.parts()
    ]
    lessons = curriculum.lesson_list()
    return {
        "title": COURSE_TITLE,
        "description": COURSE_DESCRIPTION,
        "reader": user,
        "parts": parts,
        "counts": {
            "parts": len(parts),
            "chapters": sum(1 for lesson in lessons if lesson["kind"] != "project"),
            "projects": sum(1 for lesson in lessons if lesson["kind"] == "project"),
            "exercises": len(curriculum.load_exercises()),
        },
        "book": course.book_info(get_settings()),
        "progress": course.progress_summary(user),
        "documents": DOCUMENTS,
    }


@_tool()
@_threaded
def list_lessons(
    ctx: Context,
    part: str | None = None,
    kind: Literal["all", "chapter", "project"] = "all",
) -> dict[str, Any]:
    """List lessons in course order, with their headings and objectives.

    Args:
        part: Only lessons whose part name contains this text, ignoring case,
            for example "Part 3" or "neural". Omit it for every lesson.
        kind: "chapter" for chapters only, "project" for projects only, or
            "all" (default) for both.

    Each lesson has its slug, number, kind, title, part, summary, minutes,
    objectives, prerequisites, `##`/`###` headings, linked exercise ID, and the
    reader's status. Bodies are omitted; call `get_lesson` for the full text.
    """
    progress = db.get_progress(_user(ctx))
    numbers = _numbers()
    needle = (part or "").strip().lower()
    lessons = []
    for index, lesson in enumerate(curriculum.lesson_list(), start=1):
        if needle and needle not in str(lesson["part"]).lower():
            continue
        if kind != "all" and lesson["kind"] != kind:
            continue
        entry = _lesson_summary(lesson, progress, numbers)
        entry.update(
            index=index,
            objectives=lesson["objectives"],
            prereqs=lesson["prereqs"],
            headings=lesson["headings"],
        )
        lessons.append(entry)
    return {"count": len(lessons), "lessons": lessons}


def _lesson_payload(lesson: dict[str, Any], user: str) -> dict[str, Any]:
    payload = dict(lesson)
    payload["number"] = _numbers().get(lesson["slug"])
    exercise_id = lesson.get("exercise")
    exercise = curriculum.get_exercise(str(exercise_id)) if exercise_id else None
    payload["exercise_title"] = exercise["title"] if exercise else None
    payload["status"] = db.get_progress(user).get(lesson["slug"], {}).get("status")
    return payload


@_tool()
@_threaded
def get_lesson(ctx: Context, slug: str) -> dict[str, Any]:
    """Read one lesson, chapter or project: its full Markdown body and front matter.

    Args:
        slug: The lesson's slug, such as "001-loss-and-honest-scores".

    Returns the title, part, kind, summary, minutes, objectives, prerequisites,
    headings, the Markdown `body` (math is LaTeX between `$` or `$$`, and
    figures are inline SVG), the previous and next slugs within its kind, its
    position, the linked exercise ID and title, and the reader's status.
    Reading a lesson here doesn't mark it started.
    """
    lesson = curriculum.get_lesson(slug)
    if lesson is None:
        raise ToolError(f"No lesson with slug {slug!r}. Call `list_lessons` for the slugs.")
    return _lesson_payload(lesson, _user(ctx))


_HEADING_LINE = re.compile(r"^#{1,3}[ \t]+(.+?)[ \t#]*$", re.MULTILINE)


def _snippet(text: str, at: int, length: int) -> str:
    start = max(0, at - 80)
    end = min(len(text), at + length + 160)
    snippet = re.sub(r"\s+", " ", text[start:end]).strip()
    return ("..." if start > 0 else "") + snippet + ("..." if end < len(text) else "")


def _section(body: str, at: int) -> str | None:
    """The heading of the section that position `at` falls in."""
    heading = None
    for match in _HEADING_LINE.finditer(body):
        if match.start() > at:
            break
        heading = match.group(1).strip()
    return heading


def _read_document(name: str) -> str | None:
    path = REPO_ROOT / "docs" / name
    return path.read_text(encoding="utf-8") if path.is_file() else None


def _search_documents() -> list[dict[str, Any]]:
    documents = []
    for lesson in curriculum.load_lessons():
        documents.append(
            {
                "kind": lesson["kind"],
                "slug": lesson["slug"],
                "title": str(lesson["title"]),
                "part": lesson["part"],
                "summary": str(lesson.get("summary", "")),
                "headings": " \n".join(h["text"] for h in lesson["headings"]),
                "body": lesson["body"],
            }
        )
    for exercise in curriculum.load_exercises().values():
        documents.append(
            {
                "kind": "exercise",
                "slug": exercise["id"],
                "title": str(exercise["title"]),
                "part": None,
                "summary": "",
                "headings": "",
                "body": "\n".join([str(exercise["brief"]), *map(str, exercise["hints"])]),
            }
        )
    for name, description in DOCUMENTS.items():
        text = _read_document(name)
        if text is not None:
            documents.append(
                {
                    "kind": "document",
                    "slug": name,
                    "title": name,
                    "part": None,
                    "summary": description,
                    "headings": " \n".join(m.group(1) for m in _HEADING_LINE.finditer(text)),
                    "body": text,
                }
            )
    return documents


@_tool()
@_threaded
def search(query: str, limit: int = 10) -> dict[str, Any]:
    """Search chapters, projects, exercises, and the course documents by text, ignoring case.

    Args:
        query: Words to find, such as "bias variance" or "kernel trick". A
            document matches when it contains every word; an exact phrase match
            ranks higher, as do matches in titles and headings.
        limit: The most results to return, 1 to 50. Default 10.

    Each result has its kind ("chapter", "project", "exercise", or "document"),
    slug, title, part, the section heading the hit falls under, and a short
    snippet around the hit. Pass a chapter or project slug to `get_lesson`, an
    exercise's slug (its ID) to `get_exercise`, and a document's slug (its
    file name) to `get_document`. Exercise results match the brief and hints.
    """
    phrase = " ".join(query.lower().split())
    terms = phrase.split()
    if not terms:
        raise ToolError("The query is empty.")
    limit = max(1, min(int(limit), MAX_SEARCH_RESULTS))

    results = []
    for doc in _search_documents():
        title = doc["title"].lower()
        headings = doc["headings"].lower()
        body_lower = doc["body"].lower()
        haystack = " ".join((title, doc["summary"].lower(), headings, body_lower))
        if not all(term in haystack for term in terms):
            continue
        score = 0.0
        if phrase in title:
            score += 20
        score += 6 * sum(term in title for term in terms)
        if phrase in headings:
            score += 8
        score += 2 * sum(term in headings for term in terms)
        score += min(body_lower.count(phrase), 10) * 1.5
        score += sum(min(body_lower.count(term), 10) for term in terms) * 0.2

        at = body_lower.find(phrase)
        length = len(phrase)
        if at < 0:
            at = next((body_lower.find(t) for t in terms if t in body_lower), -1)
            length = len(terms[0])
        if at >= 0:
            snippet = _snippet(doc["body"], at, length)
            section = _section(doc["body"], at)
        else:
            snippet = doc["summary"]
            section = None
        results.append(
            {
                "kind": doc["kind"],
                "slug": doc["slug"],
                "title": doc["title"],
                "part": doc["part"],
                "section": section,
                "snippet": snippet,
                "score": round(score, 2),
            }
        )
    results.sort(key=lambda result: result["score"], reverse=True)
    return {"query": query, "total": len(results), "results": results[:limit]}


# Projects ---------------------------------------------------------------


@_tool()
@_threaded
def list_projects(ctx: Context) -> dict[str, Any]:
    """List the hands-on projects, in course order.

    Each project applies several chapters to a realistic problem. It has its
    slug, number (P1, P2, and so on), title, part, summary, minutes,
    objectives, the prerequisite lessons, the chapter it follows
    (`after_chapter`), its exercise ID, and the reader's status. A project is a
    lesson, so call `get_lesson` or `get_project` for the write-up.
    """
    progress = db.get_progress(_user(ctx))
    numbers = _numbers()
    projects = []
    for lesson in curriculum.lesson_list():
        if lesson["kind"] != "project":
            continue
        entry = _lesson_summary(lesson, progress, numbers)
        entry.update(
            objectives=lesson["objectives"],
            prereqs=lesson["prereqs"],
            after_chapter=_after_chapter(lesson["slug"]),
        )
        projects.append(entry)
    return {"count": len(projects), "projects": projects}


@_tool()
@_threaded
def get_project(ctx: Context, slug: str) -> dict[str, Any]:
    """Read one project: its front matter and the full Markdown write-up.

    Args:
        slug: The project's slug, from `list_projects`, such as
            "002p-project-explore-before-you-model".

    Returns what `get_lesson` returns, plus `after_chapter`.
    """
    lesson = curriculum.get_lesson(slug)
    if lesson is None or lesson["kind"] != "project":
        raise ToolError(f"No project with slug {slug!r}. Call `list_projects` for the slugs.")
    payload = _lesson_payload(lesson, _user(ctx))
    payload["after_chapter"] = _after_chapter(slug)
    return payload


# Exercises --------------------------------------------------------------


@_tool()
@_threaded
def list_exercises(ctx: Context) -> dict[str, Any]:
    """List every exercise with its brief, so you can pick one without reading each lesson.

    Each has its ID, title, brief, timeout in seconds, whether it needs a GPU,
    the packages it imports, the lessons that link to it, and the reader's
    status on those lessons (null, "started", or "solved"). Call `get_exercise`
    for the hints, starter code, and checks.
    """
    progress = db.get_progress(_user(ctx))
    exercises = []
    for exercise in curriculum.load_exercises().values():
        lessons = _lessons_for(exercise["id"])
        status = None
        for slug in lessons:
            state = progress.get(slug, {}).get("status")
            if state == "solved" or (state == "started" and status is None):
                status = state
        exercises.append(
            {
                "id": exercise["id"],
                "title": exercise["title"],
                "brief": exercise["brief"],
                "timeout": exercise["timeout"],
                "needs_gpu": exercise["needs_gpu"],
                "packages": exercise["packages"],
                "lessons": lessons,
                "status": status,
            }
        )
    return {"count": len(exercises), "exercises": exercises}


@_tool()
@_threaded
def get_exercise(exercise_id: str) -> dict[str, Any]:
    """Get an exercise: its brief, hints, starter code, and the checks' source.

    Args:
        exercise_id: The exercise ID, which a lesson names in its `exercise`
            field. It usually equals the lesson slug.

    Returns the title, brief, hints, `starter` (the file the reader edits),
    `tests` (the source of tests.py, the checks a `check` run applies), the
    timeout in seconds, whether it needs a GPU, the packages it imports, and the
    lessons that link to it. The worked solution is withheld; see `get_solution`.
    """
    exercise = _exercise_or_error(exercise_id)
    payload = curriculum.public_exercise(exercise)
    tests = Path(exercise["directory"]) / "tests.py"
    payload["tests"] = tests.read_text(encoding="utf-8") if tests.is_file() else ""
    payload["lessons"] = _lessons_for(exercise_id)
    return payload


@_tool()
@_threaded
def get_solution(exercise_id: str) -> dict[str, Any]:
    """Get an exercise's worked solution.

    Args:
        exercise_id: The exercise ID.

    The solution gives the answer away. Only fetch it when the reader asks to
    see it, or to compare after they have solved the exercise.
    """
    exercise = _exercise_or_error(exercise_id)
    return {"id": exercise_id, "solution": exercise["solution"]}


# Documents and the book -------------------------------------------------


@_tool()
@_threaded
def get_document(name: str) -> dict[str, Any]:
    """Read one of the course's reference documents, in Markdown.

    Args:
        name: "CURRICULUM.md" for the curriculum and the reasoning behind its
            order, or "CONTENT_CONTRACT.md" for the format every lesson,
            exercise, and project follows. Case and the ".md" are optional.
    """
    wanted = name.strip().upper().removesuffix(".MD") + ".md"
    if wanted not in DOCUMENTS:
        raise ToolError(f"No document {name!r}. Choose one of: {', '.join(DOCUMENTS)}.")
    text = _read_document(wanted)
    if text is None:
        raise ToolError(f"{wanted} isn't on this server.")
    return {"name": wanted, "description": DOCUMENTS[wanted], "markdown": text}


@_tool()
@_threaded
def get_book() -> dict[str, Any]:
    """Get the details of the course's PDF book, if this server has one.

    Returns `book`, which has the title, page count, size in bytes, build time,
    commit, and the site-relative download `url`, or null when the server has
    no PDF. The PDF itself isn't sent over MCP; the reader downloads it in a
    browser, signed in. Its text is the same as the lessons `get_lesson` returns.
    """
    return {"book": course.book_info(get_settings())}


# The reader's state -----------------------------------------------------


@_tool()
@_threaded
def get_progress(ctx: Context) -> dict[str, Any]:
    """Get the reader's progress: per-lesson status, totals, and recent runs.

    `lessons` lists every lesson the reader has started or solved, in course
    order, with its status and when it last changed (Unix seconds). `summary`
    has the solved count, the total, and the lesson touched last. `stats` has
    the solved count, run count, and compute seconds. `runs` has the 10 most
    recent runs, newest first; see `list_runs` for more.
    """
    user = _user(ctx)
    progress = db.get_progress(user)
    numbers = _numbers()
    lessons = [
        {
            "slug": lesson["slug"],
            "number": numbers.get(lesson["slug"]),
            "title": lesson["title"],
            "status": progress[lesson["slug"]]["status"],
            "updated_at": progress[lesson["slug"]]["updated_at"],
        }
        for lesson in curriculum.lesson_list()
        if lesson["slug"] in progress
    ]
    return {
        "reader": user,
        "summary": course.progress_summary(user),
        "lessons": lessons,
        "stats": db.stats(user),
        "runs": db.recent_runs(user, limit=10),
    }


@_tool()
@_threaded
def list_runs(
    ctx: Context, limit: int = 25, exercise_id: str | None = None
) -> dict[str, Any]:
    """List the reader's recent runs, newest first.

    Args:
        limit: The most runs to return, 1 to 100. Default 25.
        exercise_id: Only this exercise's runs. Omit it for every exercise.

    Each run has its exercise ID, mode ("run" or "check"), backend ("local" or
    "runpod"), `ok` (1 or 0), checks passed and total, seconds, and when it
    started (Unix seconds).
    """
    limit = max(1, min(int(limit), MAX_RUNS))
    runs = db.recent_runs(_user(ctx), limit=limit, exercise_id=exercise_id)
    return {"count": len(runs), "runs": runs}


@_tool()
@_threaded
def get_draft(ctx: Context, exercise_id: str) -> dict[str, Any]:
    """Get the reader's saved code for an exercise.

    Args:
        exercise_id: The exercise ID.

    `code` is null when the reader has no draft; the editor then shows the
    starter code from `get_exercise`.
    """
    _exercise_or_error(exercise_id)
    return {"exercise_id": exercise_id, "code": db.get_draft(_user(ctx), exercise_id)}


@_tool(writes=True)
@_threaded
def save_draft(ctx: Context, exercise_id: str, code: str) -> dict[str, Any]:
    """Save code as the reader's draft for an exercise.

    Args:
        exercise_id: The exercise ID.
        code: The whole file, which replaces the reader's current draft. The
            reader sees it in the editor the next time they open the lesson.
    """
    _exercise_or_error(exercise_id)
    if len(code) > MAX_CODE_LENGTH:
        raise ToolError(f"The code is longer than {MAX_CODE_LENGTH} characters.")
    db.save_draft(_user(ctx), exercise_id, code)
    return {"exercise_id": exercise_id, "saved": True}


@_tool(writes=True, destructive=True)
@_threaded
def reset_draft(ctx: Context, exercise_id: str) -> dict[str, Any]:
    """Delete the reader's saved code for an exercise, so the editor shows the starter again.

    Args:
        exercise_id: The exercise ID.

    This discards the reader's work on the exercise. Call it only when they ask.
    """
    _exercise_or_error(exercise_id)
    db.delete_draft(_user(ctx), exercise_id)
    return {"exercise_id": exercise_id, "saved": False}


@_tool(writes=True)
@_threaded
def set_progress(
    ctx: Context, slug: str, status: Literal["started", "solved"]
) -> dict[str, Any]:
    """Mark a lesson started or solved for the reader.

    Args:
        slug: The lesson's slug.
        status: "started" or "solved". A solved lesson is never downgraded to
            started. A passing `run_exercise` check marks its lesson solved on
            its own, so you rarely need this.
    """
    if curriculum.get_lesson(slug) is None:
        raise ToolError(f"No lesson with slug {slug!r}.")
    user = _user(ctx)
    db.set_progress(user, slug, status)
    return {"slug": slug, "status": db.get_progress(user)[slug]["status"]}


@_tool()
@_threaded
def export_my_data(ctx: Context) -> dict[str, Any]:
    """Export everything stored for the reader: progress, every draft, and every run.

    This is what the site's export download contains, and it can be large
    because it includes the full text of each draft. Prefer `get_progress`,
    `get_draft`, and `list_runs` for a single fact.
    """
    return json.loads(db.export_all(_user(ctx)))


# Running code -----------------------------------------------------------


@_tool()
async def get_compute() -> dict[str, Any]:
    """Get the runner's limits and whether a GPU endpoint is available.

    `local` has the number of parallel workers, the memory limit in MB (0 means
    none beyond the container's), and the longest timeout in seconds. `runpod`
    says whether a GPU endpoint is configured and reachable and, if so, its
    GPU type and worker and job counts. Lessons marked `gpu` fall back to the
    CPU runner when no endpoint is configured.
    """
    settings = get_settings()
    return {
        "local": {
            "workers": settings.local_workers,
            "memory_mb": settings.local_memory_mb,
            "timeout_ceiling": settings.local_timeout_ceiling,
        },
        "runpod": await runpod_backend.probe(settings),
    }


@_tool(writes=True)
async def run_exercise(
    ctx: Context,
    exercise_id: str,
    code: str,
    mode: Literal["check", "run"] = "check",
    prefer_gpu: bool = False,
) -> dict[str, Any]:
    """Run code for an exercise on the course's runner, as the reader.

    Args:
        exercise_id: The exercise ID.
        code: The whole submission file, starting from the starter code.
        mode: "check" runs the exercise's checks against the code and reports
            each one. "run" executes the file and returns its output.
        prefer_gpu: Ask for the GPU backend when one is configured, as the
            editor's GPU toggle does. Exercises that need a GPU use it anyway.

    This does what the editor's Run and Check buttons do: it saves the code as
    the reader's draft, records the run, and marks the linked lessons solved
    when every check passes. It waits for the run to finish, which can take up
    to the exercise's timeout.

    Returns `ok`, `passed` and `total` checks, `seconds`, the backend used,
    each check's description, status, and failure message, the program's
    stdout and stderr (long output is trimmed in the middle), and any errors.
    Figures a program draws are left out.
    """
    user = _user(ctx)
    if len(code) > MAX_CODE_LENGTH:
        raise ToolError(f"The code is longer than {MAX_CODE_LENGTH} characters.")
    settings = get_settings()
    run = await anyio.to_thread.run_sync(
        lambda: start_run(user, exercise_id, mode, code, settings, prefer_gpu=prefer_gpu)
    )
    if run is None:
        raise ToolError(
            f"No exercise with ID {exercise_id!r}. Call `list_exercises` for the IDs."
        )

    stdout: list[str] = []
    stderr: list[str] = []
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    status: list[str] = []
    done: dict[str, Any] = {"ok": False}
    async for event in run.events:
        kind = event.get("type")
        if kind == "stdout":
            stdout.append(str(event.get("text", "")))
        elif kind == "stderr":
            stderr.append(str(event.get("text", "")))
        elif kind == "check":
            check = {
                "index": event.get("index"),
                "description": event.get("description"),
                "status": event.get("status"),
                "seconds": event.get("seconds"),
            }
            for key in ("message", "output"):
                if event.get(key):
                    check[key] = _clip(str(event[key]), MAX_CHECK_TEXT)
            checks.append(check)
        elif kind == "error":
            errors.append(str(event.get("message", "")))
        elif kind == "status":
            status.append(str(event.get("text", "")))
        elif kind == "done":
            done = event

    return {
        "exercise_id": exercise_id,
        "mode": mode,
        "backend": run.backend,
        "ok": bool(done.get("ok")),
        "passed": done.get("passed"),
        "total": done.get("total"),
        "seconds": done.get("seconds"),
        "checks": checks,
        "stdout": _clip("".join(stdout), MAX_STREAM_CHARS),
        "stderr": _clip("".join(stderr), MAX_STREAM_CHARS),
        "errors": errors,
        "status": status,
        "lessons_solved": _lessons_for(exercise_id) if mode == "check" and done.get("ok") else [],
    }


# Resources --------------------------------------------------------------


@mcp.resource(
    "ml://lessons/{slug}",
    name="lesson",
    description="A chapter's or project's Markdown body. The slug comes from `list_lessons`.",
    mime_type="text/markdown",
)
@_threaded
def lesson_resource(slug: str) -> str:
    lesson = curriculum.get_lesson(slug)
    if lesson is None:
        raise ValueError(f"No lesson with slug {slug!r}.")
    return lesson["body"]


@mcp.resource(
    "ml://exercises/{exercise_id}",
    name="exercise-starter",
    description="An exercise's starter file. The ID comes from `list_exercises`.",
    mime_type="text/x-python",
)
@_threaded
def exercise_resource(exercise_id: str) -> str:
    exercise = curriculum.get_exercise(exercise_id)
    if exercise is None:
        raise ValueError(f"No exercise with ID {exercise_id!r}.")
    return exercise["starter"]


# The HTTP endpoint ------------------------------------------------------

_manager: StreamableHTTPSessionManager | None = None


@asynccontextmanager
async def lifespan() -> AsyncIterator[None]:
    """Run the MCP session manager for the life of the app.

    A session manager runs only once, so each app start makes a new one. That
    lets tests start the app more than once in one process.
    """
    global _manager
    manager = StreamableHTTPSessionManager(
        app=mcp._mcp_server,
        json_response=True,
        stateless=True,
        security_settings=mcp.settings.transport_security,
    )
    async with manager.run():
        _manager = manager
        try:
            yield
        finally:
            _manager = None


def _unauthorized(error: str | None, description: str) -> JSONResponse:
    challenge = 'Bearer realm="ml-mcp"'
    if error:
        challenge += f', error="{error}"'
    return JSONResponse(
        {"error": error or "unauthorized", "error_description": description},
        status_code=401,
        headers={"WWW-Authenticate": challenge},
    )


class _Endpoint:
    """The ASGI app at `/mcp`: check the bearer token, then hand over to MCP.

    A class rather than a function, because Starlette wraps a plain function
    route as a request handler instead of mounting it as an ASGI app.
    """

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await _serve(scope, receive, send)


async def _serve(scope: Scope, receive: Receive, send: Send) -> None:
    headers = Headers(scope=scope)
    scheme, _, token = headers.get("authorization", "").partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token:
        response = _unauthorized(None, "Send an MCP token as 'Authorization: Bearer <token>'.")
        await response(scope, receive, send)
        return
    user = await anyio.to_thread.run_sync(user_for_token, token)
    if user is None:
        response = _unauthorized("invalid_token", "The token is unknown or revoked.")
        await response(scope, receive, send)
        return
    if _manager is None:
        response = JSONResponse({"error": "The MCP server isn't running."}, status_code=503)
        await response(scope, receive, send)
        return
    await _manager.handle_request({**scope, USER_SCOPE_KEY: user}, receive, send)


endpoint = _Endpoint()
