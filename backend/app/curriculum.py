"""Loads lessons and exercises from `content/`.

A lesson is one Markdown file with YAML front matter. An exercise is a directory
holding `meta.json`, `starter.py`, `tests.py`, and `solution.py`. Both are cached
after the first read; `reload_content` clears the cache while you are writing.

A lesson with `draft: true` is served on a local run and omitted from a deployed
image. The deployed image is the one whose `GIT_SHA` is the commit it was built
from. A local run leaves that setting at `unknown`.

The format is specified in `docs/CONTENT_CONTRACT.md`.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import frontmatter

from .config import get_settings


def _text_list(value: Any) -> list[str]:
    """Coerce a front-matter list to strings.

    An unquoted colon in a YAML list item parses as a mapping rather than a
    string, and a mapping reaching the client renders as nothing at all — React
    refuses to render an object and blanks the page. `scripts/check_content.py`
    rejects that at authoring time; this keeps a slip from taking the app down.
    """
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            out.extend(f"{k}: {v}" for k, v in item.items())
        else:
            out.append(str(item))
    return out


def _lessons_dir() -> Path:
    return get_settings().content_dir / "lessons"


def _exercises_dir() -> Path:
    return get_settings().content_dir / "exercises"


def renames() -> dict[str, list[str]]:
    """Old lesson slugs and the slugs that replaced them, from `renames.json`.

    When a lesson is split or renamed, its entry here lets readers keep their
    progress. A missing file means nothing was renamed.
    """
    path = get_settings().content_dir / "renames.json"
    if not path.is_file():
        return {}
    return {str(k): [str(s) for s in v] for k, v in json.loads(path.read_text()).items()}


def _is_draft(meta: dict[str, Any]) -> bool:
    """True only when the front matter sets `draft` to a boolean true."""
    return meta.get("draft") is True


@lru_cache
def _all_lessons() -> list[dict[str, Any]]:
    """Every lesson file, drafts included, ordered by filename."""
    lessons: list[dict[str, Any]] = []
    directory = _lessons_dir()
    if not directory.is_dir():
        return lessons
    for path in sorted(directory.glob("*.md")):
        post = frontmatter.load(path)
        meta = dict(post.metadata)
        slug = str(meta.get("slug") or path.stem)
        lessons.append(
            {
                "slug": slug,
                "title": meta.get("title", slug),
                "part": meta.get("part", "Uncategorised"),
                "summary": meta.get("summary", ""),
                "minutes": meta.get("minutes"),
                "objectives": _text_list(meta.get("objectives", [])),
                "prereqs": _text_list(meta.get("prereqs", [])),
                "exercise": meta.get("exercise"),
                "gpu": bool(meta.get("gpu", False)),
                # A project closes a part; chapters and projects are numbered apart.
                "kind": "project" if meta.get("kind") == "project" else "chapter",
                "draft": _is_draft(meta),
                "body": post.content,
            }
        )
    return lessons


def read_lessons(*, include_drafts: bool) -> list[dict[str, Any]]:
    """Lesson records. Pass `include_drafts=False` to drop unfinished lessons."""
    lessons = _all_lessons()
    if include_drafts:
        return lessons
    return [lesson for lesson in lessons if not lesson["draft"]]


def draft_exercise_ids() -> set[str]:
    """Exercise ids that belong to a lesson marked `draft: true`."""
    return {
        str(lesson["exercise"])
        for lesson in _all_lessons()
        if lesson["draft"] and lesson.get("exercise")
    }


@lru_cache
def load_lessons() -> list[dict[str, Any]]:
    """Lessons the running app serves.

    A local run includes drafts. A deployed image omits them. The choice is
    `Settings.production`, which reads the `GIT_SHA` value deploy already sets.
    """
    return read_lessons(include_drafts=not get_settings().production)


def lesson_list() -> list[dict[str, Any]]:
    """Lesson metadata without the body, for the sidebar and the dashboard."""
    return [{k: v for k, v in lesson.items() if k != "body"} for lesson in load_lessons()]


PROJECTS = "Projects"


def parts() -> list[dict[str, Any]]:
    """Chapters grouped into parts, in the order the parts first appear.

    Projects leave their part and close the list as one section of their own,
    in the order of the chapters each one needs.
    """
    grouped: list[dict[str, Any]] = []
    index: dict[str, dict[str, Any]] = {}
    for lesson in lesson_list():
        part = PROJECTS if lesson["kind"] == "project" else lesson["part"]
        if part not in index:
            index[part] = {"part": part, "lessons": []}
            grouped.append(index[part])
        index[part]["lessons"].append(lesson)
    return sorted(grouped, key=lambda group: group["part"] == PROJECTS)


def get_lesson(slug: str) -> dict[str, Any] | None:
    """One lesson with its body, and the slugs on either side of it.

    `index`, `count`, `prev`, and `next` place it among lessons of its own
    kind, so chapters read "Lesson 24 of 132" and projects "Project 1 of 8".
    """
    lessons = load_lessons()
    order = {lesson["slug"]: i for i, lesson in enumerate(lessons)}
    if slug not in order:
        return None
    lesson = dict(lessons[order[slug]])
    # Chapters and projects sit in separate lists, so each steps through its own.
    same_kind = [other["slug"] for other in lessons if other["kind"] == lesson["kind"]]
    i = same_kind.index(slug)
    lesson["prev"] = same_kind[i - 1] if i > 0 else None
    lesson["next"] = same_kind[i + 1] if i + 1 < len(same_kind) else None
    lesson["index"] = i + 1
    lesson["count"] = len(same_kind)
    return lesson


@lru_cache
def load_exercises() -> dict[str, dict[str, Any]]:
    exercises: dict[str, dict[str, Any]] = {}
    root = _exercises_dir()
    if not root.is_dir():
        return exercises
    for path in sorted(root.iterdir()):
        meta_file = path / "meta.json"
        if not meta_file.is_file():
            continue
        try:
            meta = json.loads(meta_file.read_text())
        except json.JSONDecodeError:
            continue
        exercise_id = str(meta.get("id", path.name))

        def read(name: str) -> str:
            target = path / name
            return target.read_text() if target.is_file() else ""

        exercises[exercise_id] = {
            "id": exercise_id,
            "title": meta.get("title", exercise_id),
            "brief": meta.get("brief", ""),
            "needs_gpu": bool(meta.get("needs_gpu", False)),
            "timeout": int(meta.get("timeout", 60)),
            "packages": meta.get("packages", []),
            "hints": meta.get("hints", []),
            "starter": read("starter.py"),
            "solution": read("solution.py"),
            "directory": str(path),
        }
    if get_settings().production:
        hidden = draft_exercise_ids()
        if hidden:
            return {key: value for key, value in exercises.items() if key not in hidden}
    return exercises


def get_exercise(exercise_id: str) -> dict[str, Any] | None:
    return load_exercises().get(exercise_id)


def public_exercise(exercise: dict[str, Any]) -> dict[str, Any]:
    """The exercise payload for the client, with the solution withheld."""
    return {k: v for k, v in exercise.items() if k not in ("solution", "directory")}


def reload_content() -> None:
    _all_lessons.cache_clear()
    load_lessons.cache_clear()
    load_exercises.cache_clear()
