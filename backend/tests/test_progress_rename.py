"""A progress row stored under a pre-shift slug resolves to the new slug."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.app import curriculum, db  # noqa: E402
from backend.app.config import get_settings  # noqa: E402

_SPEC = importlib.util.spec_from_file_location("insert_lesson", ROOT / "scripts" / "insert_lesson.py")
assert _SPEC is not None and _SPEC.loader is not None
tool = importlib.util.module_from_spec(_SPEC)
sys.modules["insert_lesson"] = tool
_SPEC.loader.exec_module(tool)

PART = "Part 1 — Foundations"


def _lesson(slug: str) -> str:
    return f"""---
title: {json.dumps(slug)}
slug: {slug}
part: {json.dumps(PART)}
summary: "A short fixture lesson."
minutes: 10
gpu: false
objectives:
  - Read the page.
  - Name the idea.
  - Point at the next lesson.
exercise: {slug}
---

# {slug}

A fixture paragraph.
"""


class ProgressRenameTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self._saved = {key: os.environ.get(key) for key in ("DATA_DIR", "CONTENT_DIR", "GIT_SHA")}
        lessons = self.root / "content" / "lessons"
        exercises = self.root / "content" / "exercises"
        lessons.mkdir(parents=True)
        exercises.mkdir(parents=True)
        for slug in ("001-alpha", "002-beta"):
            (lessons / f"{slug}.md").write_text(_lesson(slug), encoding="utf-8")
            directory = exercises / slug
            directory.mkdir()
            (directory / "meta.json").write_text(
                json.dumps({"id": slug, "title": slug, "brief": "b", "needs_gpu": False, "timeout": 30}) + "\n",
                encoding="utf-8",
            )
        (self.root / "content" / "renames.json").write_text(
            json.dumps({"legacy-beta": ["002-beta"]}, indent=1) + "\n", encoding="utf-8"
        )
        (self.root / "docs").mkdir()
        (self.root / "docs" / "CURRICULUM.md").write_text(
            f"""# Curriculum

## {PART}

| Slug | Title | Exercise delivers | Needs GPU |
|---|---|---|---|
| `001-alpha` | Alpha | Do alpha. | No |
| `002-beta` | Beta | Do beta. | No |
""",
            encoding="utf-8",
        )
        os.environ["DATA_DIR"] = str(self.root / "data")
        os.environ["CONTENT_DIR"] = str(self.root / "content")
        os.environ["GIT_SHA"] = "unknown"
        get_settings.cache_clear()
        curriculum.reload_content()

    def tearDown(self) -> None:
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()
        curriculum.reload_content()
        self.tmp.cleanup()

    def _shift(self) -> None:
        tool.insert_lesson(self.root, after="001-alpha", slug="hold")
        get_settings.cache_clear()
        curriculum.reload_content()
        db.init()

    def test_progress_on_the_pre_shift_slug_moves(self) -> None:
        self._shift()
        db.set_progress("ada", "002-beta", "solved")
        carried = db.carry_progress(curriculum.renames())
        self.assertEqual(carried, 1)
        progress = db.get_progress("ada")
        self.assertEqual(set(progress), {"003-beta"})
        self.assertEqual(progress["003-beta"]["status"], "solved")

    def test_progress_on_an_older_slug_follows_the_updated_value(self) -> None:
        self._shift()
        self.assertEqual(curriculum.renames()["legacy-beta"], ["003-beta"])
        db.set_progress("ada", "legacy-beta", "started")
        carried = db.carry_progress(curriculum.renames())
        self.assertEqual(carried, 1)
        progress = db.get_progress("ada")
        self.assertEqual(set(progress), {"003-beta"})
        self.assertEqual(progress["003-beta"]["status"], "started")


if __name__ == "__main__":
    unittest.main()
