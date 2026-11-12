"""Draft lessons are visible on a local run and hidden when GIT_SHA is set."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.app import curriculum  # noqa: E402
from backend.app.config import get_settings  # noqa: E402


def _lesson(slug: str, draft: bool) -> str:
    marker = "draft: true\n" if draft else ""
    return f"""---
title: {json.dumps(slug)}
slug: {slug}
part: "Part 1 — Foundations"
summary: "A short fixture lesson."
minutes: 10
gpu: false
objectives:
  - Read the page.
  - Name the idea.
  - Point at the next lesson.
exercise: {slug}
{marker}---

# {slug}

A fixture paragraph.

## First *section*

```python
## not a heading
```

### A `code` subsection with $x$

## Second section
"""


class DraftVisibilityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self._saved = {key: os.environ.get(key) for key in ("DATA_DIR", "CONTENT_DIR", "GIT_SHA")}
        lessons = root / "content" / "lessons"
        exercises = root / "content" / "exercises"
        lessons.mkdir(parents=True)
        exercises.mkdir(parents=True)
        for slug, draft in (("001-alpha", False), ("002-hold", True)):
            (lessons / f"{slug}.md").write_text(_lesson(slug, draft), encoding="utf-8")
            directory = exercises / slug
            directory.mkdir()
            (directory / "meta.json").write_text(
                json.dumps({"id": slug, "title": slug, "brief": "b", "needs_gpu": False, "timeout": 30}) + "\n",
                encoding="utf-8",
            )
        os.environ["DATA_DIR"] = str(root / "data")
        os.environ["CONTENT_DIR"] = str(root / "content")
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

    def _reload(self, sha: str) -> None:
        os.environ["GIT_SHA"] = sha
        get_settings.cache_clear()
        curriculum.reload_content()

    def test_local_run_shows_the_draft(self) -> None:
        self.assertFalse(get_settings().production)
        slugs = [lesson["slug"] for lesson in curriculum.load_lessons()]
        self.assertEqual(slugs, ["001-alpha", "002-hold"])
        draft = curriculum.get_lesson("002-hold")
        self.assertIsNotNone(draft)
        assert draft is not None
        self.assertTrue(draft["draft"])
        self.assertIn("002-hold", curriculum.load_exercises())

    def test_deployed_image_hides_the_draft(self) -> None:
        self._reload("0123456789abcdef")
        self.assertTrue(get_settings().production)
        slugs = [lesson["slug"] for lesson in curriculum.load_lessons()]
        self.assertEqual(slugs, ["001-alpha"])
        self.assertIsNone(curriculum.get_lesson("002-hold"))
        self.assertNotIn("002-hold", curriculum.load_exercises())
        self.assertIn("001-alpha", curriculum.load_exercises())
        published = curriculum.read_lessons(include_drafts=False)
        self.assertEqual([lesson["slug"] for lesson in published], ["001-alpha"])

    def test_lessons_list_plain_text_headings(self) -> None:
        listed = {lesson["slug"]: lesson for lesson in curriculum.lesson_list()}
        self.assertEqual(
            listed["001-alpha"]["headings"],
            [
                {"text": "First section", "level": 2},
                {"text": "A code subsection with x", "level": 3},
                {"text": "Second section", "level": 2},
            ],
        )
        self.assertNotIn("body", listed["001-alpha"])


if __name__ == "__main__":
    unittest.main()
