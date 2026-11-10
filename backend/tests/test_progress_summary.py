"""The hub's progress summary and its path-scoped CORS allowance."""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from backend.app import curriculum, db  # noqa: E402
from backend.app.config import get_settings  # noqa: E402
from backend.app.main import app  # noqa: E402

HUB = "https://courses.ifkash.dev"
SECRET = "test-proxy-secret"
SIGNED_IN = {"x-sso-proxy": SECRET, "remote-user": "ada"}


def _lesson(slug: str, draft: bool) -> str:
    marker = "draft: true\n" if draft else ""
    return f"""---
title: "Title of {slug}"
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
"""


class ProgressSummaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        keys = ("DATA_DIR", "CONTENT_DIR", "GIT_SHA", "SSO_PROXY_SECRET", "SSO_USERS")
        self._saved = {key: os.environ.get(key) for key in keys}
        lessons = root / "content" / "lessons"
        lessons.mkdir(parents=True)
        for slug, draft in (
            ("001-alpha", False),
            ("002-beta", False),
            ("003-gamma", False),
            ("004-hold", True),
        ):
            (lessons / f"{slug}.md").write_text(_lesson(slug, draft), encoding="utf-8")
        os.environ["DATA_DIR"] = str(root / "data")
        os.environ["CONTENT_DIR"] = str(root / "content")
        # A deployed image: the draft lesson is not served.
        os.environ["GIT_SHA"] = "0123456789abcdef"
        os.environ["SSO_PROXY_SECRET"] = SECRET
        os.environ["SSO_USERS"] = ""
        get_settings.cache_clear()
        curriculum.reload_content()
        db.init()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()
        curriculum.reload_content()
        self.tmp.cleanup()

    def _stamp(self, slug: str, when: float, user: str = "ada") -> None:
        with sqlite3.connect(get_settings().db_path) as connection:
            connection.execute(
                "UPDATE progress SET updated_at = ? WHERE username = ? AND slug = ?",
                (when, user, slug),
            )

    def test_counts_solved_lessons_and_names_the_latest(self) -> None:
        db.set_progress("ada", "001-alpha", "solved")
        db.set_progress("ada", "002-beta", "started")
        db.set_progress("ada", "gone-slug", "solved")  # not in the list
        db.set_progress("ada", "004-hold", "solved")  # a draft, not served
        db.set_progress("grace", "003-gamma", "solved")  # another reader
        self._stamp("001-alpha", 100.0)
        self._stamp("002-beta", 200.0)
        self._stamp("gone-slug", 900.0)
        self._stamp("004-hold", 800.0)
        response = self.client.get(SUMMARY_PATH, headers=SIGNED_IN)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"done": 1, "total": 3, "last": {"slug": "002-beta", "title": "Title of 002-beta"}},
        )

    def test_no_progress_gives_null_last(self) -> None:
        response = self.client.get(SUMMARY_PATH, headers=SIGNED_IN)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"done": 0, "total": 3, "last": None})

    def test_allowed_origin_gets_the_headers_when_signed_in(self) -> None:
        response = self.client.get(SUMMARY_PATH, headers={**SIGNED_IN, "origin": HUB})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], HUB)
        self.assertEqual(response.headers["access-control-allow-credentials"], "true")
        self.assertIn("Origin", response.headers["vary"])

    def test_signed_out_is_401_with_the_headers(self) -> None:
        response = self.client.get(SUMMARY_PATH, headers={"origin": HUB})
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers["access-control-allow-origin"], HUB)
        self.assertEqual(response.headers["access-control-allow-credentials"], "true")
        self.assertIn("Origin", response.headers["vary"])

    def test_other_origin_gets_no_allow_origin(self) -> None:
        for origin in ("https://evil.example", "http://courses.ifkash.dev", HUB + ".evil.example"):
            response = self.client.get(SUMMARY_PATH, headers={**SIGNED_IN, "origin": origin})
            self.assertEqual(response.status_code, 200)
            self.assertNotIn("access-control-allow-origin", response.headers)
            self.assertNotIn("access-control-allow-credentials", response.headers)
            self.assertIn("Origin", response.headers["vary"])

    def test_no_origin_still_varies(self) -> None:
        response = self.client.get(SUMMARY_PATH, headers=SIGNED_IN)
        self.assertNotIn("access-control-allow-origin", response.headers)
        self.assertIn("Origin", response.headers["vary"])

    def test_other_paths_get_no_headers(self) -> None:
        response = self.client.get("/api/health", headers={"origin": HUB})
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("access-control-allow-origin", response.headers)
        self.assertNotIn("access-control-allow-credentials", response.headers)
        self.assertNotIn("origin", response.headers.get("vary", "").lower())


SUMMARY_PATH = "/api/progress/summary"

if __name__ == "__main__":
    unittest.main()
