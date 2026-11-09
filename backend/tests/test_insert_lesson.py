"""Unit tests for scripts/insert_lesson.py."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location("insert_lesson", ROOT / "scripts" / "insert_lesson.py")
assert _SPEC is not None and _SPEC.loader is not None
tool = importlib.util.module_from_spec(_SPEC)
sys.modules["insert_lesson"] = tool
_SPEC.loader.exec_module(tool)

PART = "Part 1 — Foundations"


def _lesson(slug: str, link: str | None = None, prereq: str | None = None) -> str:
    prereq_block = f"prereqs:\n  - {prereq}\n" if prereq else ""
    target = link or slug
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
{prereq_block}exercise: {slug}
---

# {slug}

See [the other lesson](/lessons/{target}).
"""


def _write_tree(root: Path) -> None:
    lessons = root / "content" / "lessons"
    exercises = root / "content" / "exercises"
    lessons.mkdir(parents=True)
    exercises.mkdir(parents=True)
    files = {
        "001-alpha": _lesson("001-alpha", link="002-beta", prereq="002-beta"),
        "002-beta": _lesson("002-beta", link="003-gamma"),
        "003-gamma": _lesson("003-gamma"),
        "002p-project-gamma": _lesson("002p-project-gamma"),
    }
    for slug, text in files.items():
        (lessons / f"{slug}.md").write_text(text, encoding="utf-8")
        directory = exercises / slug
        directory.mkdir()
        meta = {"id": slug, "title": "Fixture", "brief": "Fixture.", "needs_gpu": False, "timeout": 30}
        (directory / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (root / "content" / "renames.json").write_text(
        json.dumps({"legacy-beta": ["002-beta"]}, indent=1) + "\n", encoding="utf-8"
    )
    (root / "docs").mkdir()
    (root / "docs" / "CURRICULUM.md").write_text(
        f"""# Curriculum

## {PART}

| Slug | Title | Exercise delivers | Needs GPU |
|---|---|---|---|
| `001-alpha` | Alpha | Do alpha. | No |
| `002-beta` | Beta | Do beta. | No |
| `003-gamma` | Gamma chapter | Do gamma chapter. | No |

## Projects

Start `002p-project-gamma` after the chapter.

| Slug | Title | Exercise delivers | Needs GPU |
|---|---|---|---|
| `002p-project-gamma` | Gamma | Do gamma. | No |
""",
        encoding="utf-8",
    )


def _snapshot(root: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            found[str(path.relative_to(root))] = path.read_text(encoding="utf-8")
    return found


class InsertLessonTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        _write_tree(self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_shift_keeps_project_mark_and_does_not_chain(self) -> None:
        tool.insert_lesson(self.root, after="001-alpha", slug="hold")
        lessons = self.root / "content" / "lessons"
        self.assertTrue((lessons / "001-alpha.md").is_file())
        self.assertFalse((lessons / "002-beta.md").is_file())
        moved = (lessons / "003-beta.md").read_text(encoding="utf-8")
        self.assertIn("slug: 003-beta", moved)
        self.assertIn("/lessons/004-gamma", moved)
        self.assertNotIn("slug: 004-beta", moved)
        earlier = (lessons / "001-alpha.md").read_text(encoding="utf-8")
        self.assertIn("slug: 001-alpha", earlier)
        self.assertIn("- 003-beta", earlier)
        self.assertIn("/lessons/003-beta", earlier)
        self.assertNotIn("002-beta", earlier)
        self.assertNotIn("004-beta", earlier)
        self.assertTrue((lessons / "004-gamma.md").is_file())
        self.assertIn("slug: 004-gamma", (lessons / "004-gamma.md").read_text(encoding="utf-8"))
        self.assertTrue((lessons / "003p-project-gamma.md").is_file())
        self.assertFalse((lessons / "002p-project-gamma.md").is_file())
        meta = json.loads((self.root / "content" / "exercises" / "003-beta" / "meta.json").read_text())
        self.assertEqual(meta["id"], "003-beta")
        self.assertEqual(
            json.loads((self.root / "content" / "exercises" / "004-gamma" / "meta.json").read_text())["id"],
            "004-gamma",
        )
        self.assertFalse((self.root / "content" / "exercises" / "002-beta").exists())

    def test_renames_follow_the_shift_and_keep_old_keys(self) -> None:
        tool.insert_lesson(self.root, after="001-alpha", slug="hold")
        renames = json.loads((self.root / "content" / "renames.json").read_text(encoding="utf-8"))
        self.assertEqual(renames["legacy-beta"], ["003-beta"])
        self.assertEqual(renames["002-beta"], ["003-beta"])
        self.assertEqual(renames["003-gamma"], ["004-gamma"])
        self.assertEqual(renames["002p-project-gamma"], ["003p-project-gamma"])
        self.assertEqual(list(renames)[0], "legacy-beta")

    def test_draft_copies_part_and_curriculum_row(self) -> None:
        tool.insert_lesson(self.root, after="001-alpha", slug="hold")
        text = (self.root / "content" / "lessons" / "002-hold.md").read_text(encoding="utf-8")
        self.assertIn("draft: true", text)
        self.assertIn(f'part: "{PART}"', text)
        self.assertIn("slug: 002-hold", text)
        self.assertIn("exercise: 002-hold", text)
        self.assertIn("- 001-alpha", text)
        curriculum = (self.root / "docs" / "CURRICULUM.md").read_text(encoding="utf-8")
        self.assertIn("| `002-hold` | Draft placeholder |", curriculum)
        self.assertIn("| `003-beta` | Beta |", curriculum)
        self.assertIn("| `004-gamma` | Gamma chapter |", curriculum)
        self.assertIn("| `003p-project-gamma` | Gamma |", curriculum)
        self.assertLess(
            curriculum.index("| `002-hold` |"),
            curriculum.index("| `003-beta` |"),
        )
        projects = curriculum.split("## Projects", 1)[1]
        self.assertNotIn("002-hold", projects)
        self.assertIn("`003p-project-gamma`", projects)

    def test_same_number_project_moves_and_draft_sorts_before_it(self) -> None:
        tool.insert_lesson(self.root, after="002-beta", slug="hold")
        lessons = self.root / "content" / "lessons"
        self.assertTrue((lessons / "002-beta.md").is_file())
        self.assertFalse((lessons / "002p-project-gamma.md").is_file())
        self.assertTrue((lessons / "003p-project-gamma.md").is_file())
        self.assertTrue((lessons / "003-hold.md").is_file())
        self.assertTrue((lessons / "004-gamma.md").is_file())
        self.assertFalse((lessons / "003-gamma.md").is_file())
        names = sorted(path.name for path in lessons.glob("*.md"))
        self.assertLess(names.index("003-hold.md"), names.index("003p-project-gamma.md"))
        moved = (lessons / "003p-project-gamma.md").read_text(encoding="utf-8")
        self.assertIn("slug: 003p-project-gamma", moved)
        curriculum = (self.root / "docs" / "CURRICULUM.md").read_text(encoding="utf-8")
        self.assertIn("`003p-project-gamma`", curriculum)
        self.assertNotIn("`002p-project-gamma`", curriculum)
        self.assertLess(
            curriculum.index("| `003-hold` |"),
            curriculum.index("| `004-gamma` |"),
        )

    def test_count_opens_distinct_stubs(self) -> None:
        tool.insert_lesson(self.root, after="001-alpha", slug="hold", count=2)
        lessons = self.root / "content" / "lessons"
        self.assertTrue((lessons / "002-hold.md").is_file())
        self.assertTrue((lessons / "003-hold-2.md").is_file())
        self.assertIn("Draft placeholder 2", (lessons / "003-hold-2.md").read_text(encoding="utf-8"))
        self.assertTrue((lessons / "004-beta.md").is_file())
        self.assertTrue((lessons / "005-gamma.md").is_file())
        self.assertIn("slug: 004-beta", (lessons / "004-beta.md").read_text(encoding="utf-8"))

    def test_dry_run_writes_nothing(self) -> None:
        before = _snapshot(self.root)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = tool.main(["--after", "001-alpha", "--slug", "hold", "--dry-run"], root=self.root)
        self.assertEqual(code, 0)
        self.assertIn("Dry run: no files were written.", stdout.getvalue())
        self.assertEqual(before, _snapshot(self.root))

    def test_replace_slugs_does_not_chain_or_cut_a_longer_id(self) -> None:
        mapping = {"002-beta": "003-beta", "003-beta": "004-beta"}
        self.assertEqual(
            tool.replace_slugs("see 002-beta and 003-beta", mapping),
            "see 003-beta and 004-beta",
        )
        self.assertEqual(
            tool.replace_slugs("015-foo-bar and 015-foo", {"015-foo": "016-foo"}),
            "015-foo-bar and 016-foo",
        )

    def test_refuses_a_rest_that_would_retire_a_live_slug(self) -> None:
        lessons = self.root / "content" / "lessons"
        exercises = self.root / "content" / "exercises"
        (lessons / "003-gamma.md").rename(lessons / "003-beta.md")
        (exercises / "003-gamma").rename(exercises / "003-beta")
        meta = exercises / "003-beta" / "meta.json"
        meta.write_text(meta.read_text(encoding="utf-8").replace("003-gamma", "003-beta"), encoding="utf-8")
        with self.assertRaises(tool.InsertError):
            tool.insert_lesson(self.root, after="001-alpha", slug="hold")

    def test_rejects_a_slug_that_would_steal_progress(self) -> None:
        with self.assertRaises(tool.InsertError):
            tool.insert_lesson(self.root, after="001-alpha", slug="beta")

    def test_rejects_unknown_after_and_bad_count(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = tool.main(["--after", "999-missing", "--slug", "hold"], root=self.root)
        self.assertEqual(code, 1)
        self.assertIn("no lesson named 999-missing", stderr.getvalue())
        with self.assertRaises(tool.InsertError):
            tool.insert_lesson(self.root, after="001-alpha", slug="hold", count=0)
        with self.assertRaises(tool.InsertError):
            tool.insert_lesson(self.root, after="001-alpha", slug="009-hold")


class DraftCheckerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        _write_tree(self.root)
        tool.insert_lesson(self.root, after="001-alpha", slug="hold")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _check(self, slug: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "check_content.py"), slug],
            cwd=ROOT,
            capture_output=True,
            text=True,
            env={**os.environ, "CONTENT_DIR": str(self.root / "content")},
            check=False,
        )

    def test_draft_passes_structure_checks(self) -> None:
        result = self._check("002-hold")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("0 problem(s)", result.stdout)

    def test_published_word_minimum_still_applies(self) -> None:
        path = self.root / "content" / "lessons" / "002-hold.md"
        path.write_text(path.read_text(encoding="utf-8").replace("draft: true\n", ""), encoding="utf-8")
        result = self._check("002-hold")
        self.assertEqual(result.returncode, 1)
        self.assertIn("the contract asks for 800 or more", result.stdout)


if __name__ == "__main__":
    unittest.main()
