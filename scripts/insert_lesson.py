#!/usr/bin/env python3
"""Insert draft lessons and shift later lessons so reader progress follows.

    python3 scripts/insert_lesson.py --after 028-softmax-and-whole-network-gradients \\
        --slug my-lesson
    python3 scripts/insert_lesson.py --after 028-softmax-and-whole-network-gradients \\
        --slug my-lesson --count 2
    python3 scripts/insert_lesson.py --after 028-softmax-and-whole-network-gradients \\
        --slug my-lesson --dry-run

The lesson named by ``--after`` stays put. Every lesson file and exercise
directory whose sort key ``(number, project, name)`` is greater than the
anchor's moves up by ``--count`` (the default is 1). A project keeps the ``p``
after its number, so a project that shares the anchor's number is later than
that anchor and moves with the later lessons. The new draft is not a project,
so it sorts before a project that lands on the same number. The opened numbers
receive draft lessons copied from a template. ``content/renames.json`` records
each move, and existing entries that still name a shifted slug are updated, so
a progress row stored under an older slug still resolves.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

_ID = re.compile(r"^(\d{3})(p)?-(.+)$")
_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

_SUMMARY = "A draft placeholder that keeps the shape of a lesson."
_DELIVERS = "Placeholder checks for a lesson that is still being written."


class InsertError(Exception):
    """The insertion cannot be applied as asked."""


@dataclass(frozen=True)
class LessonId:
    """A lesson or exercise id: three digits, an optional project mark, and a name."""

    number: int
    project: bool
    rest: str

    @classmethod
    def parse(cls, name: str) -> LessonId:
        match = _ID.fullmatch(name)
        if not match:
            raise InsertError(f"{name} is not a lesson id. Use NNN-slug or NNNp-project-slug.")
        return cls(int(match.group(1)), match.group(2) == "p", match.group(3))

    @property
    def slug(self) -> str:
        mark = "p" if self.project else ""
        return f"{self.number:03d}{mark}-{self.rest}"

    def shifted(self, count: int) -> LessonId:
        return LessonId(self.number + count, self.project, self.rest)


def _sort_key(slug: str) -> tuple[int, bool, str]:
    item = LessonId.parse(slug)
    return (item.number, item.project, item.rest)


def _validate_slug(slug: str) -> None:
    if not _NAME.fullmatch(slug) or re.match(r"^\d{3}(?:p-|-)", slug):
        raise InsertError(
            "--slug is the name only, for example ridge-regression, without a number prefix."
        )


def _stub_names(slug: str, count: int) -> list[str]:
    if count == 1:
        return [slug]
    return [slug if index == 1 else f"{slug}-{index}" for index in range(1, count + 1)]


def _stub_title(index: int, count: int) -> str:
    if count == 1:
        return "Draft placeholder"
    return f"Draft placeholder {index}"


def _lesson_ids(root: Path) -> list[LessonId]:
    directory = root / "content" / "lessons"
    if not directory.is_dir():
        raise InsertError(f"{directory} is not a directory.")
    return [LessonId.parse(path.stem) for path in sorted(directory.glob("*.md"))]


def _exercise_ids(root: Path) -> list[LessonId]:
    directory = root / "content" / "exercises"
    if not directory.is_dir():
        raise InsertError(f"{directory} is not a directory.")
    found: list[LessonId] = []
    for path in sorted(directory.iterdir()):
        if not path.is_dir() or path.name.startswith(".") or path.name == "__pycache__":
            continue
        found.append(LessonId.parse(path.name))
    return found


def _shift_map(ids: list[LessonId], anchor: LessonId, count: int) -> dict[str, str]:
    """Map every id that sorts after `anchor` to the id `count` numbers later.

    Sort order is ``(number, project, name)``. A project shares its number with
    the chapter it follows and still sorts later, because ``project`` is true.
    Comparing the number alone would leave that project in place and put the
    new chapter after it.
    """
    mapping: dict[str, str] = {}
    anchor_key = _sort_key(anchor.slug)
    for item in ids:
        if _sort_key(item.slug) > anchor_key:
            mapping[item.slug] = item.shifted(count).slug
    return mapping


def _front_matter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise InsertError(f"{path.name} has no YAML front matter.")
    end = text.find("\n---\n", 3)
    if end < 0:
        raise InsertError(f"{path.name} has no closing front matter.")
    loaded = yaml.safe_load(text[4:end])
    if not isinstance(loaded, dict):
        raise InsertError(f"{path.name} front matter is not a mapping.")
    return loaded


def replace_slugs(text: str, mapping: dict[str, str]) -> str:
    """Replace lesson ids in `text` without chaining a shift into the next one.

    A direct substitution of 015-foo with 016-foo would then be caught by the
    rule that moves 016-foo to 017-foo. Placeholders break that chain.
    """
    if not mapping:
        return text
    tokens: dict[str, str] = {}
    for index, old in enumerate(sorted(mapping, key=len, reverse=True)):
        token = f"\x00SLUG{index}\x00"
        tokens[token] = mapping[old]
        pattern = re.compile(rf"(?<![A-Za-z0-9-]){re.escape(old)}(?![A-Za-z0-9-])")
        text = pattern.sub(token, text)
    for token, new in tokens.items():
        text = text.replace(token, new)
    return text


def _merge_maps(lesson_shifts: dict[str, str], exercise_shifts: dict[str, str]) -> dict[str, str]:
    merged = dict(lesson_shifts)
    for old, new in exercise_shifts.items():
        if old in merged and merged[old] != new:
            raise InsertError(f"{old} would shift to both {merged[old]} and {new}.")
        merged[old] = new
    return merged


def _final_names(ids: list[LessonId], shifts: dict[str, str], extra: list[str]) -> list[str]:
    names = [shifts.get(item.slug, item.slug) for item in ids]
    names.extend(extra)
    if len(names) != len(set(names)):
        raise InsertError("the shift would put two items on one slug.")
    return names


def _load_renames(root: Path) -> dict[str, list[str]]:
    path = root / "content" / "renames.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise InsertError("content/renames.json must be a JSON object.")
    cleaned: dict[str, list[str]] = {}
    for key, values in data.items():
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise InsertError(f"content/renames.json entry {key} is not a list of slugs.")
        cleaned[str(key)] = list(values)
    return cleaned


def _apply_renames(data: dict[str, list[str]], mapping: dict[str, str]) -> dict[str, list[str]]:
    for old in mapping:
        if old in data:
            raise InsertError(
                f"{old} is both a current lesson and a retired slug in content/renames.json. "
                "Refusing to shift it, because progress on the two would merge."
            )
    for new in mapping.values():
        if new in data:
            raise InsertError(
                f"{new} is a retired slug in content/renames.json. "
                "Shifting a lesson onto it would move that lesson's progress away on the next launch."
            )
    updated = {key: [mapping.get(item, item) for item in values] for key, values in data.items()}
    for old in sorted(mapping, key=_sort_key):
        updated[old] = [mapping[old]]
    return updated


def _insert_curriculum_row(lines: list[str], part: str, stub: LessonId, title: str) -> None:
    heading = f"## {part}"
    try:
        start = lines.index(heading)
    except ValueError:
        raise InsertError(f"docs/CURRICULUM.md has no {heading!r} heading for {stub.slug}.") from None
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    data_rows = [i for i in range(start, end) if lines[i].startswith("| `")]
    if not data_rows:
        raise InsertError(f"docs/CURRICULUM.md has no table under {heading}.")
    row = f"| `{stub.slug}` | {title} | {_DELIVERS} | No |"
    insert_at = data_rows[-1] + 1
    for index in data_rows:
        found = re.search(r"`([^`]+)`", lines[index])
        if found is None:
            continue
        try:
            current = LessonId.parse(found.group(1))
        except InsertError:
            continue
        if (current.number, current.project) > (stub.number, False):
            insert_at = index
            break
    lines.insert(insert_at, row)


def _update_curriculum(text: str, mapping: dict[str, str], part: str, stubs: list[tuple[LessonId, str]]) -> str:
    updated = replace_slugs(text, mapping)
    ending = "\n" if updated.endswith("\n") else ""
    lines = updated.splitlines()
    for stub, title in stubs:
        _insert_curriculum_row(lines, part, stub, title)
    return "\n".join(lines) + ending


def _yaml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _render_lesson(stub: LessonId, title: str, part: str, prereq: str) -> str:
    return f"""---
title: {_yaml_string(title)}
slug: {stub.slug}
part: {_yaml_string(part)}
summary: {_yaml_string(_SUMMARY)}
minutes: 15
gpu: false
objectives:
  - State the question the chapter answers.
  - Work a two-number example by hand.
  - Implement double_score and compare the result with the checks.
prereqs:
  - {prereq}
exercise: {stub.slug}
draft: true
---

# {title}

This page holds a place in the course for a chapter that is not written. It keeps the shape of a finished lesson so the checker can read it, and the exercise is a two-line function you replace when the chapter is ready.

The opening of a finished chapter starts from a concrete situation and says why the idea matters. This draft names the place in the sequence and stops there.

## What you already know

- A lesson page states one idea and then asks you to build it.
- A function takes an array and returns an array of the same shape.
- The checks compare your result with a known array.

## A single step

A finished chapter climbs from a picture to a formula. This draft stops at the picture, so you can see the step the chapter fills in.

```figure
<svg viewBox="0 0 560 220" role="img" aria-label="Two points, a score of 1 and a score of 2">
  <circle cx="160" cy="140" r="6" class="fill-a"/>
  <circle cx="400" cy="80" r="6" class="fill-b"/>
  <text x="180" y="144">1</text>
  <text x="420" y="84">2</text>
</svg>
Figure 1. The draft marks one step, from a score of 1 to a score of 2.
```

Take the number 1. Multiplying by 2 gives 2. The exercise asks you to do that to every entry of an array, so the vector `[1, 2, 3]` becomes `[2, 4, 6]`.

## Build it step by step

1. **`double_score(scores)`** takes an array of any shape and returns an array of the same shape. Multiply every entry by 2. The mistake people most often make is to add 2 instead of multiplying.

## The exercise

You implement `double_score`. The checks compare your array with twice the input on a vector, a matrix, zero, and a negative number.
"""


def _starter_source() -> str:
    return '''"""Draft exercise. Double every entry of an array."""

from __future__ import annotations

import numpy as np


def double_score(scores: np.ndarray) -> np.ndarray:
    """Return twice each entry.

    Args:
        scores: An array of any shape.

    Returns:
        An array of the same shape, with every entry multiplied by 2.
    """
    raise NotImplementedError("write double_score")


if __name__ == "__main__":
    print(double_score(np.array([1.0, 2.0, 3.0])))
'''


def _solution_source() -> str:
    return '''"""Reference solution for the draft placeholder exercise."""

from __future__ import annotations

import numpy as np


def double_score(scores: np.ndarray) -> np.ndarray:
    """Return twice each entry.

    Args:
        scores: An array of any shape.

    Returns:
        An array of the same shape, with every entry multiplied by 2.
    """
    return np.asarray(scores, dtype=float) * 2


if __name__ == "__main__":
    print(double_score(np.array([1.0, 2.0, 3.0])))
'''


def _tests_source() -> str:
    return '''"""Checks for the draft placeholder exercise."""

import numpy as np

from mlcheck import check, close, fail, has


@check("double_score doubles a vector: [1, 2, 3] becomes [2, 4, 6]")
def test_vector(sub):
    fn = has(sub, "double_score")
    close(fn(np.array([1.0, 2.0, 3.0])), np.array([2.0, 4.0, 6.0]), atol=1e-12, label="twice [1, 2, 3]")


@check("double_score keeps the shape of a matrix")
def test_matrix(sub):
    fn = has(sub, "double_score")
    got = fn(np.array([[1.0, 2.0], [3.0, 4.0]]))
    close(got, np.array([[2.0, 4.0], [6.0, 8.0]]), atol=1e-12, label="twice a 2 by 2 matrix")
    if got.shape != (2, 2):
        fail(f"expected shape (2, 2), got {got.shape}")


@check("double_score leaves zero at zero")
def test_zero(sub):
    fn = has(sub, "double_score")
    close(fn(np.array([0.0])), np.array([0.0]), atol=1e-12, label="twice zero")


@check("double_score doubles a negative number: -1.5 becomes -3")
def test_negative(sub):
    fn = has(sub, "double_score")
    close(fn(np.array([-1.5])), np.array([-3.0]), atol=1e-12, label="twice -1.5")
'''


def _write_stub(root: Path, stub: LessonId, title: str, part: str, prereq: str) -> None:
    lesson = root / "content" / "lessons" / f"{stub.slug}.md"
    lesson.write_text(_render_lesson(stub, title, part, prereq), encoding="utf-8")
    exercise = root / "content" / "exercises" / stub.slug
    exercise.mkdir(parents=True)
    meta = {
        "id": stub.slug,
        "title": "Double every score",
        "brief": "Multiply every entry of an array by 2. This draft holds the exercise until the lesson is written.",
        "needs_gpu": False,
        "timeout": 60,
        "packages": ["numpy"],
        "hints": ["Multiply the array by 2. Adding 2 passes one check and fails the rest."],
    }
    (exercise / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (exercise / "starter.py").write_text(_starter_source(), encoding="utf-8")
    (exercise / "solution.py").write_text(_solution_source(), encoding="utf-8")
    (exercise / "tests.py").write_text(_tests_source(), encoding="utf-8")


def _rewrite_lessons(root: Path, mapping: dict[str, str], dry_run: bool) -> int:
    changed = 0
    for path in sorted((root / "content" / "lessons").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        updated = replace_slugs(text, mapping)
        if updated == text:
            continue
        changed += 1
        if not dry_run:
            path.write_text(updated, encoding="utf-8")
    return changed


def _rewrite_meta_ids(root: Path, shifts: dict[str, str], dry_run: bool) -> None:
    directory = root / "content" / "exercises"
    for old, new in shifts.items():
        path = directory / old / "meta.json"
        if not path.is_file():
            raise InsertError(f"{old} has no meta.json.")
        text = path.read_text(encoding="utf-8")

        def replace(match: re.Match[str], new: str = new) -> str:
            return f"{match.group(1)}{new}{match.group(2)}"

        pattern = re.compile(rf'("id"\s*:\s*"){re.escape(old)}(")')
        updated, count = pattern.subn(replace, text, count=1)
        if count != 1:
            raise InsertError(f"{path} has no id {old}.")
        if not dry_run:
            path.write_text(updated, encoding="utf-8")


def _rename_moved(pairs: list[tuple[Path, Path]]) -> None:
    staged: list[tuple[Path, Path]] = []
    for src, dst in pairs:
        if not src.exists():
            raise InsertError(f"{src} is missing.")
        temporary = src.with_name(src.name + ".__shifting__")
        if temporary.exists():
            raise InsertError(f"{temporary} already exists.")
        src.rename(temporary)
        staged.append((temporary, dst))
    for temporary, dst in staged:
        if dst.exists():
            raise InsertError(f"refusing to overwrite {dst}.")
        temporary.rename(dst)


def _format_plan(
    anchor: LessonId,
    count: int,
    part: str,
    stubs: list[tuple[LessonId, str]],
    lesson_shifts: dict[str, str],
    exercise_shifts: dict[str, str],
    dry_run: bool,
) -> str:
    lines = [
        f"After {anchor.slug}, open {count} number{'s' if count != 1 else ''}.",
        "Draft lessons:",
    ]
    for stub, title in stubs:
        lines.append(f"  {stub.slug}  {title}  ({part})")
    lines.append(
        f"Shift {len(lesson_shifts)} lesson{'s' if len(lesson_shifts) != 1 else ''} "
        f"and {len(exercise_shifts)} exercise{'s' if len(exercise_shifts) != 1 else ''}."
    )
    for old in sorted(lesson_shifts, key=_sort_key):
        lines.append(f"  {old} -> {lesson_shifts[old]}")
    if exercise_shifts != lesson_shifts:
        lines.append("Exercise ids that differ from the lesson ids:")
        for old in sorted(exercise_shifts, key=_sort_key):
            if lesson_shifts.get(old) != exercise_shifts[old]:
                lines.append(f"  {old} -> {exercise_shifts[old]}")
    if lesson_shifts:
        lines.append(
            f"Rename map: append {len(lesson_shifts)} "
            f"entr{'ies' if len(lesson_shifts) != 1 else 'y'}, and update existing "
            "values that name a shifted slug."
        )
    else:
        lines.append("Rename map: nothing shifts, so content/renames.json stays as it is.")
    lines.append(f"Curriculum: update shifted slugs and insert {len(stubs)} draft row{'s' if len(stubs) != 1 else ''}.")
    if dry_run:
        lines.append("Dry run: no files were written.")
    else:
        lines.append(
            f"Wrote {len(stubs)} draft lesson{'s' if len(stubs) != 1 else ''} "
            f"and shifted {len(lesson_shifts)} lesson{'s' if len(lesson_shifts) != 1 else ''} "
            f"and {len(exercise_shifts)} exercise{'s' if len(exercise_shifts) != 1 else ''}."
        )
    return "\n".join(lines)


def insert_lesson(root: Path, after: str, slug: str, count: int = 1, dry_run: bool = False) -> str:
    """Shift later lessons and write draft stubs. Return the plan text.

    ``dry_run`` checks the tree and prints the plan, and it does not write.
    """
    if count < 1:
        raise InsertError("--count must be at least 1.")
    _validate_slug(slug)
    root = root.resolve()
    lessons = _lesson_ids(root)
    exercises = _exercise_ids(root)
    matches = [item for item in lessons if item.slug == after]
    if not matches:
        raise InsertError(f"no lesson named {after}.")
    anchor = matches[0]
    meta = _front_matter(root / "content" / "lessons" / f"{anchor.slug}.md")
    part = meta.get("part")
    if not isinstance(part, str) or not part.strip():
        raise InsertError(f"{anchor.slug} has no part string to copy onto the draft.")

    lesson_shifts = _shift_map(lessons, anchor, count)
    exercise_shifts = _shift_map(exercises, anchor, count)
    mapping = _merge_maps(lesson_shifts, exercise_shifts)
    stubs = [
        LessonId(anchor.number + offset, False, name)
        for offset, name in enumerate(_stub_names(slug, count), start=1)
    ]
    titled = [(stub, _stub_title(index, count)) for index, stub in enumerate(stubs, start=1)]

    renames = _load_renames(root)
    occupied = {item.slug for item in lessons} | {item.slug for item in exercises} | set(renames)
    for stub, _title in titled:
        if stub.slug in occupied or stub.slug in mapping.values():
            raise InsertError(
                f"the draft {stub.slug} reuses a slug that already identifies a lesson "
                "or a retired progress row. Pick a different --slug so reader progress "
                "stays on the lesson that moved."
            )
    live = _final_names(lessons, lesson_shifts, [stub.slug for stub, _title in titled])
    _final_names(exercises, exercise_shifts, [stub.slug for stub, _title in titled])
    updated_renames = _apply_renames(renames, lesson_shifts) if lesson_shifts else renames
    overlap = set(updated_renames) & set(live)
    if overlap:
        shown = ", ".join(sorted(overlap)[:3])
        raise InsertError(
            f"retired slug {shown} would also be a live lesson. "
            "Reader progress on that lesson would move again on the next launch."
        )

    curriculum_path = root / "docs" / "CURRICULUM.md"
    if not curriculum_path.is_file():
        raise InsertError("docs/CURRICULUM.md is missing.")
    curriculum = _update_curriculum(
        curriculum_path.read_text(encoding="utf-8"), mapping, part, titled
    )

    # Validate meta ids before any rename, including on a dry run.
    _rewrite_lessons(root, mapping, dry_run)
    _rewrite_meta_ids(root, exercise_shifts, dry_run)

    if not dry_run:
        _rename_moved(
            [
                (
                    root / "content" / "lessons" / f"{old}.md",
                    root / "content" / "lessons" / f"{new}.md",
                )
                for old, new in lesson_shifts.items()
            ]
        )
        _rename_moved(
            [
                (
                    root / "content" / "exercises" / old,
                    root / "content" / "exercises" / new,
                )
                for old, new in exercise_shifts.items()
            ]
        )
        if lesson_shifts:
            (root / "content" / "renames.json").write_text(
                json.dumps(updated_renames, indent=1) + "\n", encoding="utf-8"
            )
        curriculum_path.write_text(curriculum, encoding="utf-8")
        for stub, title in titled:
            _write_stub(root, stub, title, part, anchor.slug)

    return _format_plan(anchor, count, part, titled, lesson_shifts, exercise_shifts, dry_run)


def main(argv: list[str] | None = None, root: Path | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Insert draft lessons and shift later lessons so reader progress follows the move."
        )
    )
    parser.add_argument("--after", required=True, metavar="NNN-slug", help="Slug of the lesson that stays put.")
    parser.add_argument(
        "--slug",
        required=True,
        metavar="new-slug-name",
        help="Name of the draft, without the number prefix.",
    )
    parser.add_argument("--count", type=int, default=1, metavar="K", help="How many numbers to open. The default is 1.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and write nothing.")
    args = parser.parse_args(argv)
    try:
        print(insert_lesson(root or ROOT, args.after, args.slug, args.count, args.dry_run))
    except InsertError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
