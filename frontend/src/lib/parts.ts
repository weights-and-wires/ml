import type { LessonMeta } from "./api";

/** "Part 3 — Neural networks" → { number: "3", name: "Neural networks" }. */
export function splitPart(part: string): { number: string; name: string } {
  const match = /^Part\s+(\d+)\s*[—–-]\s*(.+)$/i.exec(part);
  return match ? { number: match[1], name: match[2] } : { number: "", name: part };
}

/**
 * A lesson's title from its slug. Falls back to the slug itself, tidied, for
 * the moment before the lesson list has loaded.
 */
export function titleOf(slug: string, lessons: LessonMeta[]): string {
  const found = lessons.find((lesson) => lesson.slug === slug);
  if (found) return found.title;
  const words = slug.replace(/^\d+-/, "").replace(/-/g, " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

/**
 * The number each lesson shows in lists: chapters count 1, 2, 3, and the
 * projects that close each part count P1, P2, P3 on their own.
 */
export function lessonNumbers(lessons: Pick<LessonMeta, "slug" | "kind">[]): Map<string, string> {
  const numbers = new Map<string, string>();
  let chapters = 0;
  let projects = 0;
  for (const lesson of lessons) {
    numbers.set(lesson.slug, lesson.kind === "project" ? `P${++projects}` : String(++chapters));
  }
  return numbers;
}
