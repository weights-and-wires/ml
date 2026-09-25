import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { Link } from "react-router-dom";

import type { Lesson } from "../lib/api";
import { KIND_LABEL, sections as readSections, symbols as readSymbols, terms as readTerms } from "../lib/lessonOutline";
import { partAccent } from "../lib/parts";
import Markdown from "./Markdown";

type Props = {
  lesson: Lesson;
  /** Opens the exercise. Absent when the lesson has none. */
  onStartExercise?: () => void;
};

const FOCUS_KEY = "ml.reader.focus";
/**
 * A section is the one being read once its top passes this line: a third of
 * the way down the pane, where the eye rests, and never above the sticky bar.
 */
function readLine(pane: HTMLElement): number {
  return Math.max(120, pane.clientHeight * 0.3);
}

function rememberedFocus(): boolean {
  try {
    return window.localStorage.getItem(FOCUS_KEY) === "1";
  } catch {
    return false;
  }
}

/** "02-linear-regression" → "Linear regression". */
function pretty(slug: string): string {
  const words = slug.replace(/^\d+-/, "").replace(/-/g, " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

export default function LessonReader({ lesson, onStartExercise }: Props) {
  const scroller = useRef<HTMLElement>(null);
  const [progress, setProgress] = useState(0);
  const [current, setCurrent] = useState(-1);
  const [read, setRead] = useState<Set<number>>(new Set());
  const [focus, setFocus] = useState(rememberedFocus);
  const [menuOpen, setMenuOpen] = useState(false);

  const outline = useMemo(() => readSections(lesson.body), [lesson.body]);
  const terms = useMemo(() => readTerms(lesson.body).slice(0, 8), [lesson.body]);
  const starter = lesson.exercise_detail?.starter ?? "";
  const symbols = useMemo(() => readSymbols(starter), [starter]);
  const accent = partAccent(lesson.part);

  // Work out which section is on screen, how far through the lesson the reader
  // is, and which sections they have scrolled past. The section classes are
  // set on the DOM directly, so the long Markdown body never rerenders.
  const measure = useCallback(() => {
    const pane = scroller.current;
    if (!pane) return;
    const max = pane.scrollHeight - pane.clientHeight;
    setProgress(max > 0 ? Math.min(1, pane.scrollTop / max) : 1);

    const top = pane.getBoundingClientRect().top + readLine(pane);
    const nodes = Array.from(pane.querySelectorAll<HTMLElement>("section[data-section]"));
    let now = -1;
    const passed = new Set<number>();
    nodes.forEach((node, index) => {
      const box = node.getBoundingClientRect();
      if (box.top <= top) now = index;
      if (box.bottom <= top || (max > 0 && pane.scrollTop >= max - 4)) passed.add(index);
    });
    nodes.forEach((node, index) => {
      node.classList.toggle("is-current", index === now);
      node.classList.toggle("is-read", passed.has(index));
    });
    // Above the first section, the introduction is what is being read.
    pane.querySelector(".lesson-intro")?.classList.toggle("is-current", now === -1);
    setCurrent(now);
    setRead((previous) =>
      previous.size === passed.size && [...passed].every((i) => previous.has(i))
        ? previous
        : passed,
    );
  }, []);

  useEffect(() => {
    const pane = scroller.current;
    if (!pane) return;
    pane.scrollTop = 0;
    let frame = 0;
    const onScroll = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(measure);
    };
    pane.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    // KaTeX and fonts settle the layout a moment after the first paint.
    const settle = window.setTimeout(measure, 250);
    measure();
    return () => {
      cancelAnimationFrame(frame);
      window.clearTimeout(settle);
      pane.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
    };
  }, [lesson.slug, measure]);

  useEffect(() => {
    try {
      window.localStorage.setItem(FOCUS_KEY, focus ? "1" : "0");
    } catch {
      // Without storage the reader picks focus mode again next time.
    }
  }, [focus]);

  // F toggles focus mode, when the keyboard is not busy in a field or the editor.
  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      const target = event.target as HTMLElement | null;
      if (
        target &&
        (target.isContentEditable ||
          ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName) ||
          target.closest(".monaco-editor"))
      ) {
        return;
      }
      if (event.key === "f" || event.key === "F") setFocus((v) => !v);
      if (event.key === "Escape") setMenuOpen(false);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  function jump(id: string) {
    setMenuOpen(false);
    const target = scroller.current?.querySelector<HTMLElement>(`#${CSS.escape(id)}`);
    target?.scrollIntoView({ behavior: "smooth", block: "start" });
    if (target?.classList.contains("term")) {
      target.classList.remove("flash");
      void target.offsetWidth;
      target.classList.add("flash");
    }
  }

  const here = current >= 0 ? outline[current] : null;
  const percent = Math.round(progress * 100);

  return (
    <article
      ref={scroller}
      className="reader min-h-0 flex-1 overflow-y-auto overscroll-contain"
      style={{ "--accent": accent } as CSSProperties}
      data-focus={focus ? "" : undefined}
    >
      <div className="reader-bar">
        <div className="reader-bar-inner mx-auto flex max-w-[72ch] items-center gap-2 px-4 lg:px-7">
          <div className="relative min-w-0 flex-1">
            <button
              type="button"
              onClick={() => setMenuOpen((v) => !v)}
              aria-expanded={menuOpen}
              aria-haspopup="true"
              className="flex min-h-9 w-full min-w-0 items-center gap-2 rounded-md px-1.5 text-left transition hover:bg-ink-850"
            >
              <span className="reader-bar-num">
                {here ? String(current + 1).padStart(2, "0") : "··"}
              </span>
              <span className="min-w-0 flex-1 truncate text-[12.5px] text-ink-200">
                {here ? here.title : "Introduction"}
              </span>
              <Chevron open={menuOpen} />
            </button>

            {menuOpen && (
              <div className="reader-menu appear" role="menu">
                <p className="px-2.5 pb-1.5 pt-1 text-[10px] font-medium uppercase tracking-[0.14em] text-ink-500">
                  On this page · {read.size} of {outline.length} read
                </p>
                <ul>
                  {outline.map((section, index) => (
                    <li key={section.id}>
                      <button
                        type="button"
                        role="menuitem"
                        onClick={() => jump(section.id)}
                        className={`reader-menu-item ${index === current ? "is-current" : ""}`}
                      >
                        <span className={`reader-menu-dot ${read.has(index) ? "is-read" : ""}`}>
                          {read.has(index) ? <Tick /> : String(index + 1)}
                        </span>
                        <span className="min-w-0 flex-1 truncate">{section.title}</span>
                        <span className="shrink-0 text-[10px] uppercase tracking-[0.1em] text-ink-600">
                          {KIND_LABEL[section.kind]}
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          <span className="hidden font-mono text-[10.5px] tabular-nums text-ink-500 sm:inline">
            {percent}%
          </span>

          <button
            type="button"
            onClick={() => setFocus((v) => !v)}
            aria-pressed={focus}
            title="Dim everything but the section you are reading (F)"
            className={`flex min-h-9 items-center gap-1.5 rounded-md px-2.5 text-[12px] transition ${
              focus
                ? "bg-[color-mix(in_oklab,var(--accent)_16%,transparent)] text-ink-100 ring-1 ring-[color-mix(in_oklab,var(--accent)_40%,transparent)]"
                : "text-ink-400 hover:bg-ink-850 hover:text-ink-200"
            }`}
          >
            <FocusIcon />
            Focus
          </button>
        </div>
        <div className="reader-progress" aria-hidden>
          <span style={{ transform: `scaleX(${progress})` }} />
        </div>
      </div>

      <div className="mx-auto max-w-[72ch] px-6 pb-10 pt-6 lg:px-9">
        <div className="mb-5 flex flex-wrap items-center gap-2 text-[11px] text-ink-400">
          <span className="part-chip">{lesson.part}</span>
          <span className="font-mono">
            Lesson {lesson.index} of {lesson.count}
          </span>
          {lesson.minutes ? (
            <>
              <span className="text-ink-700">·</span>
              <span>{lesson.minutes} min</span>
            </>
          ) : null}
          <span className="text-ink-700">·</span>
          <span>{outline.length} sections</span>
        </div>

        <h1 className="reader-title mb-6">{lesson.title}</h1>

        {lesson.objectives.length > 0 && (
          <section className="goals mb-8">
            <h2 className="goals-title">By the end you can</h2>
            <ol className="space-y-2">
              {lesson.objectives.map((objective, index) => (
                <li key={index} className="flex gap-3 text-[13.5px] leading-snug text-ink-200">
                  <span className="goals-num">{index + 1}</span>
                  <span className="pt-px">{objective}</span>
                </li>
              ))}
            </ol>
            {lesson.prereqs.length > 0 && (
              <p className="mt-3.5 border-t border-ink-800 pt-3 text-[12px] text-ink-400">
                Builds on{" "}
                {lesson.prereqs.map((prereq, index) => (
                  <span key={prereq}>
                    {index > 0 && ", "}
                    <Link
                      to={`/lessons/${prereq}`}
                      className="text-[var(--accent)] underline underline-offset-4"
                    >
                      {pretty(prereq).toLowerCase()}
                    </Link>
                  </span>
                ))}
                .
              </p>
            )}
          </section>
        )}

        <Markdown symbols={symbols}>{lesson.body}</Markdown>

        {terms.length >= 3 && <Flashcards terms={terms} onFind={jump} />}

        {onStartExercise && lesson.exercise_detail && (
          <section className="cta">
            <div className="min-w-0 flex-1">
              <p className="cta-kicker">Your turn</p>
              <p className="cta-title">{lesson.exercise_detail.title}</p>
              <p className="cta-brief">{lesson.exercise_detail.brief}</p>
            </div>
            <button type="button" onClick={onStartExercise} className="cta-button">
              Start coding
              <span className="cta-arrow" aria-hidden>
                →
              </span>
            </button>
          </section>
        )}

        <nav className="mt-10 flex items-stretch gap-3 border-t border-ink-800 pt-5">
          {lesson.prev ? (
            <Link
              to={`/lessons/${lesson.prev}`}
              className="flex-1 rounded-lg border border-ink-800 px-3.5 py-2.5 transition hover:border-ink-700 hover:bg-ink-900"
            >
              <span className="block text-[10px] uppercase tracking-[0.14em] text-ink-600">
                ← Previous
              </span>
              <span className="mt-0.5 block truncate text-[13px] text-ink-200">
                {pretty(lesson.prev)}
              </span>
            </Link>
          ) : (
            <span className="flex-1" />
          )}
          {lesson.next ? (
            <Link
              to={`/lessons/${lesson.next}`}
              className="flex-1 rounded-lg border border-ink-800 px-3.5 py-2.5 text-right transition hover:border-ink-700 hover:bg-ink-900"
            >
              <span className="block text-[10px] uppercase tracking-[0.14em] text-ink-600">
                Next →
              </span>
              <span className="mt-0.5 block truncate text-[13px] text-ink-200">
                {pretty(lesson.next)}
              </span>
            </Link>
          ) : (
            <span className="flex-1" />
          )}
        </nav>
      </div>
    </article>
  );
}

/**
 * The lesson's defined terms as cards: the term on the front, the sentence
 * that defines it on the back. Recalling before flipping is what makes a term
 * stick, which rereading does not.
 */
function Flashcards({
  terms,
  onFind,
}: {
  terms: { id: string; term: string; sentence: string }[];
  onFind: (id: string) => void;
}) {
  const [flipped, setFlipped] = useState<Set<string>>(new Set());

  function flip(id: string) {
    setFlipped((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  return (
    <section className="cards" aria-label="Test yourself">
      <div className="mb-3">
        <h2 className="cards-title">Test yourself</h2>
        <p className="mt-0.5 text-[12.5px] text-ink-400">
          Say what each term means out loud, then flip the card to check.{" "}
          <span className="font-mono text-[11px] text-ink-500">
            {flipped.size}/{terms.length}
          </span>
        </p>
      </div>
      <ul className="grid gap-2.5 sm:grid-cols-2">
        {terms.map((term) => {
          const open = flipped.has(term.id);
          return (
            <li key={term.id} className={`card ${open ? "is-flipped" : ""}`}>
              <button
                type="button"
                className="card-face"
                aria-pressed={open}
                onClick={() => flip(term.id)}
              >
                {open ? (
                  <span className="card-back">
                    <Markdown inline>{term.sentence}</Markdown>
                  </span>
                ) : (
                  <span className="card-front">
                    <span className="card-term">{term.term}</span>
                    <span className="card-hint">Tap to reveal</span>
                  </span>
                )}
              </button>
              {open && (
                <button type="button" className="card-find" onClick={() => onFind(term.id)}>
                  Find it in the lesson
                </button>
              )}
            </li>
          );
        })}
      </ul>
    </section>
  );
}

function Chevron({ open }: { open: boolean }) {
  return (
    <svg
      width="12"
      height="12"
      viewBox="0 0 12 12"
      aria-hidden
      className={`shrink-0 text-ink-500 transition ${open ? "rotate-180" : ""}`}
    >
      <path d="M3 4.5 6 7.5 9 4.5" stroke="currentColor" strokeWidth="1.4" fill="none" strokeLinecap="round" />
    </svg>
  );
}

function Tick() {
  return (
    <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden>
      <path d="M2 5.2 4.1 7.3 8 3" stroke="currentColor" strokeWidth="1.6" fill="none" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function FocusIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden>
      <path d="M1.5 4.5v-3h3M9.5 1.5h3v3M12.5 9.5v3h-3M4.5 12.5h-3v-3" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
      <circle cx="7" cy="7" r="1.8" fill="currentColor" />
    </svg>
  );
}
