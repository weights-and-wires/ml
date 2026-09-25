import { useEffect, useMemo, useState } from "react";
import { NavLink, useParams } from "react-router-dom";

import type { Part, ProgressMap } from "../lib/api";
import { partAccent } from "../lib/parts";

type Props = {
  parts: Part[];
  progress: ProgressMap;
  /** Whether the narrow-screen drawer is showing. */
  open: boolean;
  /** Whether the wide-screen reader has hidden the list. */
  collapsed: boolean;
  onClose: () => void;
  onCollapse: () => void;
};

const OPEN_PARTS_KEY = "ml.sidebar.parts";

export default function Sidebar({
  parts,
  progress,
  open,
  collapsed,
  onClose,
  onCollapse,
}: Props) {
  const { slug } = useParams();
  const [shutParts, setShutParts] = useState<Set<string>>(() => {
    try {
      const raw = window.localStorage.getItem(OPEN_PARTS_KEY);
      return new Set<string>(raw ? JSON.parse(raw) : []);
    } catch {
      return new Set<string>();
    }
  });
  const [filter, setFilter] = useState("");

  useEffect(() => {
    try {
      window.localStorage.setItem(
        OPEN_PARTS_KEY,
        JSON.stringify([...shutParts]),
      );
    } catch {
      // Storage being off only costs the reader their collapse state.
    }
  }, [shutParts]);

  const query = filter.trim().toLowerCase();
  const shown = useMemo(() => {
    if (!query) return parts;
    return parts
      .map((part) => ({
        ...part,
        lessons: part.lessons.filter(
          (lesson) =>
            lesson.title.toLowerCase().includes(query) ||
            lesson.summary.toLowerCase().includes(query) ||
            lesson.slug.includes(query),
        ),
      }))
      .filter((part) => part.lessons.length > 0);
  }, [parts, query]);

  const totals = useMemo(() => {
    const all = parts.flatMap((part) => part.lessons);
    const solved = all.filter((l) => progress[l.slug]?.status === "solved").length;
    return { all: all.length, solved };
  }, [parts, progress]);

  function toggle(part: string) {
    setShutParts((current) => {
      const next = new Set(current);
      if (next.has(part)) next.delete(part);
      else next.add(part);
      return next;
    });
  }

  // A hidden pane keeps its 18rem of content and loses its width, so the list
  // is clipped on the way out instead of reflowing into a narrower column.
  // visibility:hidden also takes the links out of the tab order.
  const drawer = open
    ? "visible translate-x-0 shadow-2xl shadow-ink-950/60"
    : "invisible -translate-x-full";
  const wide = collapsed
    ? "lg:invisible lg:w-0 lg:translate-x-0 lg:border-r-0"
    : "lg:visible lg:w-72 lg:translate-x-0";

  return (
    <>
      {open && (
        <button
          type="button"
          aria-label="Close the lesson list"
          onClick={onClose}
          className="fixed inset-0 z-40 bg-ink-950/70 backdrop-blur-[2px] lg:hidden"
        />
      )}

      <nav
        aria-label="Lessons"
        className={`fixed inset-y-0 left-0 z-50 flex w-72 shrink-0 flex-col overflow-hidden border-r border-ink-800 bg-ink-900 transition-[transform,width,visibility] duration-200 ease-out lg:static ${drawer} ${wide}`}
      >
        <div className="flex w-72 min-h-0 flex-1 flex-col">
          <div className="border-b border-ink-800 px-4 py-3.5">
            <div className="mb-3 flex items-center justify-between gap-2">
              <span className="text-[11px] font-medium uppercase tracking-[0.16em] text-ink-400">
                Curriculum
              </span>
              <span className="flex items-center gap-1.5">
                <span className="font-mono text-[11px] tabular-nums text-ink-400">
                  {totals.solved}/{totals.all}
                </span>
                <button
                  type="button"
                  onClick={onCollapse}
                  aria-label="Hide the lesson list"
                  title="Hide the lesson list (Cmd or Ctrl with B)"
                  className="-mr-1 hidden rounded p-1 text-ink-600 transition hover:bg-ink-850 hover:text-ink-300 lg:block"
                >
                  <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden>
                    <path
                      d="M7.5 2 3.5 6l4 4"
                      stroke="currentColor"
                      strokeWidth="1.5"
                      fill="none"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                </button>
                {/* The drawer covers the header, and with it the button that
                    opened the drawer, so it carries its own way out. */}
                <button
                  type="button"
                  onClick={onClose}
                  aria-label="Close the lesson list"
                  className="-mr-2 flex size-9 items-center justify-center rounded-md text-ink-400 transition hover:bg-ink-850 hover:text-ink-200 lg:hidden"
                >
                  <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden>
                    <path
                      d="M3.5 3.5l7 7M10.5 3.5l-7 7"
                      stroke="currentColor"
                      strokeWidth="1.5"
                      strokeLinecap="round"
                    />
                  </svg>
                </button>
              </span>
            </div>
            <div className="h-1 overflow-hidden rounded-full bg-ink-800">
              <div
                className="h-full rounded-full bg-gradient-to-r from-good-500/70 to-good-500 transition-[width] duration-500"
                style={{
                  width: `${totals.all ? (totals.solved / totals.all) * 100 : 0}%`,
                }}
              />
            </div>

            <div className="relative mt-3">
              <svg
                width="12"
                height="12"
                viewBox="0 0 12 12"
                aria-hidden
                className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-ink-600"
              >
                <circle
                  cx="5"
                  cy="5"
                  r="3.4"
                  stroke="currentColor"
                  strokeWidth="1.3"
                  fill="none"
                />
                <path
                  d="M7.6 7.6 10.5 10.5"
                  stroke="currentColor"
                  strokeWidth="1.3"
                  strokeLinecap="round"
                />
              </svg>
              <input
                type="search"
                value={filter}
                onChange={(event) => setFilter(event.target.value)}
                placeholder="Filter lessons"
                aria-label="Filter lessons"
                className="w-full rounded-lg border border-ink-800 bg-ink-850 py-2 pl-7 pr-2.5 text-base text-ink-200 transition placeholder:text-ink-600 hover:border-ink-700 focus:border-wire-500/50 focus:bg-ink-900 focus:outline-none lg:py-1.5 lg:text-xs"
              />
            </div>
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-2 py-2">
            {shown.map((part) => {
              const shut = shutParts.has(part.part) && !query;
              const done = part.lessons.filter(
                (l) => progress[l.slug]?.status === "solved",
              ).length;
              const complete = part.lessons.length > 0 && done === part.lessons.length;
              return (
                <section key={part.part} className="mb-1">
                  <button
                    type="button"
                    onClick={() => toggle(part.part)}
                    aria-expanded={!shut}
                    className="flex w-full items-center gap-1.5 rounded-md px-2 py-2.5 text-left text-[11px] font-medium uppercase tracking-[0.1em] text-ink-400 transition hover:bg-ink-850 hover:text-ink-300 lg:py-1.5"
                  >
                    <svg
                      width="10"
                      height="10"
                      viewBox="0 0 10 10"
                      className={`shrink-0 transition-transform duration-200 ${shut ? "" : "rotate-90"}`}
                      aria-hidden
                    >
                      <path d="M3 1.5 7 5 3 8.5" stroke="currentColor" strokeWidth="1.4" fill="none" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                    <span
                      className="size-1.5 shrink-0 rounded-full"
                      style={{ background: partAccent(part.part) }}
                      aria-hidden
                    />
                    <span className="flex-1 truncate">{part.part}</span>
                    <span
                      className={`font-mono text-[10px] tabular-nums ${complete ? "text-good-500" : "text-ink-600"}`}
                    >
                      {done}/{part.lessons.length}
                    </span>
                  </button>

                  {!shut && (
                    <ul className="mt-0.5 space-y-px">
                      {part.lessons.map((lesson) => {
                        const state = progress[lesson.slug]?.status;
                        return (
                          <li key={lesson.slug}>
                            <NavLink
                              to={`/lessons/${lesson.slug}`}
                              onClick={onClose}
                              className={({ isActive }) =>
                                `group relative flex items-start gap-2 rounded-md py-2.5 pl-3 pr-2 text-[13px] leading-snug transition lg:py-1.5 ${
                                  isActive || slug === lesson.slug
                                    ? "bg-wire-500/12 text-ink-100 before:absolute before:inset-y-1 before:left-0 before:w-0.5 before:rounded-full before:bg-wire-500"
                                    : "text-ink-300 hover:bg-ink-850 hover:text-ink-200"
                                }`
                              }
                            >
                              <StatusDot state={state} />
                              <span className="min-w-0 flex-1">
                                <span className="block truncate">{lesson.title}</span>
                              </span>
                              {lesson.gpu && (
                                <span
                                  title="This lesson's exercise runs on a GPU"
                                  className="mt-0.5 shrink-0 rounded border border-warm-500/30 bg-warm-500/10 px-1 font-mono text-[9px] uppercase text-warm-500"
                                >
                                  gpu
                                </span>
                              )}
                            </NavLink>
                          </li>
                        );
                      })}
                    </ul>
                  )}
                </section>
              );
            })}

            {shown.length === 0 && (
              <p className="px-3 py-6 text-center text-xs text-ink-600">
                No lesson matches that.
              </p>
            )}
          </div>
        </div>
      </nav>
    </>
  );
}

function StatusDot({ state }: { state?: "started" | "solved" }) {
  const colour =
    state === "solved"
      ? "bg-good-500 ring-2 ring-good-500/20"
      : state === "started"
        ? "bg-ink-400"
        : "bg-ink-700";
  const label =
    state === "solved" ? "Solved" : state === "started" ? "Started" : "Not started";
  return (
    <span
      className={`mt-[7px] size-1.5 shrink-0 rounded-full ${colour}`}
      title={label}
      aria-label={label}
    />
  );
}
