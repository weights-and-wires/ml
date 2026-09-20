import { useEffect, useMemo, useState } from "react";
import { NavLink, useParams } from "react-router-dom";

import type { Part, ProgressMap } from "../lib/api";

type Props = {
  parts: Part[];
  progress: ProgressMap;
  open: boolean;
  onClose: () => void;
};

const OPEN_PARTS_KEY = "ml.sidebar.parts";

export default function Sidebar({ parts, progress, open, onClose }: Props) {
  const { slug } = useParams();
  const [collapsed, setCollapsed] = useState<Set<string>>(() => {
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
        JSON.stringify([...collapsed]),
      );
    } catch {
      // Storage being off only costs the reader their collapse state.
    }
  }, [collapsed]);

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
    setCollapsed((current) => {
      const next = new Set(current);
      if (next.has(part)) next.delete(part);
      else next.add(part);
      return next;
    });
  }

  return (
    <>
      {open && (
        <button
          type="button"
          aria-label="Close the lesson list"
          onClick={onClose}
          className="fixed inset-0 z-20 bg-ink-950/70 lg:hidden"
        />
      )}

      <nav
        aria-label="Lessons"
        className={`fixed inset-y-0 left-0 z-30 flex w-72 flex-col border-r border-ink-800 bg-ink-900 transition-transform lg:static lg:translate-x-0 ${
          open ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="border-b border-ink-800 px-4 py-3.5">
          <div className="mb-3 flex items-baseline justify-between">
            <span className="text-[11px] font-medium uppercase tracking-[0.16em] text-ink-400">
              Curriculum
            </span>
            <span className="font-mono text-[11px] text-ink-400">
              {totals.solved}/{totals.all}
            </span>
          </div>
          <div className="h-1 overflow-hidden rounded-full bg-ink-800">
            <div
              className="h-full rounded-full bg-good-500 transition-[width] duration-500"
              style={{
                width: `${totals.all ? (totals.solved / totals.all) * 100 : 0}%`,
              }}
            />
          </div>
          <input
            type="search"
            value={filter}
            onChange={(event) => setFilter(event.target.value)}
            placeholder="Filter lessons"
            aria-label="Filter lessons"
            className="mt-3 w-full rounded-md border border-ink-800 bg-ink-850 px-2.5 py-1.5 text-xs text-ink-200 placeholder:text-ink-600 focus:border-wire-500/50 focus:outline-none"
          />
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-2 py-2">
          {shown.map((part) => {
            const shut = collapsed.has(part.part) && !query;
            const done = part.lessons.filter(
              (l) => progress[l.slug]?.status === "solved",
            ).length;
            return (
              <section key={part.part} className="mb-1">
                <button
                  type="button"
                  onClick={() => toggle(part.part)}
                  aria-expanded={!shut}
                  className="flex w-full items-center gap-1.5 rounded-md px-2 py-1.5 text-left text-[11px] font-medium uppercase tracking-[0.1em] text-ink-400 transition hover:bg-ink-850 hover:text-ink-300"
                >
                  <svg
                    width="10"
                    height="10"
                    viewBox="0 0 10 10"
                    className={`shrink-0 transition-transform ${shut ? "" : "rotate-90"}`}
                    aria-hidden
                  >
                    <path d="M3 1.5 7 5 3 8.5" stroke="currentColor" strokeWidth="1.4" fill="none" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  <span className="flex-1 truncate">{part.part}</span>
                  <span className="font-mono text-[10px] tabular-nums text-ink-600">
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
                              `group flex items-start gap-2 rounded-md px-2 py-1.5 text-[13px] leading-snug transition ${
                                isActive || slug === lesson.slug
                                  ? "bg-wire-500/12 text-ink-100"
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
      </nav>
    </>
  );
}

function StatusDot({ state }: { state?: "started" | "solved" }) {
  const colour =
    state === "solved"
      ? "bg-good-500"
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
