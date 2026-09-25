import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";

import type { CSSProperties } from "react";

import { api, type Part, type ProgressMap } from "../lib/api";
import { partAccent, splitPart } from "../lib/parts";

type Props = { parts: Part[]; progress: ProgressMap; username: string };

export default function Dashboard({ parts, progress, username }: Props) {
  const [me, setMe] = useState<Awaited<ReturnType<typeof api.me>> | null>(null);

  useEffect(() => {
    api.me().then(setMe).catch(() => undefined);
  }, []);

  const lessons = useMemo(() => parts.flatMap((part) => part.lessons), [parts]);
  const solved = lessons.filter((l) => progress[l.slug]?.status === "solved");
  const started = lessons.filter((l) => progress[l.slug]?.status === "started");
  const next =
    lessons.find((l) => progress[l.slug]?.status === "started") ??
    lessons.find((l) => !progress[l.slug]) ??
    lessons[0];

  return (
    <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain">
      <div className="mx-auto max-w-4xl px-6 py-10 lg:px-10">
        <header className="mb-8 flex items-center gap-6">
          <div className="min-w-0 flex-1">
            <p className="text-[11px] font-medium uppercase tracking-[0.18em] text-ink-400">
              Weights and wires
            </p>
            <h1 className="mt-1.5 text-3xl font-semibold tracking-tight text-ink-100">
              Machine learning, one build at a time
            </h1>
            <p className="mt-2.5 max-w-2xl text-sm leading-relaxed text-ink-400">
              Every lesson explains one idea in plain words, works a small
              example by hand, and then walks you through building it. Your
              code runs against checks that tell you exactly what is right and
              what is not.
            </p>
          </div>
          <ProgressRing done={solved.length} total={lessons.length} />
        </header>

        <div className="mb-9 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat label="Solved" value={`${solved.length}`} of={`${lessons.length}`} />
          <Stat label="In progress" value={`${started.length}`} />
          <Stat label="Runs" value={me ? `${me.stats.runs}` : "—"} />
          <Stat
            label="Compute"
            value={me ? formatSeconds(me.stats.compute_seconds) : "—"}
          />
        </div>

        {next && (
          <Link
            to={`/lessons/${next.slug}`}
            style={{ "--accent": partAccent(next.part) } as CSSProperties}
            className="continue group mb-10 flex items-center gap-4 rounded-2xl p-5 transition"
          >
            <span className="continue-num">
              {String(lessons.indexOf(next) + 1).padStart(2, "0")}
            </span>
            <span className="min-w-0 flex-1">
              <span className="text-[11px] font-semibold uppercase tracking-[0.16em] text-[color-mix(in_oklab,var(--accent)_75%,white)]">
                {progress[next.slug] ? "Pick up where you left off" : "Start here"}
              </span>
              <span className="mt-1.5 block text-lg font-medium text-ink-100">
                {next.title}
              </span>
              <span className="mt-1 block text-[13px] leading-snug text-ink-300">
                {next.summary}
              </span>
              {next.minutes ? (
                <span className="mt-2 inline-block font-mono text-[11px] text-ink-500">
                  about {next.minutes} min
                </span>
              ) : null}
            </span>
            <span className="continue-go">
              {progress[next.slug] ? "Continue" : "Start"}
              <span aria-hidden className="cta-arrow">
                →
              </span>
            </span>
          </Link>
        )}

        <div className="space-y-9">
          {parts.map((part) => {
            const { number, name } = splitPart(part.part);
            const done = part.lessons.filter(
              (l) => progress[l.slug]?.status === "solved",
            ).length;
            return (
              <section
                key={part.part}
                style={{ "--accent": partAccent(part.part) } as CSSProperties}
              >
                <div className="mb-3 flex items-center gap-3">
                  <span className="part-badge">{number || "·"}</span>
                  <h2 className="min-w-0 flex-1 truncate text-[15px] font-semibold text-ink-100">
                    {name}
                  </h2>
                  <span className="font-mono text-[11px] tabular-nums text-ink-500">
                    {done}/{part.lessons.length}
                  </span>
                </div>
                <div className="mb-3 h-1 overflow-hidden rounded-full bg-ink-800">
                  <div
                    className="h-full rounded-full bg-[var(--accent)] transition-[width] duration-500"
                    style={{
                      width: `${part.lessons.length ? (done / part.lessons.length) * 100 : 0}%`,
                    }}
                  />
                </div>
                <ul className="grid gap-2 sm:grid-cols-2">
                  {part.lessons.map((lesson) => {
                    const state = progress[lesson.slug]?.status;
                    const index = lessons.indexOf(lesson) + 1;
                    return (
                      <li key={lesson.slug}>
                        <Link
                          to={`/lessons/${lesson.slug}`}
                          className="lesson-card group flex h-full gap-3 rounded-xl p-3.5"
                          data-state={state ?? "new"}
                        >
                          <span className="lesson-card-num">
                            {state === "solved" ? "✓" : String(index).padStart(2, "0")}
                          </span>
                          <span className="min-w-0 flex-1">
                            <span className="mb-1 flex items-center gap-2">
                              <span className="min-w-0 flex-1 truncate text-[13.5px] font-medium text-ink-100">
                                {lesson.title}
                              </span>
                              {lesson.gpu && (
                                <span className="shrink-0 rounded border border-warm-500/30 bg-warm-500/10 px-1 font-mono text-[9px] uppercase text-warm-500">
                                  gpu
                                </span>
                              )}
                            </span>
                            <span className="line-clamp-2 block text-[12px] leading-snug text-ink-400">
                              {lesson.summary}
                            </span>
                            <span className="mt-1.5 flex items-center gap-2 font-mono text-[10.5px] text-ink-500">
                              {lesson.minutes ? <span>{lesson.minutes} min</span> : null}
                              {state === "started" && (
                                <span className="text-[color-mix(in_oklab,var(--accent)_75%,white)]">
                                  in progress
                                </span>
                              )}
                              {state === "solved" && <span className="text-good-500">solved</span>}
                            </span>
                          </span>
                        </Link>
                      </li>
                    );
                  })}
                </ul>
              </section>
            );
          })}
        </div>

        {me && me.runs.length > 0 && (
          <section className="mt-10 border-t border-ink-800 pt-6">
            <h2 className="mb-3 text-[11px] font-medium uppercase tracking-[0.14em] text-ink-400">
              Recent runs, {username}
            </h2>
            <ul className="space-y-1">
              {me.runs.slice(0, 10).map((run, index) => (
                <li
                  key={index}
                  className="flex items-center gap-3 rounded-md px-2 py-1.5 font-mono text-[11.5px] text-ink-400 transition odd:bg-ink-900/70 hover:bg-ink-850"
                >
                  <span
                    className={`size-1.5 shrink-0 rounded-full ${run.ok ? "bg-good-500" : "bg-bad-500"}`}
                  />
                  <span className="min-w-0 flex-1 truncate text-ink-300">
                    {run.exercise_id}
                  </span>
                  <span>{run.mode}</span>
                  <span className="text-ink-600">{run.backend}</span>
                  {run.total != null && (
                    <span className="tabular-nums">
                      {run.passed}/{run.total}
                    </span>
                  )}
                  <span className="w-12 text-right tabular-nums text-ink-600">
                    {run.seconds ? `${run.seconds.toFixed(1)}s` : ""}
                  </span>
                </li>
              ))}
            </ul>
          </section>
        )}
      </div>
    </div>
  );
}

function Stat({ label, value, of }: { label: string; value: string; of?: string }) {
  return (
    <div className="rounded-lg border border-ink-800 bg-ink-900/70 px-3.5 py-3 transition hover:border-ink-700">
      <div className="text-[10px] font-medium uppercase tracking-[0.14em] text-ink-400">
        {label}
      </div>
      <div className="mt-1 font-mono text-xl text-ink-100 tabular-nums">
        {value}
        {of && <span className="text-sm text-ink-600">/{of}</span>}
      </div>
    </div>
  );
}

/** The share of lessons solved, as a ring that fills in. */
function ProgressRing({ done, total }: { done: number; total: number }) {
  const r = 30;
  const c = 2 * Math.PI * r;
  const share = total ? done / total : 0;
  return (
    <div
      className="relative hidden size-20 shrink-0 sm:block"
      role="img"
      aria-label={`${done} of ${total} lessons solved`}
    >
      <svg viewBox="0 0 72 72" className="size-full -rotate-90">
        <circle cx="36" cy="36" r={r} fill="none" stroke="var(--color-ink-800)" strokeWidth="6" />
        <circle
          cx="36"
          cy="36"
          r={r}
          fill="none"
          stroke="url(#ring)"
          strokeWidth="6"
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={c * (1 - share)}
          style={{ transition: "stroke-dashoffset 600ms ease-out" }}
        />
        <defs>
          <linearGradient id="ring" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="var(--color-wire-500)" />
            <stop offset="1" stopColor="var(--color-good-500)" />
          </linearGradient>
        </defs>
      </svg>
      <span className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="font-mono text-base leading-none text-ink-100 tabular-nums">
          {Math.round(share * 100)}%
        </span>
        <span className="mt-1 text-[9px] uppercase tracking-[0.12em] text-ink-500">solved</span>
      </span>
    </div>
  );
}

function formatSeconds(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  if (seconds < 3600) return `${(seconds / 60).toFixed(0)}m`;
  return `${(seconds / 3600).toFixed(1)}h`;
}
