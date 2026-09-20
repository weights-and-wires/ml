import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";

import { api, type Part, type ProgressMap } from "../lib/api";

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
    <div className="min-h-0 flex-1 overflow-y-auto">
      <div className="mx-auto max-w-4xl px-6 py-10 lg:px-10">
        <header className="mb-8">
          <p className="text-[11px] font-medium uppercase tracking-[0.18em] text-ink-400">
            Weights and wires
          </p>
          <h1 className="mt-1.5 text-3xl font-semibold tracking-tight text-ink-100">
            Machine learning
          </h1>
          <p className="mt-2.5 max-w-2xl text-sm leading-relaxed text-ink-400">
            A course for people who already have the mathematics. Every lesson
            derives its result and then asks you to implement it, and the checks
            verify the mathematics rather than the shape of your code.
          </p>
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
            className="group mb-9 flex items-center gap-4 rounded-xl border border-wire-500/30 bg-gradient-to-br from-wire-500/12 to-wire-500/5 p-5 transition hover:border-wire-500/50 hover:from-wire-500/16"
          >
            <span className="min-w-0 flex-1">
              <span className="text-[11px] font-medium uppercase tracking-[0.14em] text-wire-400">
                {progress[next.slug] ? "Carry on with" : "Start with"}
              </span>
              <span className="mt-1.5 block text-lg font-medium text-ink-100">
                {next.title}
              </span>
              <span className="mt-1 block text-[13px] leading-snug text-ink-300">
                {next.summary}
              </span>
            </span>
            <span className="flex size-8 shrink-0 items-center justify-center rounded-full border border-wire-500/30 text-wire-400 transition group-hover:translate-x-0.5 group-hover:border-wire-500/60 group-hover:bg-wire-500/15">
              <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden>
                <path
                  d="M2 7h9m-3.5-4 4 4-4 4"
                  stroke="currentColor"
                  strokeWidth="1.6"
                  fill="none"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </span>
          </Link>
        )}

        <div className="space-y-8">
          {parts.map((part) => (
            <section key={part.part}>
              <h2 className="mb-3 text-[11px] font-medium uppercase tracking-[0.14em] text-ink-400">
                {part.part}
              </h2>
              <ul className="grid gap-2 sm:grid-cols-2">
                {part.lessons.map((lesson) => {
                  const state = progress[lesson.slug]?.status;
                  return (
                    <li key={lesson.slug}>
                      <Link
                        to={`/lessons/${lesson.slug}`}
                        className="flex h-full flex-col rounded-lg border border-ink-800 bg-ink-900/70 p-3.5 transition duration-150 hover:-translate-y-px hover:border-wire-500/30 hover:bg-ink-850"
                      >
                        <div className="mb-1 flex items-center gap-2">
                          <span
                            className={`size-1.5 shrink-0 rounded-full ${
                              state === "solved"
                                ? "bg-good-500"
                                : state === "started"
                                  ? "bg-ink-400"
                                  : "bg-ink-700"
                            }`}
                          />
                          <span className="min-w-0 flex-1 truncate text-[13.5px] font-medium text-ink-100">
                            {lesson.title}
                          </span>
                          {lesson.gpu && (
                            <span className="shrink-0 rounded border border-warm-500/30 bg-warm-500/10 px-1 font-mono text-[9px] uppercase text-warm-500">
                              gpu
                            </span>
                          )}
                        </div>
                        <p className="line-clamp-2 text-[12px] leading-snug text-ink-400">
                          {lesson.summary}
                        </p>
                      </Link>
                    </li>
                  );
                })}
              </ul>
            </section>
          ))}
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

function formatSeconds(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  if (seconds < 3600) return `${(seconds / 60).toFixed(0)}m`;
  return `${(seconds / 3600).toFixed(1)}h`;
}
