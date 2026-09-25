import { useEffect, useRef, useState, type CSSProperties } from "react";
import { Link } from "react-router-dom";

import type { RunEvent } from "../lib/api";

export type ConsoleLine =
  | { kind: "stdout" | "stderr" | "status"; text: string }
  | { kind: "error"; text: string; detail?: string | null };

export type CheckResult = Extract<RunEvent, { type: "check" }>;

type Props = {
  lines: ConsoleLine[];
  checks: CheckResult[];
  planned: number;
  running: boolean;
  mode: "run" | "check" | null;
  backend: string | null;
  summary: Extract<RunEvent, { type: "done" }> | null;
  /** The next lesson's slug, offered once every check passes. */
  next?: string | null;
};

/**
 * The bottom half of the exercise pane: a transcript for **Run**, and a list of
 * checks for **Check**. Both scroll to the newest line while a run is going.
 */
export default function RunConsole({
  lines,
  checks,
  planned,
  running,
  mode,
  backend,
  summary,
  next,
}: Props) {
  const bottom = useRef<HTMLDivElement>(null);
  const top = useRef<HTMLDivElement>(null);
  const [stuck, setStuck] = useState(true);
  const solved = mode === "check" && !running && !!summary?.ok;

  // A full pass is news, so it scrolls the banner into view rather than the
  // end of the list.
  useEffect(() => {
    if (solved) top.current?.scrollIntoView({ block: "start" });
  }, [solved]);

  useEffect(() => {
    if (stuck && !solved) bottom.current?.scrollIntoView({ block: "end" });
  }, [lines.length, checks.length, stuck, summary, solved]);

  const idle = !running && lines.length === 0 && checks.length === 0;

  return (
    <div className="flex min-h-0 flex-1 flex-col bg-ink-950">
      <header className="flex shrink-0 items-center gap-2 border-b border-ink-800 px-3 py-1.5">
        <span className="text-[11px] font-medium uppercase tracking-[0.14em] text-ink-400">
          {mode === "check" ? "Checks" : "Output"}
        </span>
        {backend && (
          <span
            className={`rounded border px-1.5 py-px font-mono text-[10px] uppercase ${
              backend === "runpod"
                ? "border-warm-500/30 bg-warm-500/10 text-warm-500"
                : "border-ink-700 bg-ink-850 text-ink-400"
            }`}
          >
            {backend === "runpod" ? "gpu · runpod" : "cpu · local"}
          </span>
        )}
        {running && (
          <span className="flex items-center gap-1.5 text-[11px] text-wire-400">
            <Spinner />
            running
          </span>
        )}
        <div className="flex-1" />
        {summary && !running && <Summary summary={summary} mode={mode} />}
      </header>

      <div
        className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-3 py-2 font-mono text-[12.5px] leading-relaxed"
        onScroll={(event) => {
          const el = event.currentTarget;
          setStuck(el.scrollHeight - el.scrollTop - el.clientHeight < 40);
        }}
      >
        {idle && (
          <p className="px-1 py-6 font-sans text-xs text-ink-600">
            Press <Key>Run</Key> to execute the file, or <Key>Check</Key> to run
            the exercise's checks.
          </p>
        )}

        <div ref={top} />
        {mode === "check" && planned > 0 && (
          <Scoreboard
            passed={checks.filter((c) => c.status !== "failed").length}
            done={checks.length}
            planned={planned}
            solved={solved}
            next={next}
          />
        )}

        {mode === "check" && planned > 0 && (
          <ol className="mb-2 space-y-1">
            {Array.from({ length: planned }).map((_, index) => {
              const result = checks.find((c) => c.index === index);
              return (
                <CheckRow key={index} index={index} result={result} running={running} />
              );
            })}
          </ol>
        )}

        {lines.map((line, index) => (
          <Line key={index} line={line} />
        ))}

        <div ref={bottom} />
      </div>
    </div>
  );
}

/**
 * How many checks pass so far, as a bar that fills in, and a celebration with
 * a way forward once they all do.
 */
function Scoreboard({
  passed,
  done,
  planned,
  solved,
  next,
}: {
  passed: number;
  done: number;
  planned: number;
  solved: boolean;
  next?: string | null;
}) {
  if (solved) {
    return (
      <div className="solved appear mb-2 font-sans">
        <div className="confetti" aria-hidden>
          {Array.from({ length: 14 }).map((_, i) => (
            <span key={i} style={{ "--i": i } as CSSProperties} />
          ))}
        </div>
        <p className="text-[13px] font-semibold text-ink-100">
          All {planned} checks pass. You built it.
        </p>
        <p className="mt-0.5 text-[12px] text-ink-300">
          The lesson is marked solved.
          {next ? " The next one builds on what you just wrote." : ""}
        </p>
        {next && (
          <Link to={`/lessons/${next}`} className="solved-next">
            Next: {sentence(next)}
            <span aria-hidden className="cta-arrow">
              →
            </span>
          </Link>
        )}
      </div>
    );
  }
  const failing = done - passed;
  return (
    <div className="mb-2 px-0.5 font-sans">
      <div className="mb-1 flex items-center justify-between text-[11px]">
        <span className="text-ink-300">
          {passed} of {planned} passing
        </span>
        {failing > 0 && done === planned && (
          <span className="text-warm-500">
            Open a failed check to see what it expected
          </span>
        )}
      </div>
      <div className="flex h-1.5 gap-0.5 overflow-hidden rounded-full">
        {Array.from({ length: planned }).map((_, i) => (
          <span
            key={i}
            className={`flex-1 transition-colors duration-300 ${
              i < passed ? "bg-good-500" : i < done ? "bg-bad-500/70" : "bg-ink-800"
            }`}
          />
        ))}
      </div>
    </div>
  );
}

/** "02-linear-regression" → "Linear regression". */
function sentence(slug: string): string {
  const words = slug.replace(/^\d+-/, "").replace(/-/g, " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

function CheckRow({
  index,
  result,
  running,
}: {
  index: number;
  result?: CheckResult;
  running: boolean;
}) {
  const [open, setOpen] = useState(false);
  const failed = result?.status === "failed";
  const pending = !result;

  return (
    <li
      className={`appear rounded-md border px-2.5 py-1.5 ${
        pending
          ? "border-ink-850 bg-ink-900/40"
          : failed
            ? "border-bad-500/25 bg-bad-500/6"
            : "border-good-500/20 bg-good-500/6"
      }`}
    >
      <button
        type="button"
        disabled={!result?.message && !result?.output}
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-start gap-2 text-left disabled:cursor-default"
      >
        <Glyph pending={pending} failed={failed} running={running} />
        <span
          className={`min-w-0 flex-1 font-sans text-[12.5px] leading-snug ${
            pending ? "text-ink-600" : failed ? "text-ink-200" : "text-ink-300"
          }`}
        >
          {result?.description ?? `Check ${index + 1}`}
        </span>
        {result && (
          <span className="shrink-0 font-mono text-[10px] tabular-nums text-ink-600">
            {result.seconds < 0.01 ? "<10ms" : `${(result.seconds * 1000).toFixed(0)}ms`}
          </span>
        )}
      </button>

      {open && result?.message && (
        <pre className="mt-1.5 overflow-x-auto whitespace-pre-wrap border-l-2 border-bad-500/40 pl-2.5 text-[11.5px] text-bad-500">
          {result.message}
        </pre>
      )}
      {open && result?.output && (
        <pre className="mt-1.5 overflow-x-auto whitespace-pre-wrap border-l-2 border-ink-700 pl-2.5 text-[11.5px] text-ink-400">
          {result.output}
        </pre>
      )}
      {!open && failed && result?.message && (
        <p className="mt-1 truncate pl-6 font-sans text-[11.5px] text-bad-500/90">
          {result.message.split("\n")[0]}
        </p>
      )}
    </li>
  );
}

function Glyph({
  pending,
  failed,
  running,
}: {
  pending: boolean;
  failed: boolean;
  running: boolean;
}) {
  if (pending) {
    return running ? (
      <span className="mt-1 size-3 shrink-0">
        <Spinner />
      </span>
    ) : (
      <span className="mt-[5px] size-2 shrink-0 rounded-full border border-ink-700" />
    );
  }
  return failed ? (
    <svg viewBox="0 0 16 16" className="mt-0.5 size-3.5 shrink-0 text-bad-500" aria-label="Failed">
      <path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  ) : (
    <svg viewBox="0 0 16 16" className="mt-0.5 size-3.5 shrink-0 text-good-500" aria-label="Passed">
      <path d="M3.5 8.5l3 3 6-7" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function Line({ line }: { line: ConsoleLine }) {
  if (line.kind === "status") {
    return (
      <p className="appear py-px font-sans text-[11.5px] text-ink-600">
        {line.text}
      </p>
    );
  }
  if (line.kind === "error") {
    return (
      <div className="appear my-1.5 rounded-md border border-bad-500/25 bg-bad-500/8 px-2.5 py-1.5">
        <p className="font-sans text-[12.5px] text-bad-500">{line.text}</p>
        {line.detail && (
          <pre className="mt-1 overflow-x-auto whitespace-pre-wrap text-[11.5px] text-bad-500/80">
            {line.detail}
          </pre>
        )}
      </div>
    );
  }
  return (
    <pre
      className={`whitespace-pre-wrap break-words ${
        line.kind === "stderr" ? "text-warm-500" : "text-ink-200"
      }`}
    >
      {line.text || " "}
    </pre>
  );
}

function Summary({
  summary,
  mode,
}: {
  summary: Extract<RunEvent, { type: "done" }>;
  mode: "run" | "check" | null;
}) {
  const seconds = summary.seconds ? `${summary.seconds.toFixed(1)}s` : null;
  if (mode === "check" && summary.total != null) {
    return (
      <span
        className={`font-mono text-[11px] ${summary.ok ? "text-good-500" : "text-bad-500"}`}
      >
        {summary.passed}/{summary.total} passed{seconds ? ` · ${seconds}` : ""}
      </span>
    );
  }
  return (
    <span
      className={`font-mono text-[11px] ${summary.ok ? "text-ink-400" : "text-bad-500"}`}
    >
      {summary.ok ? "finished" : "failed"}
      {seconds ? ` · ${seconds}` : ""}
    </span>
  );
}

function Spinner() {
  return (
    <svg viewBox="0 0 16 16" className="size-3 animate-spin" aria-hidden>
      <circle cx="8" cy="8" r="6" stroke="currentColor" strokeOpacity="0.25" strokeWidth="2" fill="none" />
      <path d="M14 8a6 6 0 0 0-6-6" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" />
    </svg>
  );
}

function Key({ children }: { children: string }) {
  return (
    <kbd className="rounded border border-ink-700 bg-ink-850 px-1 py-px font-mono text-[10px] text-ink-300">
      {children}
    </kbd>
  );
}
