import { useCallback, useEffect, useRef, useState } from "react";

import { api, ApiError, runStream, type ExerciseDetail, type RunEvent } from "../lib/api";
import CodeEditor from "./CodeEditor";
import RunConsole, { type CheckResult, type ConsoleLine } from "./RunConsole";
import SplitPane from "./SplitPane";

type Props = {
  exercise: ExerciseDetail;
  initialCode: string;
  /** Whether a RunPod endpoint is configured at all. */
  gpuAvailable: boolean;
  /** Whether this lesson's work is worth sending to a GPU. */
  gpuLesson: boolean;
  onSolved: () => void;
};

const AUTOSAVE_MS = 1200;

export default function ExercisePane({
  exercise,
  initialCode,
  gpuAvailable,
  gpuLesson,
  onSolved,
}: Props) {
  const [code, setCode] = useState(initialCode);
  const [lines, setLines] = useState<ConsoleLine[]>([]);
  const [checks, setChecks] = useState<CheckResult[]>([]);
  const [planned, setPlanned] = useState(0);
  const [running, setRunning] = useState(false);
  const [mode, setMode] = useState<"run" | "check" | null>(null);
  const [backend, setBackend] = useState<string | null>(null);
  const [summary, setSummary] = useState<Extract<RunEvent, { type: "done" }> | null>(
    null,
  );
  const [hintsOpen, setHintsOpen] = useState(false);
  // Checks are sized for the CPU and a cold serverless worker costs more than
  // the whole suite, so the GPU is opt-in and only where a lesson asks for it.
  const [useGpu, setUseGpu] = useState(false);
  const [revealed, setRevealed] = useState<string | null>(null);
  const [saved, setSaved] = useState<"idle" | "saving" | "saved">("idle");

  const abort = useRef<AbortController | null>(null);
  const codeRef = useRef(code);
  codeRef.current = code;
  const gpuRef = useRef(useGpu);
  gpuRef.current = useGpu;

  // A new exercise resets everything the previous one left behind.
  useEffect(() => {
    setCode(initialCode);
    setLines([]);
    setChecks([]);
    setPlanned(0);
    setSummary(null);
    setMode(null);
    setBackend(null);
    setHintsOpen(false);
    setUseGpu(false);
    setRevealed(null);
    setSaved("idle");
  }, [exercise.id, initialCode]);

  // Autosave the draft, so a closed tab never loses work.
  useEffect(() => {
    if (code === initialCode) return;
    setSaved("saving");
    const timer = window.setTimeout(() => {
      api
        .saveDraft(exercise.id, code)
        .then(() => setSaved("saved"))
        .catch(() => setSaved("idle"));
    }, AUTOSAVE_MS);
    return () => window.clearTimeout(timer);
  }, [code, exercise.id, initialCode]);

  useEffect(() => () => abort.current?.abort(), []);

  const start = useCallback(
    async (next: "run" | "check") => {
      abort.current?.abort();
      const controller = new AbortController();
      abort.current = controller;

      setRunning(true);
      setMode(next);
      setLines([]);
      setChecks([]);
      setPlanned(0);
      setSummary(null);
      setBackend(null);

      try {
        for await (const event of runStream(
          {
            exercise_id: exercise.id,
            mode: next,
            code: codeRef.current,
            prefer_gpu: gpuRef.current,
          },
          controller.signal,
        )) {
          switch (event.type) {
            case "begin":
              setBackend(event.backend);
              break;
            case "plan":
              setPlanned(event.total);
              break;
            case "check":
              setChecks((current) => [...current, event]);
              break;
            case "status":
              setLines((current) => [...current, { kind: "status", text: event.text }]);
              break;
            case "stdout":
            case "stderr":
              setLines((current) => [
                ...current,
                { kind: event.type, text: event.text },
              ]);
              break;
            case "error":
              setLines((current) => [
                ...current,
                { kind: "error", text: event.message, detail: event.detail },
              ]);
              break;
            case "done":
              setSummary(event);
              if (next === "check" && event.ok) onSolved();
              break;
          }
        }
      } catch (exc) {
        if (controller.signal.aborted) return;
        setLines((current) => [
          ...current,
          {
            kind: "error",
            text:
              exc instanceof ApiError
                ? exc.message
                : "Lost the connection to the runner.",
          },
        ]);
      } finally {
        if (!controller.signal.aborted) setRunning(false);
      }
    },
    [exercise.id, onSolved],
  );

  function stop() {
    abort.current?.abort();
    setRunning(false);
    setLines((current) => [...current, { kind: "status", text: "Stopped." }]);
  }

  async function reset() {
    if (!window.confirm("Replace your code with the starter file?")) return;
    await api.resetDraft(exercise.id).catch(() => undefined);
    setCode(exercise.starter);
  }

  async function reveal() {
    if (revealed) {
      setRevealed(null);
      return;
    }
    if (
      !window.confirm(
        "Show the reference solution? Your own code stays where it is.",
      )
    ) {
      return;
    }
    try {
      const body = await api.solution(exercise.id);
      setRevealed(body.solution);
    } catch {
      setRevealed("# The solution could not be loaded.");
    }
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col bg-ink-900">
      <header className="shrink-0 border-b border-ink-800 px-3 py-2">
        <div className="flex items-center gap-2">
          <h2 className="min-w-0 flex-1 truncate text-[13px] font-medium text-ink-100">
            {exercise.title}
          </h2>
          {gpuLesson && (
            <span
              title={
                gpuAvailable
                  ? "The larger run in this exercise is worth a GPU"
                  : "No GPU endpoint is configured, so every run stays on the CPU"
              }
              className={`rounded border px-1.5 py-px font-mono text-[10px] uppercase ${
                gpuAvailable
                  ? "border-warm-500/30 bg-warm-500/10 text-warm-500"
                  : "border-ink-700 bg-ink-850 text-ink-500"
              }`}
            >
              gpu
            </span>
          )}
          <span className="font-mono text-[10px] text-ink-600">
            {exercise.timeout}s limit
          </span>
        </div>
        <p className="mt-0.5 line-clamp-2 text-[12px] leading-snug text-ink-400">
          {exercise.brief}
        </p>

        <div className="mt-2 flex flex-wrap items-center gap-1.5">
          {running ? (
            <button
              type="button"
              onClick={stop}
              className="rounded-md border border-bad-500/40 bg-bad-500/10 px-3.5 py-2 text-xs font-medium text-bad-500 transition hover:bg-bad-500/20 lg:px-2.5 lg:py-1"
            >
              Stop
            </button>
          ) : (
            <>
              <button
                type="button"
                onClick={() => start("run")}
                className="rounded-md border border-ink-700 bg-ink-850 px-3.5 py-2 text-xs font-medium text-ink-200 transition hover:border-ink-600 hover:bg-ink-800 lg:px-2.5 lg:py-1"
                title="Run the file (Cmd or Ctrl with Return)"
              >
                Run
              </button>
              <button
                type="button"
                onClick={() => start("check")}
                className="rounded-md bg-wire-500 px-3.5 py-2 text-xs font-medium text-white transition hover:bg-wire-600 lg:px-2.5 lg:py-1"
                title="Run the checks (Cmd or Ctrl with Shift and Return)"
              >
                Check
              </button>

              {gpuLesson ? (
                <label
                  title={
                    gpuAvailable
                      ? "Send this run to a RunPod GPU worker. A cold worker takes " +
                        "a few seconds to start, so the checks are quicker on the CPU."
                      : "No GPU endpoint is configured, so every run stays on the CPU."
                  }
                  className={`ml-1 flex items-center gap-1.5 rounded-md border px-2.5 py-2 text-xs transition lg:px-2 lg:py-1 ${
                    !gpuAvailable
                      ? "cursor-not-allowed border-ink-800 text-ink-600"
                      : useGpu
                        ? "cursor-pointer border-warm-500/40 bg-warm-500/10 text-warm-500"
                        : "cursor-pointer border-ink-700 text-ink-400 hover:border-ink-600 hover:text-ink-300"
                  }`}
                >
                  <input
                    type="checkbox"
                    className="size-3 accent-warm-500"
                    checked={useGpu && gpuAvailable}
                    disabled={!gpuAvailable}
                    onChange={(event) => setUseGpu(event.target.checked)}
                  />
                  GPU
                </label>
              ) : null}
            </>
          )}

          <div className="flex-1" />

          <span className="font-mono text-[10px] text-ink-600">
            {saved === "saving" ? "saving…" : saved === "saved" ? "saved" : ""}
          </span>
          {exercise.hints.length > 0 && (
            <button
              type="button"
              onClick={() => setHintsOpen((v) => !v)}
              aria-expanded={hintsOpen}
              className="rounded-md px-2.5 py-2 text-xs text-ink-400 transition hover:bg-ink-850 hover:text-ink-200 lg:px-2 lg:py-1"
            >
              Hints ({exercise.hints.length})
            </button>
          )}
          <button
            type="button"
            onClick={reset}
            className="rounded-md px-2.5 py-2 text-xs text-ink-400 transition hover:bg-ink-850 hover:text-ink-200 lg:px-2 lg:py-1"
          >
            Reset
          </button>
          <button
            type="button"
            onClick={reveal}
            className="rounded-md px-2.5 py-2 text-xs text-ink-400 transition hover:bg-ink-850 hover:text-ink-200 lg:px-2 lg:py-1"
          >
            {revealed ? "Hide solution" : "Solution"}
          </button>
        </div>

        {hintsOpen && (
          <ol className="mt-2 space-y-1.5 rounded-md border border-ink-800 bg-ink-850 p-2.5">
            {exercise.hints.map((hint, index) => (
              <li key={index} className="flex gap-2 text-[12px] leading-snug text-ink-300">
                <span className="shrink-0 font-mono text-[10px] text-ink-600">
                  {index + 1}
                </span>
                <span>{hint}</span>
              </li>
            ))}
          </ol>
        )}
      </header>

      {revealed !== null ? (
        <div className="min-h-0 flex-1 overflow-auto overscroll-contain bg-ink-950">
          <div className="border-b border-ink-800 bg-ink-900 px-3 py-1.5 text-[11px] uppercase tracking-[0.14em] text-ink-400">
            Reference solution
          </div>
          <pre className="overflow-x-auto p-3 font-mono text-[12.5px] leading-relaxed text-ink-200">
            {revealed}
          </pre>
        </div>
      ) : (
        <SplitPane
          direction="column"
          storageKey={`ml.split.exercise`}
          initial={62}
          min={25}
          max={85}
          className="min-h-0 flex-1"
          label="Resize the editor and the output"
          first={
            <div className="min-h-0 flex-1 overflow-hidden">
              <CodeEditor
                value={code}
                onChange={setCode}
                onRun={() => start("run")}
                onCheck={() => start("check")}
              />
            </div>
          }
          second={
            <RunConsole
              lines={lines}
              checks={checks}
              planned={planned}
              running={running}
              mode={mode}
              backend={backend}
              summary={summary}
            />
          }
        />
      )}
    </div>
  );
}
