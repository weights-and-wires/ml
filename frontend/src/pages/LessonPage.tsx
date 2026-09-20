import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import ExercisePane from "../components/ExercisePane";
import Markdown from "../components/Markdown";
import SplitPane from "../components/SplitPane";
import { api, ApiError, type Lesson } from "../lib/api";

type Props = {
  gpuAvailable: boolean;
  onProgress: () => void;
};

export default function LessonPage({ gpuAvailable, onProgress }: Props) {
  const { slug = "" } = useParams();
  const navigate = useNavigate();
  const [lesson, setLesson] = useState<Lesson | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    setLesson(null);
    setError(null);
    api
      .lesson(slug)
      .then((body) => {
        if (!live) return;
        setLesson(body);
        onProgress();
      })
      .catch((exc) => {
        if (!live) return;
        setError(
          exc instanceof ApiError && exc.status === 404
            ? "There is no lesson at that address."
            : "Could not load the lesson.",
        );
      });
    return () => {
      live = false;
    };
  }, [slug, onProgress]);

  // Left and right arrows move between lessons when the focus is not in an
  // input or the editor.
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
      if (event.key === "[" && lesson?.prev) navigate(`/lessons/${lesson.prev}`);
      if (event.key === "]" && lesson?.next) navigate(`/lessons/${lesson.next}`);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [lesson, navigate]);

  const onSolved = useCallback(() => {
    onProgress();
  }, [onProgress]);

  if (error) {
    return (
      <div className="flex flex-1 items-center justify-center p-8 text-center">
        <div>
          <p className="text-sm text-ink-300">{error}</p>
          <Link
            to="/"
            className="mt-3 inline-block text-sm text-wire-400 underline underline-offset-4"
          >
            Back to the dashboard
          </Link>
        </div>
      </div>
    );
  }

  if (!lesson) {
    return (
      <div className="flex flex-1 items-center justify-center text-xs text-ink-600">
        Loading…
      </div>
    );
  }

  const reading = (
    <article className="min-h-0 flex-1 overflow-y-auto">
      <div className="mx-auto max-w-[68ch] px-6 py-7 lg:px-9">
        <div className="mb-5 flex flex-wrap items-center gap-2 text-[11px] text-ink-400">
          <span className="uppercase tracking-[0.14em]">{lesson.part}</span>
          <span className="text-ink-700">·</span>
          <span className="font-mono">
            {lesson.index} of {lesson.count}
          </span>
          {lesson.minutes ? (
            <>
              <span className="text-ink-700">·</span>
              <span>{lesson.minutes} min</span>
            </>
          ) : null}
        </div>

        {lesson.objectives.length > 0 && (
          <section className="mb-7 rounded-lg border border-ink-800 bg-ink-900 p-4">
            <h2 className="mb-2 text-[11px] font-medium uppercase tracking-[0.14em] text-ink-400">
              By the end you can
            </h2>
            <ul className="space-y-1.5">
              {lesson.objectives.map((objective, index) => (
                <li
                  key={index}
                  className="flex gap-2 text-[13px] leading-snug text-ink-300"
                >
                  <span className="mt-[7px] size-1 shrink-0 rounded-full bg-wire-500" />
                  <span>{objective}</span>
                </li>
              ))}
            </ul>
            {lesson.prereqs.length > 0 && (
              <p className="mt-3 border-t border-ink-800 pt-2.5 text-[12px] text-ink-400">
                Builds on{" "}
                {lesson.prereqs.map((prereq, index) => (
                  <span key={prereq}>
                    {index > 0 && ", "}
                    <Link
                      to={`/lessons/${prereq}`}
                      className="text-wire-400 underline underline-offset-4"
                    >
                      {prereq.replace(/^\d+-/, "").replace(/-/g, " ")}
                    </Link>
                  </span>
                ))}
                .
              </p>
            )}
          </section>
        )}

        <Markdown>{lesson.body}</Markdown>

        <nav className="mt-10 flex items-stretch gap-3 border-t border-ink-800 pt-5">
          {lesson.prev ? (
            <Link
              to={`/lessons/${lesson.prev}`}
              className="flex-1 rounded-lg border border-ink-800 px-3.5 py-2.5 transition hover:border-ink-700 hover:bg-ink-900"
            >
              <span className="block text-[10px] uppercase tracking-[0.14em] text-ink-600">
                Previous
              </span>
              <span className="mt-0.5 block truncate text-[13px] text-ink-200">
                {lesson.prev.replace(/^\d+-/, "").replace(/-/g, " ")}
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
                Next
              </span>
              <span className="mt-0.5 block truncate text-[13px] text-ink-200">
                {lesson.next.replace(/^\d+-/, "").replace(/-/g, " ")}
              </span>
            </Link>
          ) : (
            <span className="flex-1" />
          )}
        </nav>
      </div>
    </article>
  );

  if (!lesson.exercise_detail) {
    return <div className="flex min-h-0 flex-1 flex-col">{reading}</div>;
  }

  return (
    <SplitPane
      storageKey="ml.split.lesson"
      initial={52}
      min={28}
      max={72}
      className="min-h-0 flex-1"
      label="Resize the lesson and the editor"
      first={reading}
      second={
        <ExercisePane
          key={lesson.exercise_detail.id}
          exercise={lesson.exercise_detail}
          initialCode={lesson.draft ?? lesson.exercise_detail.starter}
          gpuAvailable={gpuAvailable}
          gpuLesson={lesson.gpu}
          onSolved={onSolved}
        />
      }
    />
  );
}
