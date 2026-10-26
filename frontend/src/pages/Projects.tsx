import { useMemo } from "react";
import { Link } from "react-router-dom";

import StatusMark from "../components/StatusMark";
import type { LessonMeta, ProgressMap } from "../lib/api";
import { lessonNumbers, projectAfter } from "../lib/parts";

type Props = { projects: LessonMeta[]; progress: ProgressMap };

/**
 * The projects, on a page of their own that the header links to. The lesson
 * list beside it keeps to the chapters.
 */
export default function Projects({ projects, progress }: Props) {
  const numbers = useMemo(() => lessonNumbers(projects), [projects]);
  const solved = projects.filter((p) => progress[p.slug]?.status === "solved").length;

  return (
    <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain">
      <div className="mx-auto max-w-3xl px-5 pb-16 pt-10 sm:px-8 lg:pt-14">
        <header>
          <h1 className="font-serif text-[2.1rem] font-semibold leading-[1.15] tracking-[-0.015em] text-ink sm:text-[2.5rem]">
            Projects
          </h1>
          <p className="mt-3 max-w-[38rem] font-serif text-[1.08rem] leading-relaxed text-ink-3">
            Each project puts several chapters to work on a realistic problem, from a messy
            spreadsheet to a small language model you serve. Each one says which chapter to
            finish first, and its checks judge the answer your whole pipeline gives.
          </p>
        </header>

        <section className="mt-12" aria-labelledby="projects-list">
          <div className="flex items-baseline gap-3 border-b border-rule-strong pb-2">
            <h2 id="projects-list" className="min-w-0 flex-1 font-serif text-[1.2rem] font-semibold text-ink">
              All projects
            </h2>
            <span className="text-[12.5px] tabular-nums text-ink-4">
              {solved} of {projects.length}
            </span>
          </div>
          <ol>
            {projects.map((project) => (
              <li key={project.slug} className="border-b border-rule">
                <Link
                  to={`/lessons/${project.slug}`}
                  className="group -mx-3 flex items-start gap-3 rounded-md px-3 py-3.5 transition-colors hover:bg-[color-mix(in_oklab,var(--ink)_4%,transparent)]"
                >
                  <span className="w-6 shrink-0 pt-0.5 text-[13px] tabular-nums text-ink-4">
                    {numbers.get(project.slug)}
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block text-[15px] font-medium leading-snug text-ink group-hover:text-accent">
                      {project.title}
                    </span>
                    <span className="mt-0.5 block text-[13.5px] leading-snug text-ink-3">
                      <span className="text-ink-4">{projectAfter(project.slug)} · </span>
                      {project.summary}
                    </span>
                  </span>
                  <span className="flex shrink-0 items-center gap-3 pt-0.5">
                    {project.minutes ? (
                      <span className="hidden text-[12.5px] tabular-nums text-ink-4 sm:inline">
                        {project.minutes} min
                      </span>
                    ) : null}
                    <StatusMark state={progress[project.slug]?.status} />
                  </span>
                </Link>
              </li>
            ))}
          </ol>
        </section>
      </div>
    </div>
  );
}
