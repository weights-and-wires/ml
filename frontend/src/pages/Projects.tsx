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
          <span className="kicker mb-3">Weights and wires · Machine learning</span>
          <h1 className="font-serif text-[clamp(35px,3.6vw,51px)] font-medium leading-[1.1] tracking-[-0.035em] text-ink">
            Projects
          </h1>
          <p className="mt-4 max-w-[38rem] text-[16px] leading-[1.8] text-muted">
            Each project puts several chapters to work on a realistic problem, from a messy
            spreadsheet to a small language model you serve. Each one says which chapter to
            finish first, and its checks judge the answer your whole pipeline gives.
          </p>
        </header>

        <section className="mt-12" aria-labelledby="projects-list">
          <div className="flex items-baseline gap-3 border-b border-strong-line pb-2">
            <h2 id="projects-list" className="min-w-0 flex-1 font-serif text-[29px] font-medium tracking-[-0.035em] text-ink">
              All projects
            </h2>
            <span className="meta tabular-nums">
              {solved} of {projects.length}
            </span>
          </div>
          <ol>
            {projects.map((project) => (
              <li key={project.slug} className="border-b border-rule">
                <Link
                  to={`/lessons/${project.slug}`}
                  className="group -mx-3 flex items-start gap-3 rounded-md px-3 py-3.5 transition-colors hover:bg-sunk"
                >
                  <span className="w-6 shrink-0 pt-1 font-mono text-[11px] tabular-nums text-accent-ink">
                    {numbers.get(project.slug)}
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block text-[15px] font-medium leading-snug text-ink group-hover:text-forest">
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
