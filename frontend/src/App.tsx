import { useCallback, useEffect, useState } from "react";
import { Link, Navigate, Route, Routes, useLocation } from "react-router-dom";

import Login from "./components/Login";
import Mark from "./components/Mark";
import Sidebar from "./components/Sidebar";
import Dashboard from "./pages/Dashboard";
import LessonPage from "./pages/LessonPage";
import { api, type Part, type ProgressMap } from "./lib/api";

const COLLAPSED_KEY = "ml.sidebar.collapsed";

function rememberedCollapse(): boolean {
  try {
    return window.localStorage.getItem(COLLAPSED_KEY) === "1";
  } catch {
    return false;
  }
}

export default function App() {
  const [username, setUsername] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [parts, setParts] = useState<Part[]>([]);
  const [progress, setProgress] = useState<ProgressMap>({});
  const [gpu, setGpu] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  // menuOpen drives the drawer on a narrow screen; collapsed hides the lesson
  // list on a wide one, which a reader wants once the editor is open.
  const [collapsed, setCollapsed] = useState(rememberedCollapse);
  const location = useLocation();

  useEffect(() => {
    api
      .session()
      .then((body) => setUsername(body.username))
      .catch(() => setUsername(null))
      .finally(() => setReady(true));
  }, []);

  const refresh = useCallback(() => {
    if (!username) return;
    api
      .lessons()
      .then((body) => {
        setParts(body.parts);
        setProgress(body.progress);
      })
      .catch(() => undefined);
  }, [username]);

  useEffect(() => {
    refresh();
    if (username) api.health().then((body) => setGpu(body.gpu)).catch(() => undefined);
  }, [username, refresh]);

  useEffect(() => setMenuOpen(false), [location.pathname]);

  useEffect(() => {
    try {
      window.localStorage.setItem(COLLAPSED_KEY, collapsed ? "1" : "0");
    } catch {
      // Storage being off only costs the reader their sidebar state.
    }
  }, [collapsed]);

  // Cmd or Ctrl with B shows and hides the lesson list, the way an editor
  // treats its own sidebar. Monaco leaves this combination alone.
  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (!(event.metaKey || event.ctrlKey) || event.altKey) return;
      if (event.key !== "b" && event.key !== "B") return;
      event.preventDefault();
      setCollapsed((v) => !v);
      setMenuOpen(false);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  if (!ready) {
    return (
      <div className="flex h-full items-center justify-center text-xs text-ink-600">
        …
      </div>
    );
  }

  if (!username) {
    return <Login onSignedIn={setUsername} />;
  }

  return (
    <div className="flex h-full flex-col">
      <header className="z-40 flex shrink-0 items-center gap-2 border-b border-ink-800 bg-ink-900/85 px-3 py-2 backdrop-blur-md">
        <button
          type="button"
          onClick={() => setMenuOpen((v) => !v)}
          aria-label={menuOpen ? "Hide the lesson list" : "Show the lesson list"}
          aria-expanded={menuOpen}
          className="flex size-10 items-center justify-center rounded-md text-ink-400 transition hover:bg-ink-850 hover:text-ink-200 lg:hidden"
        >
          <MenuIcon />
        </button>

        <button
          type="button"
          onClick={() => setCollapsed((v) => !v)}
          aria-label={collapsed ? "Show the lesson list" : "Hide the lesson list"}
          aria-expanded={!collapsed}
          title={`${collapsed ? "Show" : "Hide"} the lesson list (Cmd or Ctrl with B)`}
          className="hidden rounded-md p-1.5 text-ink-400 transition hover:bg-ink-850 hover:text-ink-200 lg:block"
        >
          <PanelIcon open={!collapsed} />
        </button>

        <Link
          to="/"
          className="flex min-h-10 items-center gap-2 rounded-md py-0.5 pr-1.5 transition hover:opacity-85 lg:min-h-0"
        >
          <span className="flex size-6 items-center justify-center rounded-md bg-ink-850 ring-1 ring-ink-800">
            <Mark size={14} />
          </span>
          <span className="text-[13px] font-medium tracking-tight text-ink-100">
            Machine learning
          </span>
        </Link>

        <div className="flex-1" />

        {gpu && (
          <span
            title="A RunPod GPU endpoint is configured"
            className="hidden items-center gap-1.5 rounded-full border border-warm-500/25 bg-warm-500/10 py-0.5 pl-2 pr-2.5 font-mono text-[10px] uppercase tracking-wide text-warm-500 sm:inline-flex"
          >
            <span className="size-1.5 rounded-full bg-warm-500" />
            gpu ready
          </span>
        )}
        <span className="hidden items-center gap-1.5 rounded-full border border-ink-800 bg-ink-850 py-0.5 pl-1 pr-2.5 sm:inline-flex">
          <span className="flex size-5 items-center justify-center rounded-full bg-wire-500/15 font-mono text-[10px] uppercase text-wire-400">
            {username.slice(0, 1)}
          </span>
          <span className="font-mono text-[11px] text-ink-300">{username}</span>
        </span>
        <button
          type="button"
          onClick={() =>
            api
              .logout()
              .then(() => setUsername(null))
              .catch(() => undefined)
          }
          className="rounded-md border border-transparent px-3 py-2.5 text-[12px] text-ink-400 transition hover:border-ink-700 hover:bg-ink-850 hover:text-ink-200 lg:px-2 lg:py-1 lg:text-[11px]"
        >
          Sign out
        </button>
      </header>

      <div className="flex min-h-0 flex-1">
        <Sidebar
          parts={parts}
          progress={progress}
          open={menuOpen}
          collapsed={collapsed}
          onClose={() => setMenuOpen(false)}
          onCollapse={() => setCollapsed(true)}
        />

        <main className="flex min-h-0 min-w-0 flex-1 flex-col">
          <Routes>
            <Route
              path="/"
              element={
                <Dashboard parts={parts} progress={progress} username={username} />
              }
            />
            <Route
              path="/lessons/:slug"
              element={<LessonPage gpuAvailable={gpu} onProgress={refresh} />}
            />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}

function MenuIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden>
      <path
        d="M2 4h12M2 8h12M2 12h12"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
    </svg>
  );
}

/** A pane outline whose left column fills in when the lesson list is showing. */
function PanelIcon({ open }: { open: boolean }) {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden>
      <rect
        x="1.75"
        y="2.75"
        width="12.5"
        height="10.5"
        rx="2"
        stroke="currentColor"
        strokeWidth="1.4"
      />
      <path
        d="M6.25 3v10"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
      />
      {open && <rect x="2.5" y="3.5" width="3" height="9" rx="1" fill="currentColor" />}
    </svg>
  );
}
