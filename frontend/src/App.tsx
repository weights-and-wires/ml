import { useCallback, useEffect, useState } from "react";
import { Link, Navigate, Route, Routes, useLocation } from "react-router-dom";

import Login from "./components/Login";
import Sidebar from "./components/Sidebar";
import Dashboard from "./pages/Dashboard";
import LessonPage from "./pages/LessonPage";
import { api, type Part, type ProgressMap } from "./lib/api";

export default function App() {
  const [username, setUsername] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [parts, setParts] = useState<Part[]>([]);
  const [progress, setProgress] = useState<ProgressMap>({});
  const [gpu, setGpu] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
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
      <header className="flex shrink-0 items-center gap-3 border-b border-ink-800 bg-ink-900 px-3 py-2">
        <button
          type="button"
          onClick={() => setMenuOpen((v) => !v)}
          aria-label="Show the lesson list"
          className="rounded-md p-1.5 text-ink-400 transition hover:bg-ink-850 hover:text-ink-200 lg:hidden"
        >
          <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden>
            <path d="M2 4h12M2 8h12M2 12h12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
          </svg>
        </button>

        <Link to="/" className="flex items-center gap-2">
          <Mark />
          <span className="text-[13px] font-medium tracking-tight text-ink-100">
            Machine learning
          </span>
        </Link>

        <div className="flex-1" />

        {gpu && (
          <span
            title="A RunPod GPU endpoint is configured"
            className="hidden rounded border border-warm-500/30 bg-warm-500/10 px-1.5 py-px font-mono text-[10px] uppercase text-warm-500 sm:inline"
          >
            gpu ready
          </span>
        )}
        <span className="hidden font-mono text-[11px] text-ink-400 sm:inline">
          {username}
        </span>
        <button
          type="button"
          onClick={() =>
            api
              .logout()
              .then(() => setUsername(null))
              .catch(() => undefined)
          }
          className="rounded-md px-2 py-1 text-[11px] text-ink-400 transition hover:bg-ink-850 hover:text-ink-200"
        >
          Sign out
        </button>
      </header>

      <div className="flex min-h-0 flex-1">
        <Sidebar
          parts={parts}
          progress={progress}
          open={menuOpen}
          onClose={() => setMenuOpen(false)}
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

function Mark() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden>
      <circle cx="5" cy="6" r="2.2" fill="#4f8cff" />
      <circle cx="5" cy="18" r="2.2" fill="#4f8cff" />
      <circle cx="19" cy="12" r="2.2" fill="#35c489" />
      <circle cx="12" cy="12" r="2.2" fill="#7b8496" />
      <path
        d="M6.8 7 10.4 10.8M6.8 17 10.4 13.2M13.8 12h3"
        stroke="#39404f"
        strokeWidth="1.4"
        strokeLinecap="round"
      />
    </svg>
  );
}
