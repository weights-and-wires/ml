import { useState, type FormEvent } from "react";

import { api, ApiError } from "../lib/api";

export default function Login({ onSignedIn }: { onSignedIn: (name: string) => void }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const result = await api.login(username.trim(), password);
      onSignedIn(result.username);
    } catch (exc) {
      setError(
        exc instanceof ApiError
          ? exc.message
          : "Could not reach the server. Try again.",
      );
      setBusy(false);
    }
  }

  return (
    <div className="flex min-h-full items-center justify-center bg-ink-950 px-6">
      <div className="w-full max-w-sm">
        <div className="mb-8">
          <div className="mb-3 flex items-center gap-2.5">
            <Mark />
            <span className="text-[11px] font-medium uppercase tracking-[0.18em] text-ink-400">
              Weights and wires
            </span>
          </div>
          <h1 className="text-2xl font-semibold tracking-tight text-ink-100">
            Machine learning
          </h1>
          <p className="mt-2 text-sm leading-relaxed text-ink-400">
            Thirty-three lessons, from least squares to mixtures of experts.
            Every one of them has code to write.
          </p>
        </div>

        <form onSubmit={submit} className="space-y-3">
          <Field
            label="Username"
            value={username}
            onChange={setUsername}
            autoComplete="username"
            autoFocus
          />
          <Field
            label="Password"
            value={password}
            onChange={setPassword}
            type="password"
            autoComplete="current-password"
          />

          {error && (
            <p
              role="alert"
              className="rounded-lg border border-bad-500/30 bg-bad-500/10 px-3 py-2 text-sm text-bad-500"
            >
              {error}
            </p>
          )}

          <button
            type="submit"
            disabled={busy || !username || !password}
            className="w-full rounded-lg bg-wire-500 px-4 py-2.5 text-sm font-medium text-white transition hover:bg-wire-600 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {busy ? "Signing in…" : "Sign in"}
          </button>
        </form>
      </div>
    </div>
  );
}

function Field({
  label,
  value,
  onChange,
  type = "text",
  autoComplete,
  autoFocus,
}: {
  label: string;
  value: string;
  onChange: (next: string) => void;
  type?: string;
  autoComplete?: string;
  autoFocus?: boolean;
}) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-xs font-medium text-ink-400">
        {label}
      </span>
      <input
        type={type}
        value={value}
        autoComplete={autoComplete}
        autoFocus={autoFocus}
        onChange={(event) => onChange(event.target.value)}
        className="w-full rounded-lg border border-ink-800 bg-ink-900 px-3 py-2.5 text-sm text-ink-100 transition placeholder:text-ink-600 focus:border-wire-500/50 focus:outline-none"
      />
    </label>
  );
}

function Mark() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden>
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
