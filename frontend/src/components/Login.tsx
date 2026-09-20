import { useState, type FormEvent } from "react";

import { api, ApiError } from "../lib/api";
import Mark from "./Mark";

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
    <div className="flex min-h-full items-center justify-center px-6 py-10">
      <div className="w-full max-w-sm rounded-2xl border border-ink-800 bg-ink-900/70 p-7 shadow-2xl shadow-ink-950/60 backdrop-blur-sm">
        <div className="mb-7">
          <div className="mb-3 flex items-center gap-2.5">
            <span className="flex size-7 items-center justify-center rounded-lg bg-ink-850 ring-1 ring-ink-800">
              <Mark size={16} />
            </span>
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
            className="w-full rounded-lg bg-wire-500 px-4 py-2.5 text-sm font-medium text-white shadow-lg shadow-wire-500/20 transition hover:bg-wire-600 disabled:cursor-not-allowed disabled:opacity-40 disabled:shadow-none"
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
        className="w-full rounded-lg border border-ink-800 bg-ink-850 px-3 py-2.5 text-sm text-ink-100 transition placeholder:text-ink-600 hover:border-ink-700 focus:border-wire-500/50 focus:bg-ink-900 focus:outline-none"
      />
    </label>
  );
}
