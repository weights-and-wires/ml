/**
 * The course mark: two inputs, a hidden unit, and an output. It appears in the
 * header and on the sign-in page, so it lives in one place.
 */
export default function Mark({ size = 16 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden
    >
      <circle cx="5" cy="6" r="2.2" fill="var(--color-wire-500)" />
      <circle cx="5" cy="18" r="2.2" fill="var(--color-wire-500)" />
      <circle cx="19" cy="12" r="2.2" fill="var(--color-good-500)" />
      <circle cx="12" cy="12" r="2.2" fill="var(--color-ink-400)" />
      <path
        d="M6.8 7 10.4 10.8M6.8 17 10.4 13.2M13.8 12h3"
        stroke="var(--color-ink-600)"
        strokeWidth="1.4"
        strokeLinecap="round"
      />
    </svg>
  );
}
