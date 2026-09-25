/** One accent per part, so a reader can tell where in the course they are. */
const PART_ACCENTS = ["#4f8cff", "#9b7bff", "#2fbf9b", "#f0a637", "#f06ba8", "#3cc4e6"];

export function partAccent(part: string): string {
  const number = Number(/Part\s+(\d+)/i.exec(part)?.[1] ?? 1);
  return PART_ACCENTS[(number - 1 + PART_ACCENTS.length) % PART_ACCENTS.length];
}

/** "Part 3 — Neural networks" → { number: "3", name: "Neural networks" }. */
export function splitPart(part: string): { number: string; name: string } {
  const match = /^Part\s+(\d+)\s*[—–-]\s*(.+)$/i.exec(part);
  return match ? { number: match[1], name: match[2] } : { number: "", name: part };
}
