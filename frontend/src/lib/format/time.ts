/** The local time of day of an epoch in milliseconds; `fallback` when it is null. */
export function formatTime(epochMs: number | null, fallback = '—'): string {
  return epochMs === null ? fallback : new Date(epochMs).toLocaleTimeString();
}
