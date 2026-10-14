/**
 * A span of seconds as `1h 05m`, `5m 03s` or `42s` (whole seconds, a negative
 * value is zero). `fallback` is printed when the value is missing or not a
 * finite number, so each surface keeps its own "no estimate yet" text.
 */
export function formatDuration(
  seconds: number | null | undefined,
  fallback = '—',
): string {
  if (seconds == null || !Number.isFinite(seconds)) return fallback;
  const total = Math.max(0, Math.round(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}h ${String(m).padStart(2, '0')}m`;
  if (m > 0) return `${m}m ${String(s).padStart(2, '0')}s`;
  return `${s}s`;
}
