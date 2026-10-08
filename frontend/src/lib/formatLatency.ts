/**
 * A served latency (milliseconds, often an unrounded float) as whole
 * milliseconds with a unit; `null`/`undefined`/non-finite is "—", never 0.
 */
export function formatLatencyMs(ms: number | null | undefined): string {
  return ms == null || !Number.isFinite(ms) ? '—' : `${Math.round(ms)} ms`;
}
