/**
 * m14 (2026-09-24 interactive pass): `/classes`' "Added" column rendered
 * `new Date(cls.added_at).toLocaleDateString()`. `added_at` is a
 * date-only string ("2026-04-29") with no time/offset, so `Date` parses
 * it as UTC midnight; `toLocaleDateString()` then converts to the
 * browser's local zone, which — for any zone behind UTC (the common
 * case, e.g. US zones) — rolls it back to the previous day (4/28).
 *
 * A date-only string names a calendar day, not an instant, so it should
 * never go through a UTC-to-local conversion at all: this reads the
 * Y-M-D components directly off the string and formats them, with no
 * `Date`/timezone math in between.
 */
export function formatDateOnly(value: string | null | undefined): string {
  if (!value) return '—';
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(value);
  if (!m) {
    // Not a bare date string (has a time/offset component) — safe to let
    // the platform do full instant-to-local conversion.
    const d = new Date(value);
    return Number.isNaN(d.getTime()) ? '—' : d.toLocaleDateString();
  }
  const [, y, mo, d] = m;
  return `${Number(mo)}/${Number(d)}/${y}`;
}
