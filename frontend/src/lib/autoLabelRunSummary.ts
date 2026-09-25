/**
 * D5 (visual audit 2026-09-24): the dashboard's "Last run summary" was a
 * pretty-printed JSON dump of the served auto-label `result`. This turns
 * `result.stages` into one row per stage — the served status/skip flag,
 * the served reason, and the stage's own scalar fields — without
 * interpreting any of them. Nested objects stay in the raw-JSON fallback.
 */

export interface RunStageRow {
  key: string;
  status: string;
  reason: string | null;
  /** Scalar fields of the stage (numbers/strings/booleans), in served order. */
  fields: Array<{ name: string; value: string }>;
}

const SKIP_KEYS = new Set(['status', 'skipped', 'reason']);

export function summarizeRunStages(result: unknown): RunStageRow[] {
  const stages = (result as { stages?: unknown } | null | undefined)?.stages;
  if (!stages || typeof stages !== 'object') return [];
  return Object.entries(stages as Record<string, unknown>).map(([key, raw]) => {
    const stage = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>;
    const status =
      typeof stage.status === 'string'
        ? stage.status
        : stage.skipped === true
          ? 'skipped'
          : '—';
    const reason = typeof stage.reason === 'string' ? stage.reason : null;
    const fields: RunStageRow['fields'] = [];
    for (const [name, v] of Object.entries(stage)) {
      if (SKIP_KEYS.has(name)) continue;
      if (typeof v === 'number' || typeof v === 'string' || typeof v === 'boolean') {
        fields.push({ name, value: String(v) });
      }
    }
    return { key, status, reason, fields };
  });
}
