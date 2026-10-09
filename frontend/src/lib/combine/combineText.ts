/** Display helpers for the combine surfaces. Formatting only. */
import { humanizeId } from '$lib/humanizeId';

/** Bytes as KB/MB/GB; `null` (not served) is "—". */
export function formatBytes(n: number | null | undefined): string {
  if (n == null) return '—';
  if (n < 1024) return `${n} B`;
  const units = ['KB', 'MB', 'GB', 'TB'];
  let v = n;
  let i = -1;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(1)} ${units[i]}`;
}

/** The backend serves no labels for job status / phase, dedup / holdout
 *  values or issue codes (plan question P4-3): print the id humanized. */
export function combineLabel(id: string | null | undefined): string {
  return id ? humanizeId(id) : '—';
}

/** A report value as one table cell: scalars verbatim, anything
 *  structured as compact JSON. */
export function reportCell(v: unknown): string {
  if (v == null) return '—';
  if (typeof v === 'string') return v;
  if (typeof v === 'number' || typeof v === 'boolean') return String(v);
  return JSON.stringify(v);
}
