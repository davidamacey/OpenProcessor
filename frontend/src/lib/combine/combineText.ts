/** Display helpers for the combine surfaces. Formatting only. */
import { humanizeId } from '$lib/humanizeId';

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
