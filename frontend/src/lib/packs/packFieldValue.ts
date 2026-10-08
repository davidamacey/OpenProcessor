/**
 * Pack field value shapes, chosen by the served `kind`: text (string), map
 * (string map) and list (`proposal_denylist`, case-insensitive globs).
 * Any other kind is shown read-only.
 */
import type { PackFieldValue } from '$lib/types_packs';

export const LIST_MAX_ENTRIES = 500;
export const LIST_MAX_ENTRY_CHARS = 200;

export function isStringList(v: unknown): v is string[] {
  return Array.isArray(v) && v.every((x) => typeof x === 'string');
}

/** Trim, then drop blanks and case-insensitive duplicates (first wins). */
export function normalizeList(rows: string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const r of rows) {
    const t = r.trim();
    const key = t.toLowerCase();
    if (!t || seen.has(key)) continue;
    seen.add(key);
    out.push(t);
  }
  return out;
}

/** Indexes of non-blank rows repeating an earlier row (case-insensitive). */
export function duplicateRows(rows: string[]): Set<number> {
  const seen = new Set<string>();
  const dups = new Set<number>();
  rows.forEach((r, i) => {
    const key = r.trim().toLowerCase();
    if (!key) return;
    if (seen.has(key)) dups.add(i);
    seen.add(key);
  });
  return dups;
}

/** The list a list field edits: a non-list stored value reads as empty. */
export function listValue(v: PackFieldValue | undefined): string[] {
  return isStringList(v) ? v : [];
}
