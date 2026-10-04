/**
 * Pack field value shapes. Three exist: text (string), map (string map)
 * and, since OpenProcessor #61, a list of strings (`proposal_denylist`,
 * case-insensitive globs). The backend's schema still serves
 * `proposal_denylist` as `kind: "text"`, so the list shape is recognised by
 * the served `kind: "list"` (if it ever says so) or the field id.
 */
import type { PackFieldValue } from '$lib/types_packs';

export const PROPOSAL_DENYLIST_FIELD = 'proposal_denylist';

export function isStringList(v: unknown): v is string[] {
  return Array.isArray(v) && v.every((x) => typeof x === 'string');
}

export function isListField(field: { field: string; kind: string }): boolean {
  return field.kind === 'list' || field.field === PROPOSAL_DENYLIST_FIELD;
}

/** The list a list field edits: a non-list stored value reads as empty. */
export function listValue(v: PackFieldValue | undefined): string[] {
  return isStringList(v) ? v : [];
}
