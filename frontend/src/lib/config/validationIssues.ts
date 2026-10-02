/**
 * Placing served validation issues (`any_domain_plan.md` §7.1
 * `ValidationIssue.field`, a dotted path) on the editor fields that own
 * them, and recognizing the `config.changed` events an editor follows.
 * Nothing here judges an issue; it only routes what the server said.
 */
import type { CurationEvent } from '$lib/sse';
import type { ValidationIssue, ValidationReport } from '$lib/types_config';

function owns(field: string, path: string | null): boolean {
  return path === field || (path ?? '').startsWith(`${field}.`);
}

/** The served issues whose `field` path names `field` (the path itself,
 *  or a dotted path under it, e.g. a map entry or a list element). */
export function issuesForField(
  report: ValidationReport | null,
  field: string,
): ValidationIssue[] {
  if (!report) return [];
  return [...report.errors, ...report.warnings].filter((i) => owns(field, i.field));
}

/** The served issues no listed field claims (whole-body, or a path the
 *  schema doesn't list). */
export function unplacedIssues(
  report: ValidationReport | null,
  fields: string[],
): ValidationIssue[] {
  if (!report) return [];
  return [...report.errors, ...report.warnings].filter(
    (i) => i.field == null || !fields.some((f) => owns(f, i.field)),
  );
}

/** True for a `config.changed` event on `axis` (§7.1: one axis id per
 *  axis: `prompt_pack`, `detection_profile`, ...). */
export function isConfigAxisEvent(e: CurationEvent, axis: string): boolean {
  return e.type === 'config.changed' && (e as { axis?: string }).axis === axis;
}
