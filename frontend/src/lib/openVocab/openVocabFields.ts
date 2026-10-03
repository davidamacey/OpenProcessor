/**
 * Rendering helpers for the open-vocabulary editor. The served schema's
 * rows (`scope` set / target / gating / tier3_hit_rate) are a subset of
 * the region-profile schema's, so `ProfileFieldEditor` renders them
 * through this adapter; every label, default, range and help text is
 * served. Nothing here checks a value: the server's validate route does.
 */
import { issuesForField } from '$lib/config/validationIssues';
import type { ValidationIssue, ValidationReport } from '$lib/types_config';
import type {
  OpenVocabFieldSchema,
  OpenVocabFieldScope,
  OpenVocabSchema,
  OpenVocabTargetBody,
} from '$lib/types_openVocab';
import type { ProfileSchemaField } from '$lib/types_profiles';

export function openVocabFieldAsProfileField(
  row: OpenVocabFieldSchema,
): ProfileSchemaField {
  return {
    field: row.field,
    label: row.label,
    group: row.scope,
    type: row.type,
    default: row.default as ProfileSchemaField['default'],
    min: row.min ?? null,
    max: row.max ?? null,
    enum: null,
    advanced: row.advanced ?? false,
    applies_when: null,
    choices_from: null,
    empty_choice: null,
    help: row.help ?? '',
  };
}

export function rowsByScope(
  schema: OpenVocabSchema,
): Record<OpenVocabFieldScope, OpenVocabFieldSchema[]> {
  const out: Record<OpenVocabFieldScope, OpenVocabFieldSchema[]> = {
    set: [],
    target: [],
    gating: [],
    tier3_hit_rate: [],
  };
  for (const f of schema.fields) out[f.scope].push(f);
  return out;
}

/** The dotted path the server's validator names a field with. */
export function issuePath(
  scope: OpenVocabFieldScope,
  field: string,
  targetIndex?: number,
): string {
  switch (scope) {
    case 'set':
      return field;
    case 'target':
      return `targets[${targetIndex ?? 0}].${field}`;
    case 'gating':
      return `gating.${field}`;
    case 'tier3_hit_rate':
      return `gating.tier3_hit_rate.${field}`;
  }
}

/** The served issues naming one cell of one target row. */
export function issuesForTargetField(
  report: ValidationReport | null,
  index: number,
  field: string,
): ValidationIssue[] {
  return issuesForField(report, issuePath('target', field, index));
}

/** A new target: each served `target` row's `default`, nothing invented. */
export function defaultTarget(schema: OpenVocabSchema): OpenVocabTargetBody {
  const out: Record<string, unknown> = {};
  for (const f of schema.fields) if (f.scope === 'target') out[f.field] = f.default;
  return out as OpenVocabTargetBody;
}
