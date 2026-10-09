/**
 * Rendering helpers for the region-profile form (any_domain_plan.md §7.3,
 * §7.6 item 2; docs/design/w4-profile-editor-ui-plan-2026-09-27.md §4).
 * Every list, label, group and range comes from the served schema and
 * vocabulary; these only look them up. No rule here judges a value.
 */
import type {
  AppliesWhen,
  Choice,
  ChoicesFrom,
  ConfigVocabulary,
  ProfileSchemaField,
  ProfileSchemaGroup,
  RegionProfileEffective,
  RegionProfileSchema,
} from '$lib/types_profiles';

/**
 * The served list a schema row's `choices_from` names, as its uniform
 * `choice` entries. Where each list lives is the spec's own table (§7.3
 * `CHOICE_SOURCES`): the vocabulary's top-level keys, or `ocr.*`.
 * `vlm_catalog` and `secret_refs` live on W9 routes, not the vocabulary:
 * `null`, and the row falls back to a text input (W4-Q8).
 */
export function choiceList(
  vocab: ConfigVocabulary | null,
  from: ChoicesFrom | null | undefined,
): Choice[] | null {
  if (!vocab || !from) return null;
  switch (from) {
    case 'detectors':
      return vocab.detectors.map((e) => e.choice);
    case 'segmenters':
      return vocab.segmenters.map((e) => e.choice);
    case 'registry_classes':
      return vocab.registry_classes.map((e) => e.choice);
    case 'text_reader_modes':
      return vocab.text_reader_modes.map((e) => e.choice);
    case 'ocr_pipeline_models':
      return vocab.ocr.pipeline_models.map((e) => e.choice);
    case 'ocr_rec_models':
      return vocab.ocr.rec_models.map((e) => e.choice);
    default:
      return null;
  }
}

export interface SelectOption {
  id: string;
  label: string;
  /** False for a stored value the served list doesn't carry. */
  listed: boolean;
}

/**
 * A single-value picker's options: the served `empty_choice` first, then
 * the served choices, then the stored value when the list doesn't carry
 * it (so opening a profile never silently changes a field; the server's
 * validation says whether that value is a problem).
 */
export function selectOptions(
  field: ProfileSchemaField,
  choices: Choice[],
  value: unknown,
): SelectOption[] {
  const out: SelectOption[] = [];
  if (field.empty_choice) {
    out.push({
      id: field.empty_choice.id ?? '',
      label: field.empty_choice.label,
      listed: true,
    });
  }
  for (const c of choices) out.push({ id: c.id, label: c.label, listed: true });
  if (typeof value === 'string' && !out.some((o) => o.id === value)) {
    out.push({ id: value, label: `${value} (not in the list)`, listed: false });
  }
  return out;
}

/**
 * Whether a row's `applies_when` holds in the saved revision, read from
 * that revision's served `effective` (W4-Q2: nothing is served for the
 * draft). `null` when the row has no condition, or the condition or the
 * `effective` block isn't known.
 */
export function appliesInSaved(
  appliesWhen: AppliesWhen | null | undefined,
  effective: RegionProfileEffective | null | undefined,
): boolean | null {
  if (!appliesWhen || !effective) return null;
  switch (appliesWhen) {
    case 'detector':
    case 'segmenter':
      return effective.legs.includes(appliesWhen);
    case 'reads_text':
      return effective.reads_text;
    case 'text_hint':
      return effective.text_hint_active;
    default:
      return null;
  }
}

export interface FieldGroup {
  id: string;
  label: string;
  fields: ProfileSchemaField[];
}

/** Schema rows grouped by the served `groups[]`, in served order; a group
 *  a row names but the list doesn't carry goes at the end under its id.
 *  Empty groups are dropped. */
export function groupFields(schema: RegionProfileSchema): FieldGroup[] {
  const groups: FieldGroup[] = schema.groups.map((g: ProfileSchemaGroup) => ({
    id: g.id,
    label: g.label,
    fields: [],
  }));
  for (const f of schema.fields) {
    let g = groups.find((x) => x.id === f.group);
    if (!g) {
      g = { id: f.group, label: f.group, fields: [] };
      groups.push(g);
    }
    g.fields.push(f);
  }
  return groups.filter((g) => g.fields.length > 0);
}

/** A served value as one line of text (defaults, list summaries). */
export function valueText(v: unknown): string {
  if (v == null || v === '') return '—';
  if (Array.isArray(v)) return v.length === 0 ? '—' : v.join(', ');
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
}

/** A number input's value: an empty input is `null` (W4-Q9). */
export function numberFromInput(raw: string): number | null {
  if (raw.trim() === '') return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
}
