/**
 * Rendering helpers for the VLM endpoint form (any_domain_plan.md §7.8.5;
 * docs/design/w9-p4-w5-w10-ui-plan-2026-10-01.md §3.2). The VLM schema's
 * rows are a subset of the region-profile schema's (`string`/`int`/
 * `float`/`bool`/`enum`, `choices_from`, `empty_choice`, `min`/`max`,
 * `advanced`, `help`, `default`; no `applies_when`), so the profile field
 * editor renders them through this adapter. Every label, default, range
 * and list is served; these only look them up.
 */
import type {
  Choice,
  ProfileSchemaField,
  RegionProfileSchema,
} from '$lib/types_profiles';
import type {
  VlmCatalogResponse,
  VlmChoice,
  VlmEndpointBody,
  VlmEndpointFieldSchema,
  VlmEndpointList,
  VlmEndpointSchema,
} from '$lib/types_vlm';

const toChoice = (c: VlmChoice): Choice => ({ id: c.id ?? '', label: c.label });

/** One VLM schema row as the profile editor's row type. */
export function vlmFieldAsProfileField(row: VlmEndpointFieldSchema): ProfileSchemaField {
  return {
    field: row.field,
    label: row.label,
    group: row.group,
    type: row.type,
    default: row.default as ProfileSchemaField['default'],
    min: row.min ?? null,
    max: row.max ?? null,
    enum: row.enum ? row.enum.map(toChoice) : null,
    advanced: row.advanced ?? false,
    applies_when: null,
    choices_from: row.choices_from ?? null,
    empty_choice: row.empty_choice ?? null,
    help: row.help ?? '',
  };
}

export function vlmSchemaAsProfileSchema(schema: VlmEndpointSchema): RegionProfileSchema {
  return {
    fields: schema.fields.map(vlmFieldAsProfileField),
    groups: schema.groups,
  };
}

/**
 * The served list a row's `choices_from` names: `secret_refs` from the
 * registry list, `vlm_catalog` from the catalog. `null` while that read
 * hasn't landed (or failed), and the row falls back to a text input. A
 * choice with a null id is the "none" option, which the row's
 * `empty_choice` already carries, so it is not repeated here.
 */
export function vlmChoices(
  row: Pick<VlmEndpointFieldSchema, 'choices_from'>,
  list: VlmEndpointList | null,
  catalog: VlmCatalogResponse | null,
): Choice[] | null {
  const source =
    row.choices_from === 'secret_refs'
      ? list?.secret_refs.map((r) => r.choice)
      : row.choices_from === 'vlm_catalog'
        ? catalog?.entries.map((e) => e.choice)
        : null;
  if (!source) return null;
  return source.filter((c) => c.id != null).map(toChoice);
}

/** A new endpoint's draft: each served row's `default`. */
export function bodyFromDefaults(schema: VlmEndpointSchema): VlmEndpointBody {
  const body: Record<string, unknown> = {};
  for (const f of schema.fields) body[f.field] = f.default;
  return body as VlmEndpointBody;
}

/** A cleared "none" pick (`''` from a select) is the row's served
 *  `empty_choice.id` (null). */
export function normalizeFieldValue(
  schema: VlmEndpointSchema | null,
  field: string,
  value: unknown,
): unknown {
  const row = schema?.fields.find((f) => f.field === field);
  if (row?.empty_choice && value === '') return row.empty_choice.id;
  return value;
}
