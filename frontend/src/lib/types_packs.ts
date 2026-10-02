/**
 * Wire types for OpenProcessor W3 (prompt-pack CRUD) and the pack half of
 * W5 (test-on-crop): `any_domain_plan.md` §3, §5.1, §7.1, §7.2, §7.5.
 * Every route is project-scoped (`{prefix}/prompt_packs...`).
 *
 * Built against the frozen spec before the backend lands W3, so nothing
 * here is pinned to the vendored OpenAPI yet. Where the spec leaves a
 * shape open, the type is the literal reading recorded in
 * docs/design/w3-pack-editor-ui-plan-2026-09-27.md §6 (question W3-Qn).
 */

import type {
  ActiveRef,
  ConfigDocBase,
  ConfigSource,
  ConfigUpdateRequest,
  ConfigValidateRequest,
  ValidationReport,
} from './types_config';

/**
 * A pack body: every `PromptPack` field except `name`, keyed by the
 * schema's `field` ids. `kind: "text"` fields are strings, `kind: "map"`
 * fields (`class_descriptions`, `synonyms`) are string maps. Kept as a
 * record so a field the backend adds renders from the schema alone.
 */
export type PackFieldValue = string | Record<string, string>;
export type PromptPackBody = Record<string, PackFieldValue>;

/** One `GET /prompt_packs` `packs[]` row. */
export interface PromptPackSummary {
  name: string;
  source: ConfigSource;
  read_only: boolean;
  revision: number | null;
  etag: string;
  description: string | null;
  asks_region_text: boolean;
  active: boolean;
  /** Present when `active`: the revision that is active. */
  active_revision?: number | null;
  updated_at: string | null;
}

/** One `GET /prompt_packs` `templates[]` row (clone-only, §3.1). */
export interface PromptPackTemplate {
  name: string;
  source: ConfigSource;
  read_only: boolean;
  path: string;
}

export interface PromptPackList {
  packs: PromptPackSummary[];
  templates: PromptPackTemplate[];
  active: ActiveRef;
  config_revision: number;
  stale: boolean;
}

/** `GET /prompt_packs/{name}` and `.../revisions/{rev}`. */
export type PromptPackDoc = ConfigDocBase<PromptPackBody>;

/** One `GET /prompt_packs/schema` `fields[]` row (§3.4). */
export interface PackSchemaField {
  field: string;
  label: string;
  group: string;
  kind: 'text' | 'map' | (string & {});
  formatted: boolean;
  required_placeholders: string[];
  allowed_placeholders: string[];
  expected_reply_keys: string[];
  optional_reply_keys: string[];
  used_by: string[];
  help: string;
}

export interface PackSchemaPlaceholder {
  name: string;
  meaning: string;
  example: string;
}

/** A pack call (`combined`, `classify`, ...). `testable` gates the
 *  test-on-crop panel for that call (W3-Q2). */
export interface PackSchemaCall {
  id: string;
  label: string;
  fields: string[];
  testable: boolean;
}

export interface PromptPackSchema {
  fields: PackSchemaField[];
  placeholders: PackSchemaPlaceholder[];
  calls: PackSchemaCall[];
  reply_key_contract: Record<string, { required: string[]; optional: string[] }>;
}

export type PackUpdateRequest = ConfigUpdateRequest<PromptPackBody>;

export type PackValidateRequest = ConfigValidateRequest<PromptPackBody>;

/** `POST /prompt_packs/test` (§7.5). Exactly one pack source: `draft`, or
 *  `pack_name` (+ `pack_revision`). Unset keys take the server's default
 *  (registry classes, active profile, active VLM endpoint). */
export interface PackTestRequest {
  pack_name?: string | null;
  pack_revision?: number | null;
  draft?: PromptPackBody | null;
  call: string;
  crop_ids: string[];
  use_region_box?: 'current' | 'none';
}

export interface PackTestVlm {
  name: string | null;
  revision: number | null;
  draft: boolean;
  model: string | null;
  resolved_model: string | null;
  sends_images_externally: boolean;
}

/** One `results[]` entry. `preview` is `preview_item` mapped through
 *  `mapRawCrop` by `testPromptPack`. */
export interface PackTestResult<P = unknown> {
  crop_id: string;
  /** Region-verify results carry the box they judged (W3-Q15). */
  box_id?: string | null;
  parse_ok: boolean;
  parse_error: string | null;
  parsed_combined: unknown;
  parsed_region: unknown;
  parsed_class: unknown;
  parsed_visible: unknown;
  preview_item: Record<string, unknown> | null;
  preview?: P | null;
}

export interface PackTestResponse<P = unknown> {
  call: string;
  pack: { name: string | null; revision: number | null; draft: boolean };
  vlm: PackTestVlm | null;
  prompt: { system: string; user_text: string };
  raw_reply: string;
  reasoning: string | null;
  latency_ms: number;
  validation: ValidationReport | null;
  results: PackTestResult<P>[];
}
