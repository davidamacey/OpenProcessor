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
} from './types_config';

/**
 * A pack body: every `PromptPack` field except `name`, keyed by the
 * schema's `field` ids. `kind: "text"` fields are strings, `kind: "map"`
 * fields (`class_descriptions`, `synonyms`) are string maps, and
 * `kind: "list"` (`proposal_denylist`) is a list of case-insensitive glob strings, and `kind: "int"` (`registry_prior_top_k`, 0 = off) is shown read-only.
 * Kept as a record so a field the backend adds renders from the schema alone.
 */
export type PackFieldValue = string | number | string[] | Record<string, string>;
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
  kind: 'text' | 'map' | 'list' | 'int' | (string & {});
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
