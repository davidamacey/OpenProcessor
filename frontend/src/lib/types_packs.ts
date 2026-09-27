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

export type ValidationSeverity = 'error' | 'warning' | 'info';

/** §7.1 `ValidationIssue`. `field` is a dotted path; null = whole body. */
export interface ValidationIssue {
  code: string;
  severity: ValidationSeverity;
  field: string | null;
  message: string;
  detail: Record<string, unknown>;
  bypassable: boolean;
}

/** §7.1 `ValidationReport`. `force_allowed` is the only thing that offers
 *  "Activate anyway". */
export interface ValidationReport {
  ok: boolean;
  errors: ValidationIssue[];
  warnings: ValidationIssue[];
  force_allowed: boolean;
}

/** §7.1 `ActiveRef`. `name: null` = no active pack (the deployment
 *  default applies). `revision: null` for builtin/file packs. */
export interface ActiveRef {
  name: string | null;
  revision: number | null;
}

/** `source` of a pack. No served labels (W3-Q4). */
export type PackSource = 'builtin' | 'file' | 'stored' | 'template' | (string & {});

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
  source: PackSource;
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
  source: PackSource;
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
export interface PromptPackDoc {
  name: string;
  source: PackSource;
  read_only: boolean;
  revision: number | null;
  etag: string;
  description: string | null;
  body: PromptPackBody;
  created_at: string | null;
  updated_at: string | null;
  updated_by: string | null;
  cloned_from: string | null;
  active: boolean;
  active_revision?: number | null;
  validation: ValidationReport;
}

export interface PromptPackRevision {
  revision: number;
  saved_at: string;
  cloned_from: string | null;
  description: string | null;
}

export interface PromptPackRevisionList {
  name: string;
  revisions: PromptPackRevision[];
}

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

/** One `applied[]` entry (§7.3 `AppliedRuntime`). */
export interface AppliedRuntime {
  process: string;
  host: string;
  applied_config_revision: number;
  profile?: ActiveRef | null;
  pack?: ActiveRef | null;
  applied_at: string | null;
  lagging: boolean;
}

/** `GET /prompt_packs/active`, and the activate / rollback response.
 *  `source`, `activated_at` and `applied` are optional: the W2 branch
 *  model doesn't carry them yet (W3-Q5). */
export interface ActiveConfigResponse {
  axis: string;
  active: ActiveRef;
  source?: PackSource | null;
  activated_at?: string | null;
  previous: ActiveRef | null;
  config_revision: number;
  stale: boolean;
  applied?: AppliedRuntime[];
}

export interface PackCloneRequest {
  new_name: string;
  revision: number | null;
  /** `"template"` disambiguates a template from a pack of the same name. */
  source: PackSource | null;
  description: string | null;
}

export interface PackUpdateRequest {
  expected_revision: number;
  description: string | null;
  body: PromptPackBody;
}

export interface PackActivateRequest {
  revision: number | null;
  expected_active: ActiveRef;
  force: boolean;
}

export interface PackValidateRequest {
  name: string | null;
  body: PromptPackBody;
}

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

/** `ConfigErrorDetail` (§7.1) with the fields pack routes set. */
export interface PackErrorDetail {
  error: string;
  message: string;
  current_revision?: number | null;
  report?: ValidationReport | null;
  current?: ActiveRef | null;
  axis?: string | null;
  valid_ids?: string[] | null;
}
