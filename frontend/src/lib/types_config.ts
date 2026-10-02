/**
 * Wire types shared by every editable config resource OpenProcessor serves
 * from its config store (W2): prompt packs (W3) and region profiles (W4).
 * `any_domain_plan.md` §7.1 (the shared models), §7.2 and §7.3 (the common
 * envelopes: revisions, clone, update, activate).
 *
 * Built against the frozen spec before the backend lands W3/W4, so nothing
 * here is pinned to the vendored OpenAPI yet.
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

/** §7.1 `ActiveRef`. `name: null` = nothing active on the axis (no pack:
 *  the deployment default applies; no profile: region detection is off).
 *  `revision: null` for a source without revisions (builtin, file, env,
 *  registered). */
export interface ActiveRef {
  name: string | null;
  revision: number | null;
}

/** Where a config doc comes from. No served labels (W3-Q4). */
export type ConfigSource =
  'builtin' | 'file' | 'stored' | 'template' | 'env' | 'registered' | (string & {});

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

/** `GET /{resource}/active`, and the activate / rollback / deactivate
 *  response. `source`, `activated_at` and `applied` are optional: the W2
 *  branch model doesn't carry them yet (W3-Q5, W4-Q5). */
export interface ActiveConfigResponse {
  axis: string;
  active: ActiveRef;
  source?: ConfigSource | null;
  activated_at?: string | null;
  previous: ActiveRef | null;
  config_revision: number;
  stale: boolean;
  applied?: AppliedRuntime[];
}

/** The fields every config doc (`GET /{resource}/{name}`) carries. */
export interface ConfigDocBase<B> {
  name: string;
  source: ConfigSource;
  read_only: boolean;
  revision: number | null;
  etag: string;
  description: string | null;
  body: B;
  created_at: string | null;
  updated_at: string | null;
  updated_by: string | null;
  cloned_from: string | null;
  active: boolean;
  active_revision?: number | null;
  validation: ValidationReport;
}

export interface ConfigRevision {
  revision: number;
  saved_at: string;
  cloned_from: string | null;
  description: string | null;
}

export interface ConfigRevisionList {
  name: string;
  revisions: ConfigRevision[];
}

export interface ConfigCloneRequest {
  new_name: string;
  revision: number | null;
  /** `"template"` disambiguates a template from a doc of the same name. */
  source: ConfigSource | null;
  description: string | null;
}

export interface ConfigUpdateRequest<B> {
  expected_revision: number;
  description: string | null;
  body: B;
}

export interface ConfigActivateRequest {
  revision: number | null;
  expected_active: ActiveRef;
  force: boolean;
}

export interface ConfigValidateRequest<B> {
  name: string | null;
  body: B;
}

/** `ConfigErrorDetail` (§7.1): every 4xx from a config route is
 *  `{detail: ConfigErrorDetail}`. Only the fields these editors read. */
export interface ConfigErrorDetail {
  error: string;
  message: string;
  current_revision?: number | null;
  report?: ValidationReport | null;
  current?: ActiveRef | null;
  axis?: string | null;
  valid_ids?: string[] | null;
}
