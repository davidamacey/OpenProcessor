/**
 * Wire types for the projects surface
 * (`docs/design/any-domain-rev3-and-projects-contract-review-2026-09-26.md`
 * §7; OpenProcessor `cutover/projects-foundation` P1 +
 * `cutover/projects-lifecycle` P3). `GET {globalApi()}/projects` is the
 * switcher vocabulary and the only global read the app needs before
 * anything scoped can fire — every scoped URL is built from a project's
 * own served `prefix`, never assembled client-side.
 *
 * Pinned key-for-key to the vendored OpenAPI by
 * `src/lib/contract/projectsContract.test.ts`.
 */

export interface ProjectCounts {
  images: number;
  items: number;
  /** `null` when the backend hasn't computed it (served, not faked as 0). */
  validated: number | null;
}

export interface ProjectOrigin {
  kind: string;
  job_id?: string;
  sources?: string[];
}

/** One entry of `GET {globalApi()}/projects`'s `projects` list. */
export interface ProjectSummary {
  slug: string;
  display_name: string;
  description: string;
  /** The scoped path prefix every backend call for this project is
   *  built from, e.g. `/curation/projects/default`. Never assembled
   *  client-side — always this served value. */
  prefix: string;
  status: string;
  /** Served: whether writes are accepted (false for everything but
   *  `active`). */
  writable: boolean;
  /** Served: whether `{prefix}/...` reads work, i.e. whether the
   *  project can be opened under `/p/<slug>`. */
  selectable: boolean;
  is_default: boolean;
  /** Served: whether a delete can ever be attempted (false for the
   *  default project). Delete is absent when false. */
  deletable: boolean;
  /** Optimistic-concurrency token every lifecycle write sends back as
   *  `expected_revision`. */
  revision: number;
  created_at: string;
  updated_at: string;
  counts: ProjectCounts;
  origin?: ProjectOrigin | null;
}

export interface ProjectCapacity {
  status: 'ok' | 'warn' | 'blocked';
  active_shards: number;
  per_project_shards: number;
  soft_limit: number;
  hard_limit: number;
  heap_max_bytes: number;
  max_shards_per_node: number;
  data_nodes: number;
  projects_until_soft_limit: number;
  message: string;
  labels: Record<string, string>;
}

export interface ProjectLimits {
  slug_pattern: string;
  slug_min: number;
  slug_max: number;
  reserved_slugs: string[];
  /** Slugs of deleted projects, retired forever (create answers 409
   *  `slug_retired`). */
  retired_slugs: string[];
  cloneable_axes: string[];
}

export interface ProjectLabels {
  /** Display copy for every served project `status`. */
  status: Record<string, string>;
}

export interface ProjectsResponse {
  default_slug: string;
  projects: ProjectSummary[];
  /** `null` only when OpenSearch is unreachable — create stays enabled
   *  and the server decides. */
  capacity: ProjectCapacity | null;
  limits: ProjectLimits;
  labels?: ProjectLabels;
  include_archived: boolean;
}

export interface ProjectWarning {
  code: string;
  message: string;
}

/** Every lifecycle mutation (create 201, PATCH, archive, unarchive,
 *  clone_settings, and a real delete's 202) answers this envelope. */
export interface ProjectLifecycleResponse {
  project: ProjectSummary;
  warnings?: ProjectWarning[];
}

export interface ProjectError {
  code: string;
  message: string;
}

/** `GET {globalApi()}/projects/{slug}` — a deep link to a slug the
 *  default list doesn't carry (e.g. an archived project). */
export interface ProjectRecordResponse extends ProjectSummary {
  resources: Record<string, unknown>;
  error?: ProjectError | null;
}

export interface CreateProjectRequest {
  slug: string;
  display_name: string;
  description?: string;
  clone_settings_from?: string | null;
  clone_axes?: string[] | null;
}

export interface PatchProjectRequest {
  display_name?: string | null;
  description?: string | null;
  expected_revision: number;
}

export interface ArchiveRequest {
  expected_revision: number;
}

export interface CloneSettingsRequest {
  from: string;
  axes?: string[] | null;
  expected_revision: number;
}

export interface DeleteBlockingIssue {
  code: string;
  message: string;
}

/** `DELETE {globalApi()}/projects/{slug}?dry_run=true` — report only,
 *  writes nothing. Not declared as a response model in the served
 *  OpenAPI (the route's response is untyped there); shape from the
 *  backend's `DeleteDryRunResponse`. */
export interface DeleteDryRunResponse {
  indexes: { name: string; docs: number; store_bytes?: number | null }[];
  dirs: { path: string; bytes: number }[];
  promoted_models: string[];
  mlflow_experiment: string;
  running_jobs: Record<string, unknown>[];
  referenced_by: Record<string, unknown>[];
  blocking: string[];
  blocking_detail?: DeleteBlockingIssue[];
}

/** The `detail` body of every project-route error
 *  (`ConfigErrorDetail`): `{error, message, ...code-specific fields}`. */
export interface ProjectErrorDetail {
  error: string;
  message: string;
  project?: string | null;
  jobs?: string[] | null;
  projects?: string[] | null;
  active_shards?: number | null;
  needed?: number | null;
  soft_limit?: number | null;
  hard_limit?: number | null;
  heap_max_bytes?: number | null;
  current_revision?: number | null;
}
