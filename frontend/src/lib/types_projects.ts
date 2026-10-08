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
import type { ModelSharingUser } from '$lib/types_models';

export interface ProjectCounts {
  images: number;
  items: number;
  /** `null` when the backend hasn't computed it (served, not faked as 0). */
  validated: number | null;
  /** v0.4.0: items with a vector; `null` when not computed. */
  items_embedded: number | null;
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
  /** Served: whether `POST /projects/{slug}/archive` accepts this
   *  project's status. The Archive action renders only when true. */
  archivable: boolean;
  /** Served: whether `POST /projects/{slug}/unarchive` accepts this
   *  project's status. The Unarchive action renders only when true. */
  unarchivable: boolean;
  /** Optimistic-concurrency token every lifecycle write sends back as
   *  `expected_revision`. */
  revision: number;
  /** Served: this project's own pipeline-pause flag (`POST {prefix}/pause`).
   *  Not the global GPU-training claim — `GET {prefix}/pause` reports that
   *  too, as `paused_by`. */
  paused: boolean;
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
  /** `active_shards + per_project_shards`: the total if one more project
   *  were created. */
  shards_after_create: number;
  /** Which limit `soft_limit` is: the heap-derived one, or the cluster's
   *  own hard limit when that is lower. */
  limit_source: 'heap' | 'cluster_max_shards_per_node';
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
  /** v0.4.0: served facts behind the warning (untyped object). */
  detail?: Record<string, unknown>;
}

/** One action the `keymap` clone axis dropped from the copy because its
 *  combo collides with the target's class hotkey — a report, never a
 *  silent unbind. */
export interface KeymapCloneConflict {
  action_id: string;
  combo: string;
  class_id: number;
  class_name: string;
}

/** Every lifecycle mutation (create 201, PATCH, archive, unarchive,
 *  clone_settings, and a real delete's 202) answers this envelope. */
export interface ProjectLifecycleResponse {
  project: ProjectSummary;
  warnings?: ProjectWarning[];
  keymap_clone_conflicts?: KeymapCloneConflict[];
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

/** One project whose active detection profile uses a model the deleted
 *  project owns and shares (`referenced_by`). The served schema types the
 *  rows as bare objects; `project` and `profile` are the keys served. */
export interface DeleteReference {
  project: string;
  profile?: string | null;
}

/** `DELETE {globalApi()}/projects/{slug}?dry_run=true` — report only,
 *  writes nothing. */
export interface DeleteDryRunResponse {
  indexes: { name: string; docs: number; store_bytes?: number | null }[];
  dirs: { path: string; bytes: number }[];
  promoted_models: string[];
  mlflow_experiment: string;
  running_jobs: Record<string, unknown>[];
  referenced_by: DeleteReference[];
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
  /** 409 `in_use` on a model sharing write: each other project with the
   *  profile that uses the model. */
  used_by?: ModelSharingUser[] | null;
}

/** `GET|POST {prefix}/pause`, `POST {prefix}/resume` (projects P2,
 *  `projects_plan.md` §5.1): `paused` is the project's own flag OR the
 *  global GPU-training claim holding its workers; `paused_by` names which
 *  (`'project'`, `'gpu_training'`, both, or none) and `reason` is the
 *  served sentence for a claim-only pause (null otherwise). */
export interface PipelinePauseState {
  project: string;
  paused: boolean;
  paused_by: string[];
  reason: string | null;
}
