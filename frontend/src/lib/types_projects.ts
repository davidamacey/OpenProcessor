/**
 * Wire types for the P1 project-scoping cutover
 * (`docs/design/any-domain-rev3-and-projects-contract-review-2026-09-26.md`,
 * OpenProcessor `cutover/projects-foundation`). `GET {globalApi()}/projects`
 * is the switcher vocabulary and the only global read the app needs
 * before anything scoped can fire — every scoped URL is built from a
 * project's own served `prefix`, never assembled client-side.
 */

export interface ProjectCounts {
  images: number;
  items: number;
  validated: number;
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
  writable: boolean;
  selectable: boolean;
  is_default: boolean;
  deletable: boolean;
  revision: number;
  created_at: string;
  updated_at: string;
  counts: ProjectCounts;
  origin: ProjectOrigin | null;
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
  cloneable_axes: string[];
}

export interface ProjectsResponse {
  default_slug: string;
  projects: ProjectSummary[];
  capacity: ProjectCapacity | null;
  limits: ProjectLimits;
  include_archived: boolean;
}
