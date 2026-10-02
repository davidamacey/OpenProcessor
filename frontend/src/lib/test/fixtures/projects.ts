/**
 * Served-shape project fixtures (`GET {globalApi()}/projects`, P1 + P3
 * lifecycle). Every non-slug value is derived from the slug or explicit,
 * so a test can tell two projects apart by any field.
 */
import { API_PREFIX } from '$lib/api';
import type {
  ProjectCapacity,
  ProjectLimits,
  ProjectsResponse,
  ProjectSummary,
} from '$lib/types_projects';

export function testProject(
  over: Partial<ProjectSummary> & { slug: string },
): ProjectSummary {
  const slug = over.slug;
  const status = over.status ?? 'active';
  return {
    display_name: `Project ${slug}`,
    description: '',
    prefix: `${API_PREFIX}/projects/${slug}`,
    status: 'active',
    writable: true,
    selectable: true,
    is_default: false,
    deletable: true,
    // Mirrors the server's own rule (ARCHIVABLE/UNARCHIVABLE_STATUSES);
    // a test overrides either to prove the UI reads the flag, not status.
    archivable: status === 'active',
    unarchivable: status === 'archived',
    revision: 3,
    paused: false,
    created_at: '2026-09-26T12:00:00Z',
    updated_at: '2026-09-26T12:00:00Z',
    counts: { images: 10, items: 20, validated: 5 },
    origin: null,
    ...over,
  };
}

export const TEST_LIMITS: ProjectLimits = {
  slug_pattern: '^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$',
  slug_min: 2,
  slug_max: 32,
  reserved_slugs: [
    'all',
    'combine',
    'global',
    'health',
    'new',
    'none',
    'projects',
    'settings',
    'vlm',
  ],
  retired_slugs: ['gone'],
  cloneable_axes: ['settings_defaults', 'classes'],
};

export function testCapacity(status: ProjectCapacity['status'] = 'ok'): ProjectCapacity {
  return {
    status,
    active_shards: status === 'blocked' ? 1000 : 12,
    per_project_shards: 6,
    soft_limit: 40,
    hard_limit: 1000,
    heap_max_bytes: 2147483648,
    max_shards_per_node: 1000,
    data_nodes: 1,
    projects_until_soft_limit: status === 'ok' ? 4 : 0,
    message: `capacity is ${status}: served message`,
    labels: {
      ok: 'Room for more projects',
      warn: 'Near the recommended shard budget',
      blocked: 'No room for another project',
    },
  };
}

export function testProjectsResponse(
  projects: ProjectSummary[],
  over: Partial<ProjectsResponse> = {},
): ProjectsResponse {
  return {
    default_slug: 'default',
    projects,
    capacity: testCapacity('ok'),
    limits: TEST_LIMITS,
    labels: {
      status: {
        active: 'Active',
        archived: 'Archived',
        building: 'Building',
        failed: 'Failed',
        deleting: 'Deleting',
        deleted: 'Deleted',
      },
    },
    include_archived: false,
    ...over,
  };
}
