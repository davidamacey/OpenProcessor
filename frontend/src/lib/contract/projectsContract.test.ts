/**
 * `types_projects.ts` vs the vendored OpenAPI schemas (OpenProcessor
 * projects P1 + P3 lifecycle). Each key map is compile-time exact
 * against its interface (`satisfies Record<keyof T, true>` rejects a
 * missing or an extra key), and the test pins it to the schema's
 * property set, so a served rename fails here instead of rendering
 * blank. Not pinned, because the served OpenAPI leaves them untyped:
 * `capacity` (a bare `dict`) and the `DELETE /projects/{project}`
 * response (dry-run report / 202 envelope) — see `types_projects.ts`.
 */
import { describe, expect, it } from 'vitest';
import spec from '../../../contracts/openprocessor/openapi/curation.json';
import type * as T from '$lib/types_projects';

type Schema = { properties?: Record<string, unknown>; required?: string[] };
const schemas = (spec as { components: { schemas: Record<string, Schema> } }).components
  .schemas;

const keys = <K extends string>(o: Record<K, true>) => Object.keys(o).sort();

const SUMMARY = {
  slug: true,
  display_name: true,
  description: true,
  prefix: true,
  status: true,
  writable: true,
  selectable: true,
  is_default: true,
  deletable: true,
  archivable: true,
  unarchivable: true,
  revision: true,
  paused: true,
  created_at: true,
  updated_at: true,
  counts: true,
  origin: true,
} satisfies Record<keyof T.ProjectSummary, true>;

const CASES: [string, string[]][] = [
  ['ProjectSummary', keys(SUMMARY)],
  [
    'ProjectRecordResponse',
    keys({ ...SUMMARY, resources: true, error: true } satisfies Record<
      keyof T.ProjectRecordResponse,
      true
    >),
  ],
  [
    'ProjectCounts',
    keys({ images: true, items: true, validated: true } satisfies Record<
      keyof T.ProjectCounts,
      true
    >),
  ],
  [
    'ProjectsResponse',
    keys({
      default_slug: true,
      projects: true,
      capacity: true,
      limits: true,
      labels: true,
      include_archived: true,
    } satisfies Record<keyof T.ProjectsResponse, true>),
  ],
  [
    'ProjectLimits',
    keys({
      slug_pattern: true,
      slug_min: true,
      slug_max: true,
      reserved_slugs: true,
      retired_slugs: true,
      cloneable_axes: true,
    } satisfies Record<keyof T.ProjectLimits, true>),
  ],
  ['ProjectLabels', keys({ status: true } satisfies Record<keyof T.ProjectLabels, true>)],
  [
    'ProjectLifecycleResponse',
    keys({ project: true, warnings: true, keymap_clone_conflicts: true } satisfies Record<
      keyof T.ProjectLifecycleResponse,
      true
    >),
  ],
  [
    'KeymapCloneConflictWire',
    keys({
      action_id: true,
      combo: true,
      class_id: true,
      class_name: true,
    } satisfies Record<keyof T.KeymapCloneConflict, true>),
  ],
  [
    'ProjectWarning',
    keys({ code: true, message: true } satisfies Record<keyof T.ProjectWarning, true>),
  ],
  [
    'CreateProjectRequest',
    keys({
      slug: true,
      display_name: true,
      description: true,
      clone_settings_from: true,
      clone_axes: true,
    } satisfies Record<keyof T.CreateProjectRequest, true>),
  ],
  [
    'PatchProjectRequest',
    keys({
      display_name: true,
      description: true,
      expected_revision: true,
    } satisfies Record<keyof T.PatchProjectRequest, true>),
  ],
  [
    'ArchiveRequest',
    keys({ expected_revision: true } satisfies Record<keyof T.ArchiveRequest, true>),
  ],
  [
    'CloneSettingsRequest',
    keys({ from: true, axes: true, expected_revision: true } satisfies Record<
      keyof T.CloneSettingsRequest,
      true
    >),
  ],
  [
    'PipelinePauseState',
    keys({ project: true, paused: true, paused_by: true, reason: true } satisfies Record<
      keyof T.PipelinePauseState,
      true
    >),
  ],
];

describe('types_projects.ts matches the vendored OpenAPI', () => {
  it.each(CASES)('%s', (name, ours) => {
    const schema = schemas[name];
    expect(schema, `${name} missing from the vendored OpenAPI`).toBeDefined();
    expect(ours).toEqual(Object.keys(schema!.properties ?? {}).sort());
  });

  it('every request field the server requires is required in our type', () => {
    // Compile-time: these objects omit every optional field.
    const create: T.CreateProjectRequest = { slug: 's', display_name: 'd' };
    const patch: T.PatchProjectRequest = { expected_revision: 1 };
    const clone: T.CloneSettingsRequest = { from: 'a', expected_revision: 1 };
    expect([create, patch, clone]).toHaveLength(3);
    expect(schemas.CreateProjectRequest!.required).toEqual(['slug', 'display_name']);
    expect(schemas.PatchProjectRequest!.required).toEqual(['expected_revision']);
    expect(schemas.CloneSettingsRequest!.required?.sort()).toEqual([
      'expected_revision',
      'from',
    ]);
  });
});
