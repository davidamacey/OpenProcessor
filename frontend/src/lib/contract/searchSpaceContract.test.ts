/**
 * `types_embeddingSpaces.ts` and the search `space` / `fuse` wire vs the
 * generated contract.
 */
import { describe, expect, it } from 'vitest';
import spec from '$contracts/openapi/curation.json';
import type * as T from '$lib/types_embeddingSpaces';
import {
  EMBEDDING_BUILD_MODES,
  EMBEDDING_SPACE_ROLES,
  EMBEDDING_SPACE_STATUSES,
  EMBEDDING_TARGETS,
} from '$lib/types_embeddingSpaces';

type Schema = {
  properties?: Record<string, { enum?: string[] }>;
  enum?: string[];
};
const doc = spec as unknown as {
  components: { schemas: Record<string, Schema> };
  paths: Record<string, Record<string, { parameters?: { name: string }[] }>>;
};
const schemas = doc.components.schemas;
const keys = <K extends string>(o: Record<K, true>) => Object.keys(o).sort();

describe('embedding space contract', () => {
  it('EmbeddingSpaceView has exactly the keys the frontend type declares', () => {
    expect(Object.keys(schemas.EmbeddingSpaceView!.properties ?? {}).sort()).toEqual(
      keys<keyof T.EmbeddingSpaceView>({
        space_key: true,
        target: true,
        model_key: true,
        model_version: true,
        dim: true,
        role: true,
        status: true,
        ready_at: true,
        docs_with_vector: true,
        run_id: true,
        run_mode: true,
        run_status: true,
      }),
    );
  });

  it('the role, status and target ids are the served enums', () => {
    const p = schemas.EmbeddingSpaceView!.properties!;
    expect(p.role!.enum).toEqual([...EMBEDDING_SPACE_ROLES]);
    expect(p.status!.enum).toEqual([...EMBEDDING_SPACE_STATUSES]);
    expect(schemas.Target!.enum).toEqual([...EMBEDDING_TARGETS]);
  });
});

describe('embedding space management wire', () => {
  const CASES: [string, string[]][] = [
    [
      'EmbeddingModelEntry',
      keys<keyof T.EmbeddingModelEntry>({
        model_key: true,
        version: true,
        modalities: true,
        native_dim: true,
        matryoshka_dims: true,
        licence: true,
        available: true,
        unavailable_reason: true,
      }),
    ],
    [
      'EmbeddingSpaceRegisterRequest',
      keys<keyof T.EmbeddingSpaceRegisterRequest>({
        model_key: true,
        model_version: true,
        target: true,
        dim: true,
      }),
    ],
    [
      'EmbeddingBuildRequest',
      keys<keyof T.EmbeddingBuildRequest>({ mode: true, batch_size: true }),
    ],
    ['EmbeddingPromoteRequest', keys<keyof T.EmbeddingPromoteRequest>({ force: true })],
    [
      'EmbeddingSpaceStartResponse',
      keys<keyof T.EmbeddingSpaceStartResponse>({
        space_key: true,
        target: true,
        job_id: true,
        run_id: true,
        bytes_estimate: true,
      }),
    ],
    [
      'EmbeddingPromoteResponse',
      keys<keyof T.EmbeddingPromoteResponse>({
        primary: true,
        previous: true,
        config_epoch: true,
      }),
    ],
    [
      'EmbeddingReindexRunResponse',
      keys<keyof T.EmbeddingReindexRun>({
        id: true,
        space_key: true,
        mode: true,
        scope: true,
        docs_total: true,
        docs_done: true,
        docs_failed: true,
        docs_skipped: true,
        bytes_estimate: true,
        started_at: true,
        finished_at: true,
        job_id: true,
        job_status: true,
        job_progress: true,
      }),
    ],
    [
      'EmbeddingRunCancelResponse',
      keys<keyof T.EmbeddingRunCancelResponse>({
        run_id: true,
        job_status: true,
        abandoned: true,
        space_status: true,
        clear_job_id: true,
      }),
    ],
  ];

  it.each(CASES)(
    '%s has exactly the keys the frontend type declares',
    (name, expected) => {
      expect(Object.keys(schemas[name]!.properties ?? {}).sort()).toEqual(expected);
    },
  );

  it('the build modes are the served enum', () => {
    expect(schemas.EmbeddingBuildRequest!.properties!.mode!.enum).toEqual([
      ...EMBEDDING_BUILD_MODES,
    ]);
  });
});

describe('search space and fuse wire', () => {
  for (const route of ['text', 'similar', 'image']) {
    it(`/search/${route} takes space and fuse and echoes both`, () => {
      const path = `/curation/projects/{project}/search/${route}`;
      const op = doc.paths[path]![route === 'image' ? 'post' : 'get']!;
      const names = (op.parameters ?? []).map((p) => p.name);
      expect(names).toContain('space');
      expect(names).toContain('fuse');
      const resp = schemas[`Search${route[0]!.toUpperCase()}${route.slice(1)}Response`]!;
      expect(Object.keys(resp.properties ?? {})).toEqual(
        expect.arrayContaining(['space', 'fuse']),
      );
    });
  }

  it('an item names the space its score was measured in', () => {
    expect(Object.keys(schemas.SearchItemDoc!.properties ?? {})).toContain('score_space');
  });
});
