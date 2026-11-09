/**
 * `types_embeddingSpaces.ts` and the search `space` / `fuse` wire vs the
 * generated contract.
 */
import { describe, expect, it } from 'vitest';
import spec from '$contracts/openapi/curation.json';
import type * as T from '$lib/types_embeddingSpaces';
import {
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
