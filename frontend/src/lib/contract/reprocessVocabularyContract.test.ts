/**
 * `ReprocessVocabulary` / `VocabEntry` (types_profiles.ts) and the
 * `reprocess` block of `ConfigVocabularyResponse` vs the vendored OpenAPI.
 */
import { describe, expect, it } from 'vitest';
import spec from '$contracts/openapi/curation.json';
import type {
  ConfigVocabulary,
  ReprocessVocabulary,
  VocabEntry,
} from '$lib/types_profiles';

type Schema = { properties?: Record<string, unknown>; required?: string[] };
const schemas = (spec as unknown as { components: { schemas: Record<string, Schema> } })
  .components.schemas;
const keys = <K extends string>(o: Record<K, true>) => Object.keys(o).sort();
const declared = (n: string) => Object.keys(schemas[n]!.properties!).sort();

describe('reprocess vocabulary types match the vendored OpenAPI', () => {
  it('VocabEntry', () => {
    expect(
      keys({ id: true, label: true, description: true } satisfies Record<
        keyof VocabEntry,
        true
      >),
    ).toEqual(declared('VocabEntry'));
  });
  it('ReprocessVocabulary', () => {
    expect(
      keys({
        scopes: true,
        filter_fields: true,
        job_statuses: true,
        lock_reasons: true,
      } satisfies Record<keyof ReprocessVocabulary, true>),
    ).toEqual(declared('ReprocessVocabulary'));
  });
  it('ConfigVocabularyResponse serves a required `reprocess` block', () => {
    const k: keyof ConfigVocabulary = 'reprocess';
    expect(schemas.ConfigVocabularyResponse!.required).toContain(k);
    expect(declared('ConfigVocabularyResponse')).toContain(k);
  });
});
