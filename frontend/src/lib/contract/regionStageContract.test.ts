/**
 * `RegionStageState` / `RegionStageCounts` vs the vendored OpenAPI
 * (OpenProcessor fce17771), key for key, and which keys are required.
 */
import { describe, expect, it } from 'vitest';
import spec from '$contracts/openapi/curation.json';
import type * as T from '$lib/types_openVocab';

type Schema = { properties?: Record<string, unknown>; required?: string[] };
const schemas = (spec as unknown as { components: { schemas: Record<string, Schema> } })
  .components.schemas;

const keys = <K extends string>(o: Record<K, true>) => Object.keys(o).sort();
type ReqKeys<X> = { [K in keyof X]-?: object extends Pick<X, K> ? never : K }[keyof X];
const required = <X>(o: Record<ReqKeys<X>, true>) => Object.keys(o).sort();

describe('region stage types vs the vendored contract', () => {
  it('RegionStageState has exactly the served properties and requireds', () => {
    const s = schemas.RegionStageState!;
    expect(
      keys({
        project: true,
        paused: true,
        paused_since: true,
        pipeline_paused: true,
        counts: true,
        rerun_skipped: true,
      } satisfies Record<keyof T.RegionStageState, true>),
    ).toEqual(Object.keys(s.properties!).sort());
    expect(
      required<T.RegionStageState>({
        project: true,
        paused: true,
        pipeline_paused: true,
        counts: true,
        rerun_skipped: true,
      }),
    ).toEqual([...s.required!].sort());
  });

  it('RegionStageCounts has exactly the served properties and requireds', () => {
    const s = schemas.RegionStageCounts!;
    const declared = keys({
      pending_detection: true,
      pending_verification: true,
      gate_skipped: true,
    } satisfies Record<keyof T.RegionStageCounts, true>);
    expect(declared).toEqual(Object.keys(s.properties!).sort());
    expect(declared).toEqual([...s.required!].sort());
  });
});
