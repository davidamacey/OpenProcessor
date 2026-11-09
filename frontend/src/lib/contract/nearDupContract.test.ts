/**
 * `types_nearDup.ts` and the item/freeze wire vs the generated contract.
 */
import { describe, expect, it } from 'vitest';
import spec from '$contracts/openapi/curation.json';
import type * as T from '$lib/types_nearDup';
import type { TestHoldoutFreezeResult } from '$lib/types_export';
import {
  NEAR_DUP_LEVELS,
  NEAR_DUP_RUN_MODES,
  NEAR_DUP_STALE_REASONS,
} from '$lib/types_nearDup';

type Schema = {
  properties?: Record<string, { enum?: string[]; items?: { enum?: string[] } }>;
};
const schemas = (spec as unknown as { components: { schemas: Record<string, Schema> } })
  .components.schemas;
const keys = <K extends string>(o: Record<K, true>) => Object.keys(o).sort();
const served = (name: string) => Object.keys(schemas[name]!.properties ?? {}).sort();

describe('near-duplicate contract', () => {
  it('NearDupRunView, NearDupLevelStatus, NearDupStatus have the declared keys', () => {
    expect(served('NearDupRunView')).toEqual(
      keys<keyof T.NearDupRunView>({
        id: true,
        level: true,
        mode: true,
        status: true,
        space: true,
        algorithm_version: true,
        profile_revision: true,
        params: true,
        members_scored: true,
        dedup_groups: true,
        leak_groups: true,
        grouped_members: true,
        largest_group: true,
        oversized_split: true,
        holdout_absorbed: true,
        started_at: true,
        finished_at: true,
      }),
    );
    expect(served('NearDupLevelStatus')).toEqual(
      keys<keyof T.NearDupLevelStatus>({
        level: true,
        current_run: true,
        pending_count: true,
        stale_reasons: true,
      }),
    );
    expect(served('NearDupStatus')).toEqual(
      keys<keyof T.NearDupStatus>({ profile: true, levels: true }),
    );
    expect(served('NearDupRunStarted')).toEqual(['job_id', 'kind']);
  });

  it('the level, mode and stale-reason ids are the served enums', () => {
    const lv = schemas.NearDupLevelStatus!.properties!;
    expect(lv.level!.enum).toEqual([...NEAR_DUP_LEVELS]);
    expect(lv.stale_reasons!.items!.enum).toEqual([...NEAR_DUP_STALE_REASONS]);
    expect(schemas.NearDupRunRequest!.properties!.mode!.enum).toEqual([
      ...NEAR_DUP_RUN_MODES,
    ]);
  });

  it('the freeze result carries exactly the served keys, including moved_by_near_dup', () => {
    expect(served('TestHoldoutFreezeResponse')).toEqual(
      keys<keyof TestHoldoutFreezeResult>({
        n_frozen: true,
        n_classes_covered: true,
        test_holdout_sha: true,
        per_class_counts: true,
        selection: true,
        percent: true,
        min_per_class: true,
        moved_by_near_dup: true,
      }),
    );
  });

  it('an item serves the three near_dup_* fields', () => {
    expect(served('ItemDoc')).toEqual(
      expect.arrayContaining([
        'near_dup_group_id',
        'near_dup_group_size',
        'near_dup_is_representative',
      ]),
    );
  });
});
