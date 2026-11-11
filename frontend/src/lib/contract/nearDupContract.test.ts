/**
 * `types_nearDup.ts` and the item/freeze wire vs the generated contract.
 */
import { describe, expect, it } from 'vitest';
import spec from '$contracts/openapi/curation.json';
import type * as T from '$lib/types_nearDup';
import type { TestHoldoutFreezeResult } from '$lib/types_export';
import {
  NEAR_DUP_DECISIONS,
  NEAR_DUP_LEVELS,
  NEAR_DUP_PURPOSES,
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

  it('the group, member, decision and profile types have the served keys', () => {
    expect(served('NearDupGroupSummary')).toEqual(
      keys<keyof T.NearDupGroupSummary>({
        group_id: true,
        purpose: true,
        size: true,
        representative_id: true,
        min_edge_cosine: true,
        min_cosine_to_rep: true,
        has_holdout: true,
      }),
    );
    expect(served('NearDupGroupPage')).toEqual(['items', 'next_cursor']);
    expect(served('NearDupGroupDetail')).toEqual(['group', 'members']);
    expect(served('NearDupMemberView')).toEqual(
      keys<keyof T.NearDupMemberView>({
        member_id: true,
        is_representative: true,
        cosine_to_rep: true,
      }),
    );
    expect(served('NearDupDecisionIn')).toEqual(
      keys<keyof T.NearDupDecisionIn>({
        level: true,
        a_id: true,
        b_id: true,
        decision: true,
      }),
    );
    expect(served('NearDupDecisionView')).toEqual(
      keys<keyof T.NearDupDecisionView>({
        id: true,
        level: true,
        a_id: true,
        b_id: true,
        decision: true,
        decided_by: true,
        decided_at: true,
      }),
    );
    expect(served('ResearchConfigDoc')).toEqual(
      keys<keyof T.NearDupProfileDoc>({
        kind: true,
        name: true,
        revision: true,
        description: true,
        body: true,
        created_at: true,
        updated_at: true,
        updated_by: true,
      }),
    );
    expect(served('ResearchActive')).toEqual(
      keys<keyof T.NearDupActiveProfile>({
        axis: true,
        state: true,
        name: true,
        revision: true,
        activated_at: true,
      }),
    );
  });

  it('the purpose and decision ids are the served enums', () => {
    expect(schemas.NearDupGroupSummary!.properties!.purpose!.enum).toEqual([
      ...NEAR_DUP_PURPOSES,
    ]);
    expect(schemas.NearDupDecisionIn!.properties!.decision!.enum).toEqual([
      ...NEAR_DUP_DECISIONS,
    ]);
  });

  it('every route the page calls is in the contract', () => {
    const paths = Object.keys(
      (spec as unknown as { paths: Record<string, unknown> }).paths,
    );
    const base = '/curation/projects/{project}';
    for (const p of [
      'near_duplicates/status',
      'near_duplicates/runs',
      'near_duplicates/groups',
      'near_duplicates/groups/{group_id}',
      'near_duplicates/decisions',
      'near_duplicates/decisions/{decision_id}',
      'near_dup_profiles/active',
      'near_dup_profiles/{name}',
      'near_dup_profiles/{name}/activate',
    ]) {
      expect(paths, p).toContain(`${base}/${p}`);
    }
  });
});
