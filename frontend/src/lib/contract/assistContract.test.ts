/**
 * `types_assist.ts` vs the vendored OpenAPI schemas (the selectable assist
 * run wire). Each key map is compile-time exact against its interface
 * (`satisfies Record<keyof T, true>` rejects a missing or an extra key) and
 * is pinned to the schema's property set, so a served rename fails here
 * instead of rendering blanks. The enums the UI offers are the served ones.
 */
import { describe, expect, it } from 'vitest';
import spec from '$contracts/openapi/curation.json';
import type * as T from '$lib/types_assist';
import { ASSIST_ACTIONS, ASSIST_STATUSES } from '$lib/types_assist';

type Schema = {
  properties?: Record<string, { enum?: string[]; items?: { enum?: string[] } }>;
  required?: string[];
};
const schemas = (spec as { components: { schemas: Record<string, Schema> } }).components
  .schemas;

const keys = <K extends string>(o: Record<K, true>) => Object.keys(o).sort();

const CASES: [string, string[]][] = [
  [
    'AssistRunRequest',
    keys({
      target: true,
      sample: true,
      actions: true,
      vlm: true,
      prompt_pack: true,
      acknowledge_external: true,
      segment: true,
      min_agreement: true,
      dry_run: true,
    } satisfies Record<keyof T.AssistRunRequest, true>),
  ],
  [
    'AssistSample',
    keys({ mode: true, size: true, seed: true, stratify_by: true } satisfies Record<
      keyof T.AssistSample,
      true
    >),
  ],
  [
    'AssistContext',
    keys({ cluster_id: true, cluster_subid: true } satisfies Record<
      keyof T.AssistContext,
      true
    >),
  ],
  [
    'SegmentSpec',
    keys({ verify: true } satisfies Record<keyof T.AssistSegmentSpec, true>),
  ],
  [
    'CropsTarget',
    keys({ kind: true, crop_ids: true, context: true } satisfies Record<
      keyof T.AssistCropsTarget,
      true
    >),
  ],
  [
    'BoxesTarget',
    keys({ kind: true, boxes: true } satisfies Record<keyof T.AssistBoxesTarget, true>),
  ],
  [
    'BoxRef',
    keys({ crop_id: true, box_id: true } satisfies Record<keyof T.BoxRef, true>),
  ],
  [
    'ClusterTarget',
    keys({ kind: true, cluster_id: true, cluster_subid: true } satisfies Record<
      keyof T.AssistClusterTarget,
      true
    >),
  ],
  [
    'FilterTarget',
    keys({ kind: true, filter: true } satisfies Record<keyof T.AssistFilterTarget, true>),
  ],
  [
    'AssistRunState',
    keys({
      run_id: true,
      status: true,
      created_at: true,
      finished_at: true,
      error: true,
      request: true,
      resolved: true,
      progress: true,
      result: true,
      scope_now: true,
    } satisfies Record<keyof T.AssistRunState, true>),
  ],
  [
    'ResolvedScope',
    keys({
      scope_count: true,
      scope_digest: true,
      sampled: true,
      sample_seed: true,
      strata: true,
      skipped: true,
      vlm: true,
      prompt_pack: true,
      estimated_vlm_calls: true,
    } satisfies Record<keyof T.AssistResolved, true>),
  ],
  [
    'SkippedCounts',
    keys({
      holdout: true,
      excluded: true,
      class_locked: true,
      box_locked: true,
      box_not_found: true,
      not_found: true,
    } satisfies Record<keyof T.AssistSkipped, true>),
  ],
  [
    'StratumInfo',
    keys({ key: true, size: true, sampled: true } satisfies Record<
      keyof T.AssistStratum,
      true
    >),
  ],
  [
    'AssistProgress',
    keys({ processed: true, total: true } satisfies Record<keyof T.AssistProgress, true>),
  ],
  [
    'AssistResult',
    keys({
      classify: true,
      attributes: true,
      verify: true,
      segment: true,
      proposal: true,
    } satisfies Record<keyof T.AssistResult, true>),
  ],
  [
    'ClassifyResult',
    keys({
      predicted: true,
      updated: true,
      failed: true,
      unreadable: true,
      locked_suggest_only: true,
    } satisfies Record<keyof T.AssistClassifyResult, true>),
  ],
  [
    'AttributesResult',
    keys({ items_written: true, invalid_values: true } satisfies Record<
      keyof T.AssistAttributesResult,
      true
    >),
  ],
  [
    'VerifyResult',
    keys({
      boxes: true,
      accepted: true,
      rejected: true,
      locked_skipped: true,
    } satisfies Record<keyof T.AssistVerifyResult, true>),
  ],
  [
    'SegmentResult',
    keys({ queued: true, done: true, pending: true, failed: true } satisfies Record<
      keyof T.AssistSegmentResult,
      true
    >),
  ],
  [
    'Proposal',
    keys({
      class_name: true,
      in_registry: true,
      agreement: true,
      answered: true,
      min_agreement: true,
      passes_gate: true,
      distribution: true,
      new_class_proposals: true,
      attribute_summary: true,
    } satisfies Record<keyof T.AssistProposal, true>),
  ],
  [
    'ClassVotes',
    keys({ class_name: true, votes: true } satisfies Record<keyof T.ClassVotes, true>),
  ],
  [
    'NewClassVotes',
    keys({ name: true, votes: true } satisfies Record<keyof T.NewClassVotes, true>),
  ],
  [
    'AttributeSummary',
    keys({ key: true, values: true, answered: true } satisfies Record<
      keyof T.AttributeSummary,
      true
    >),
  ],
  [
    'AttributeValueCount',
    keys({ value: true, count: true } satisfies Record<
      keyof T.AttributeValueCount,
      true
    >),
  ],
  [
    'ScopeNow',
    keys({ count: true, overlap: true, jaccard: true, changed: true } satisfies Record<
      keyof T.AssistScopeNow,
      true
    >),
  ],
  [
    'AssistApplyRequest',
    keys({
      class_name: true,
      label_source: true,
      allow_changed_scope: true,
      dry_run: true,
    } satisfies Record<keyof T.AssistApplyRequest, true>),
  ],
  [
    'AssistApplyResponse',
    keys({
      dry_run: true,
      class_name: true,
      selected: true,
      skipped_locked: true,
      updated: true,
      updated_ids: true,
      conflicts: true,
      scope_now: true,
    } satisfies Record<keyof T.AssistApplyResponse, true>),
  ],
  [
    'ApplyConflict',
    keys({ crop_id: true, current_source: true } satisfies Record<
      keyof T.AssistApplyConflict,
      true
    >),
  ],
  ['AssistRunList', keys({ runs: true } satisfies Record<keyof T.AssistRunList, true>)],
  [
    'ItemAssistWire',
    keys({
      run_id: true,
      run_created_at: true,
      vlm_endpoint: true,
      vlm_model: true,
      prompt_pack: true,
      prompt_pack_revision: true,
      class_name: true,
      class_confidence: true,
      attributes: true,
      boxes: true,
    } satisfies Record<keyof T.ItemAssist, true>),
  ],
  [
    'AssistAttributeWire',
    keys({ key: true, value: true, confidence: true } satisfies Record<
      keyof T.AssistAttribute,
      true
    >),
  ],
  [
    'AssistBoxWire',
    keys({ box_id: true, attributes: true } satisfies Record<
      keyof T.AssistBoxAttributes,
      true
    >),
  ],
];

describe('assist types vs the generated contract', () => {
  for (const [schema, expected] of CASES) {
    it(`${schema} has exactly the keys the frontend type declares`, () => {
      const served = schemas[schema];
      expect(served, `${schema} missing from the contract`).toBeDefined();
      expect(Object.keys(served!.properties ?? {}).sort()).toEqual(expected);
    });
  }

  it('the actions and statuses the UI knows are the served enums', () => {
    expect([...ASSIST_ACTIONS].sort()).toEqual(
      [...schemas.AssistRunRequest!.properties!.actions!.items!.enum!].sort(),
    );
    expect([...ASSIST_STATUSES].sort()).toEqual(
      [...schemas.AssistRunState!.properties!.status!.enum!].sort(),
    );
  });
});
