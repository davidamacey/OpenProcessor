/**
 * `types_import.ts` and the Reprocess vocabulary vs the vendored OpenAPI
 * (OpenProcessor f582aa05, W10). Each key map is compile-time exact
 * against its interface (`satisfies Record<keyof T, true>` rejects a
 * missing or an extra key) and is pinned here to the schema's property
 * set, so a served rename fails here instead of rendering blank. The
 * Reprocess scope and region-mode arrays are pinned to the request
 * enums, since the backend serves no vocabulary for them.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import spec from '$contracts/openapi/curation.json';
import { reprocessBatch, reprocessCrop, reprocessImage } from '$lib/api';
import { REGION_MODES, REPROCESS_SCOPES } from '$lib/datasets/reprocessVocabulary';
import { ReprocessFlow } from '$lib/datasets/reprocessController.svelte';
import type * as T from '$lib/types_import';

type Schema = {
  properties?: Record<string, { enum?: string[]; items?: { enum?: string[] } }>;
  additionalProperties?: boolean;
};
const schemas = (spec as unknown as { components: { schemas: Record<string, Schema> } })
  .components.schemas;

const keys = <K extends string>(o: Record<K, true>) => Object.keys(o).sort();
const declared = (name: string): string[] => {
  const s = schemas[name];
  if (!s?.properties) throw new Error(`schema not found: ${name}`);
  return Object.keys(s.properties).sort();
};

const CASES: [string, string[]][] = [
  [
    'LabeledChoice',
    keys({ value: true, label: true, description: true } satisfies Record<
      keyof T.LabeledChoice,
      true
    >),
  ],
  [
    'DatasetFormatInfo',
    keys({ format: true, label: true } satisfies Record<keyof T.DatasetFormatInfo, true>),
  ],
  [
    'DatasetUploadLimits',
    keys({
      max_bytes: true,
      max_files: true,
      ttl_hours: true,
      preview_max_files: true,
    } satisfies Record<keyof T.DatasetUploadLimits, true>),
  ],
  [
    'DatasetIssueCatalogEntry',
    keys({
      code: true,
      severity: true,
      blocking: true,
      bypassable: true,
      label: true,
    } satisfies Record<keyof T.DatasetIssueCatalogEntry, true>),
  ],
  [
    'DatasetFormatsResponse',
    keys({
      formats: true,
      issues: true,
      mapping_actions: true,
      match_kinds: true,
      processing_modes: true,
      parents_modes: true,
      trust_levels: true,
      upload_limits: true,
      status_labels: true,
    } satisfies Record<keyof T.DatasetFormatsResponse, true>),
  ],
  [
    'DatasetSource',
    keys({ path: true, format: true, coco_annotations: true } satisfies Record<
      keyof T.DatasetSource,
      true
    >),
  ],
  [
    'ClassMappingEntry',
    keys({
      dataset_class: true,
      action: true,
      class_id: true,
      new_class_name: true,
      new_class_group: true,
    } satisfies Record<keyof T.ClassMappingEntry, true>),
  ],
  [
    'DatasetImportOptions',
    keys({
      processing: true,
      label_trust: true,
      parents: true,
      freeze_test_split: true,
      missing_label: true,
      region_negatives: true,
      region_containment: true,
      name: true,
      source_tag: true,
      force: true,
    } satisfies Record<keyof T.DatasetImportOptions, true>),
  ],
  [
    'DatasetPreviewRequest',
    keys({
      source: true,
      mapping: true,
      accept_suggestions: true,
      options: true,
    } satisfies Record<keyof T.DatasetPreviewRequest, true>),
  ],
  [
    'DatasetImportRequest',
    keys({
      source: true,
      mapping: true,
      accept_suggestions: true,
      options: true,
      expected_import_key: true,
    } satisfies Record<keyof T.DatasetImportRequest, true>),
  ],
  [
    'DatasetIssueWire',
    keys({
      code: true,
      id: true,
      severity: true,
      blocking: true,
      bypassable: true,
      message: true,
      count: true,
      samples: true,
    } satisfies Record<keyof T.DatasetIssue, true>),
  ],
  [
    'DatasetIssueSampleWire',
    keys({ file: true, line: true, detail: true } satisfies Record<
      keyof T.DatasetIssueSample,
      true
    >),
  ],
  [
    'DatasetSplitRow',
    keys({
      split: true,
      images: true,
      labeled: true,
      negatives: true,
      unlabeled: true,
      boxes: true,
    } satisfies Record<keyof T.DatasetSplitRow, true>),
  ],
  [
    'IndexWouldHaveMapped',
    keys({ class_id: true, class_name: true } satisfies Record<
      keyof T.DatasetClassRef,
      true
    >),
  ],
  [
    'MappingSuggestionWire',
    keys({ action: true, class_id: true, class_name: true, match: true } satisfies Record<
      keyof T.MappingSuggestion,
      true
    >),
  ],
  [
    'ResolvedMapTarget',
    keys({
      dataset_class: true,
      kind: true,
      class_id: true,
      class_name: true,
      created: true,
    } satisfies Record<keyof T.ResolvedMapTarget, true>),
  ],
  [
    'DatasetClassRow',
    keys({
      dataset_class: true,
      dataset_id: true,
      boxes: true,
      images: true,
      source_class_id: true,
      index_would_have_mapped_to: true,
      merged_from: true,
      suggestion: true,
      resolved: true,
    } satisfies Record<keyof T.DatasetClassRow, true>),
  ],
  [
    'DatasetRegionInfo',
    keys({
      profile: true,
      region_class_name: true,
      parent_classes: true,
      parents_mode: true,
      standalone_boxes: true,
    } satisfies Record<keyof T.DatasetRegionInfo, true>),
  ],
  [
    'DatasetTotals',
    keys({
      images: true,
      boxes: true,
      images_already_indexed: true,
      images_to_ingest: true,
    } satisfies Record<keyof T.DatasetTotals, true>),
  ],
  [
    'DatasetEstimate',
    keys({ detector_images: true, embeddings: true } satisfies Record<
      keyof T.DatasetEstimate,
      true
    >),
  ],
  [
    'DatasetPreview',
    keys({
      project: true,
      format: true,
      root: true,
      source_sha: true,
      import_key: true,
      op_export: true,
      splits: true,
      totals: true,
      classes: true,
      region: true,
      issues: true,
      blocking: true,
      force_allowed: true,
      estimate: true,
    } satisfies Record<keyof T.DatasetPreview, true>),
  ],
  [
    'DatasetImportProgress',
    keys({
      images_total: true,
      images_done: true,
      images_failed: true,
      chunks_total: true,
      chunks_done: true,
      images_per_s: true,
      eta_s: true,
    } satisfies Record<keyof T.DatasetImportProgress, true>),
  ],
  [
    'DatasetImportReportWire',
    keys({
      images_created: true,
      images_reused: true,
      images_failed: true,
      images_skipped: true,
      items_reconciled_removed: true,
      items_created: true,
      items_updated: true,
      items_noop: true,
      labels_written: true,
      boxes_written: true,
      standalone_regions: true,
      negatives: true,
      unlabeled: true,
      parents_detected: true,
      proposals_created: true,
      proposals_merged: true,
      holdout_frozen: true,
      label_conflicts_locked: true,
      disagreements: true,
    } satisfies Record<keyof T.DatasetImportReport, true>),
  ],
  [
    'NextStep',
    keys({ action: true, method: true, path: true, reason: true } satisfies Record<
      keyof T.NextStep,
      true
    >),
  ],
  [
    'DatasetUndoReportWire',
    keys({
      import_id: true,
      dry_run: true,
      items_deleted: true,
      items_restored: true,
      items_kept_human_edited: true,
      items_kept_shared: true,
      items_reinstated: true,
      class_labels_removed: true,
      boxes_removed: true,
      boxes_kept_human_edited: true,
      proposals_deleted: true,
      holdout_flags_cleared: true,
      images_deleted: true,
      images_kept: true,
      classes_deprecated: true,
      samples: true,
    } satisfies Record<keyof T.DatasetUndoReport, true>),
  ],
  [
    'DatasetUndoRequest',
    keys({
      dry_run: true,
      remove_images: true,
      deprecate_created_classes: true,
    } satisfies Record<keyof T.DatasetUndoRequest, true>),
  ],
  [
    'DatasetImportJob',
    keys({
      project: true,
      import_id: true,
      import_key: true,
      name: true,
      status: true,
      reused: true,
      progress: true,
      waiting_for: true,
      actions: true,
      report: true,
      mapping: true,
      options: true,
      source: true,
      issues_summary: true,
      undo: true,
      next_steps: true,
      started_at: true,
      updated_at: true,
      finished_at: true,
      poll_after_s: true,
      labels: true,
      error: true,
    } satisfies Record<keyof T.DatasetImportJob, true>),
  ],
  [
    'ImportActions',
    keys({
      can_cancel: true,
      can_resume: true,
      can_undo: true,
    } satisfies Record<keyof T.ImportActions, true>),
  ],
  [
    'ImportAction',
    keys({ allowed: true, reason: true } satisfies Record<keyof T.ImportAction, true>),
  ],
  [
    'DatasetImportEntry',
    keys({
      source_stem: true,
      rel_path: true,
      image_id: true,
      image_path: true,
      image_created: true,
      split: true,
      label_state: true,
      status: true,
      error_kind: true,
      boxes: true,
      items: true,
    } satisfies Record<keyof T.DatasetImportEntry, true>),
  ],
  [
    'DatasetUploadResponse',
    keys({
      upload_id: true,
      dataset_path: true,
      bytes: true,
      files: true,
    } satisfies Record<keyof T.DatasetUploadResponse, true>),
  ],
  // -- Reprocess --
  [
    'ReprocessFilter',
    keys({
      all_images: true,
      class_id: true,
      class_names: true,
      class_source: true,
      classifier_conf_lt: true,
      cluster_id: true,
      conf_max: true,
      conf_min: true,
      dataset_split: true,
      detector: true,
      embedding_state: true,
      exclude_class_names: true,
      import_id: true,
      include_detected: true,
      item_text: true,
      label_source: true,
      label_validated: true,
      max_area: true,
      max_rank: true,
      min_area: true,
      min_blur_ratio: true,
      missing_provenance: true,
      missing_status: true,
      needs_new_class: true,
      on_negative_frame: true,
      open_vocab_set: true,
      open_vocab_status: true,
      origin: true,
      profile_not: true,
      profile_revision_below: true,
      proposed_by_import: true,
      reason: true,
      region_gate_skipped: true,
      region_status: true,
      review_dismissed: true,
      review_status: true,
      source: true,
      source_prompt: true,
    } satisfies Record<keyof T.ReprocessFilter, true>),
  ],
  [
    'ReprocessTargets',
    keys({
      image_ids: true,
      crop_ids: true,
      filter: true,
      limit: true,
      sample: true,
      seed: true,
    } satisfies Record<keyof T.ReprocessTargets, true>),
  ],
  [
    'EmbedOptions',
    keys({ only_missing: true, parts: true } satisfies Record<
      keyof T.EmbedOptions,
      true
    >),
  ],
  [
    'ReprocessRequest',
    keys({
      targets: true,
      scopes: true,
      region_mode: true,
      dry_run: true,
      embed: true,
    } satisfies Record<keyof T.ReprocessRequest, true>),
  ],
  [
    'ReprocessOneRequest',
    keys({ scopes: true, region_mode: true, dry_run: true } satisfies Record<
      keyof T.ReprocessOneRequest,
      true
    >),
  ],
  [
    'BreakdownRow',
    keys({ detector: true, reason: true, count: true } satisfies Record<
      keyof T.BreakdownRow,
      true
    >),
  ],
  [
    'ReprocessScopeResult',
    keys({
      scope: true,
      selected: true,
      locked_skipped: true,
      queued: true,
      failed: true,
      not_found: true,
      breakdown: true,
      detail: true,
    } satisfies Record<keyof T.ReprocessScopeResult, true>),
  ],
  [
    'ReprocessJobInfo',
    keys({
      job_id: true,
      status: true,
      scopes: true,
      results: true,
      images_total: true,
      images_done: true,
      images_failed: true,
      error: true,
      started_at: true,
      updated_at: true,
      finished_at: true,
      poll_after_s: true,
    } satisfies Record<keyof T.ReprocessJob, true>),
  ],
  [
    'ReprocessWireResponse',
    keys({ dry_run: true, scopes: true, job: true, items: true } satisfies Record<
      keyof T.ReprocessResponse,
      true
    >),
  ],
];

describe('types_import.ts keys match the vendored OpenAPI schemas', () => {
  it('loaded a non-trivial case list (guards a vacuous pass)', () => {
    expect(CASES.length).toBeGreaterThan(30);
  });
  for (const [schema, tsKeys] of CASES) {
    it(schema, () => {
      expect(tsKeys).toEqual(declared(schema));
    });
  }
});

describe('Reprocess vocabulary is pinned to the request enums', () => {
  const enumOf = (schema: string, prop: string, items = false): string[] => {
    const p = schemas[schema]!.properties![prop]!;
    return [...((items ? p.items?.enum : p.enum) ?? [])].sort();
  };

  it('REPROCESS_SCOPES is the scopes enum, on every request schema', () => {
    expect([...REPROCESS_SCOPES].sort()).toEqual(
      enumOf('ReprocessOneRequest', 'scopes', true),
    );
    expect([...REPROCESS_SCOPES].sort()).toEqual(
      enumOf('ReprocessRequest', 'scopes', true),
    );
    expect(REPROCESS_SCOPES.length).toBeGreaterThan(0);
  });

  it('REGION_MODES is the region_mode enum, on every request schema', () => {
    expect([...REGION_MODES].sort()).toEqual(
      enumOf('ReprocessOneRequest', 'region_mode'),
    );
    expect([...REGION_MODES].sort()).toEqual(enumOf('ReprocessRequest', 'region_mode'));
    expect(REGION_MODES.length).toBeGreaterThan(0);
  });
});

describe('Reprocess request bodies send only declared keys', () => {
  afterEach(() => vi.unstubAllGlobals());

  function capture() {
    const fetchMock = vi.fn().mockImplementation(
      async () =>
        new Response(JSON.stringify({ dry_run: false, scopes: [], items: [] }), {
          status: 200,
          headers: { 'content-type': 'application/json' },
        }),
    );
    vi.stubGlobal('fetch', fetchMock);
    return () =>
      fetchMock.mock.calls.map(
        (c) =>
          [String(c[0]), JSON.parse(String((c[1] as RequestInit).body))] as [
            string,
            Record<string, unknown>,
          ],
      );
  }

  it('POST /crops/{id}/reprocess and /images/{id}/reprocess (ReprocessOneRequest)', async () => {
    const calls = capture();
    const full = {
      scopes: [...REPROCESS_SCOPES],
      region_mode: 'reverify' as const,
      dry_run: false,
    };
    await reprocessCrop('c1', full);
    await reprocessImage('img_1', full);
    for (const [, body] of calls()) {
      expect(Object.keys(body).sort()).toEqual(declared('ReprocessOneRequest'));
    }
  });

  it('POST /reprocess (ReprocessRequest) from the flow', async () => {
    const calls = capture();
    await reprocessBatch({
      targets: { crop_ids: ['a'] },
      scopes: ['embed'],
      region_mode: 'redetect',
      dry_run: true,
    });
    const allowed = declared('ReprocessRequest');
    for (const k of Object.keys(calls()[0]![1])) expect(allowed).toContain(k);
  });

  it('the single-target flow sends exactly the one-request keys (region_mode only with region)', async () => {
    const calls = capture();
    const f = new ReprocessFlow({ kind: 'image', imageId: 'img_1' });
    f.toggleScope('detect', true);
    f.setRegionMode('reverify');
    await f.apply();
    expect(calls()[0]![1]).toEqual({ scopes: ['detect'], dry_run: false });
    const g = new ReprocessFlow({ kind: 'crop', cropId: 'c1' });
    g.toggleScope('region', true);
    g.setRegionMode('reverify');
    await g.apply();
    const body = calls()[1]![1];
    expect(body).toEqual({ scopes: ['region'], region_mode: 'reverify', dry_run: false });
    for (const k of Object.keys(body))
      expect(declared('ReprocessOneRequest')).toContain(k);
  });
});
