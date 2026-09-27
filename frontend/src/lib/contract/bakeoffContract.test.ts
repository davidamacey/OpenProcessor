/**
 * `types_bakeoff.ts` vs the vendored OpenAPI schemas (OpenProcessor #34
 * v2 bake-off wire). Each key map is compile-time exact against its
 * interface (`satisfies Record<keyof T, true>` rejects a missing or an
 * extra key), and the test pins it to the schema's property set, so a
 * served rename fails here instead of rendering "—" silently.
 */
import { describe, expect, it } from 'vitest';
import spec from '../../../contracts/openprocessor/openapi/curation.json';
import type * as T from '$lib/types_bakeoff';

type Schema = { properties?: Record<string, unknown>; required?: string[] };
const schemas = (spec as { components: { schemas: Record<string, Schema> } }).components
  .schemas;

const keys = <K extends string>(o: Record<K, true>) => Object.keys(o).sort();

const CASES: [string, string[]][] = [
  [
    'EvalDatasetClass',
    keys({
      eval_class_id: true,
      name: true,
      registry_class_id: true,
      registry_class_name: true,
      n_objects: true,
      n_images: true,
    } satisfies Record<keyof T.EvalDatasetClass, true>),
  ],
  [
    'EvalDataset',
    keys({
      id: true,
      source: true,
      group: true,
      name: true,
      path: true,
      is_current: true,
      dataset_kind: true,
      nc: true,
      classes: true,
      n_images: true,
      n_objects: true,
      n_background_images: true,
      frozen_test_sha: true,
      test_label_sha: true,
      sha_source: true,
      dataset_sha: true,
      exported_at: true,
      unlabeled_items_on_exported_images: true,
      frozen_ok: true,
    } satisfies Record<keyof T.EvalDataset, true>),
  ],
  [
    'EvalDatasetList',
    keys({ datasets: true, count: true } satisfies Record<keyof T.EvalDatasetList, true>),
  ],
  [
    'TrainTestOverlap',
    keys({ n_images: true, fraction: true } satisfies Record<
      keyof T.TrainTestOverlap,
      true
    >),
  ],
  [
    'TrainedModelForDataset',
    keys({
      dataset_id: true,
      same_export: true,
      same_frozen_test: true,
      n_classes_mapped: true,
      train_test_overlap: true,
    } satisfies Record<keyof T.TrainedModelForDataset, true>),
  ],
  [
    'TrainedModel',
    keys({
      run_id: true,
      display_name: true,
      model_family: true,
      model_size: true,
      imgsz: true,
      checkpoint_path: true,
      finished_at: true,
      campaign_id: true,
      train_export_id: true,
      dataset_sha: true,
      frozen_test_sha: true,
      class_names: true,
      single_cls: true,
      trainer_map50: true,
      trainer_map50_split: true,
      for_dataset: true,
    } satisfies Record<keyof T.TrainedModel, true>),
  ],
  [
    'TrainedModelList',
    keys({ models: true, count: true } satisfies Record<keyof T.TrainedModelList, true>),
  ],
  [
    'BakeoffProfileRow',
    keys({
      name: true,
      description: true,
      kind: true,
      default: true,
      class_filter: true,
      imgsz: true,
      conf_floor: true,
      nms_iou: true,
      op_conf: true,
      op_iou: true,
      rank_metric: true,
      default_backend: true,
      triton_model: true,
      context_class_ids: true,
      context_class_names: true,
      baselines_path: true,
    } satisfies Record<keyof T.BakeoffProfile, true>),
  ],
  [
    'BakeoffProfileList',
    keys({
      profiles: true,
      count: true,
      default_profile: true,
      default_error: true,
    } satisfies Record<keyof T.BakeoffProfileList, true>),
  ],
  [
    'BaselineModel',
    keys({
      name: true,
      backend: true,
      weights: true,
      imgsz: true,
      mode: true,
      class_map: true,
      backend_options: true,
      training_data: true,
      triton_model: true,
    } satisfies Record<keyof T.BaselineModel, true>),
  ],
  [
    'BaselineModelList',
    keys({ baselines: true, count: true } satisfies Record<
      keyof T.BaselineModelList,
      true
    >),
  ],
  [
    'RunModelRef',
    keys({
      source: true,
      run_id: true,
      display_name: true,
      backend: true,
      mode: true,
    } satisfies Record<keyof T.RunModelRef, true>),
  ],
  [
    'BaselineModelRef',
    keys({ source: true, name: true, display_name: true } satisfies Record<
      keyof T.BaselineModelRef,
      true
    >),
  ],
  [
    'CustomModelRef',
    keys({
      source: true,
      name: true,
      backend: true,
      weights: true,
      triton_model: true,
      imgsz: true,
      mode: true,
      class_map: true,
      backend_options: true,
      display_name: true,
    } satisfies Record<keyof T.CustomModelRef, true>),
  ],
  ['DatasetRef', keys({ id: true } satisfies Record<keyof T.DatasetRef, true>)],
  [
    'QuantizeRequest',
    keys({
      run_id: true,
      formats: true,
      n_calib: true,
      calib_split: true,
      throughput: true,
    } satisfies Record<keyof T.QuantizeRequest, true>),
  ],
  [
    'BakeoffRunRequest',
    keys({
      job_id: true,
      profile: true,
      datasets: true,
      models: true,
      quantize: true,
    } satisfies Record<keyof T.BakeoffRunRequest, true>),
  ],
  [
    'NotCoveredClass',
    keys({ eval_class_id: true, name: true } satisfies Record<
      keyof T.NotCoveredClass,
      true
    >),
  ],
  [
    'UnmappedModelClass',
    keys({ model_class_id: true, name: true, n_predictions: true } satisfies Record<
      keyof T.UnmappedModelClass,
      true
    >),
  ],
  [
    'ClassMapping',
    keys({
      method: true,
      model_to_eval: true,
      model_to_eval_names: true,
      unmapped_model_classes: true,
      not_covered_eval_classes: true,
      warnings: true,
    } satisfies Record<keyof T.ClassMapping, true>),
  ],
  [
    'AcceptedDataset',
    keys({
      id: true,
      path: true,
      frozen_test_sha: true,
      test_label_sha: true,
      n_eval_classes: true,
    } satisfies Record<keyof T.AcceptedDataset, true>),
  ],
  [
    'AcceptedModel',
    keys({
      model: true,
      display_name: true,
      source: true,
      class_mapping: true,
      train_test_overlap: true,
    } satisfies Record<keyof T.AcceptedModel, true>),
  ],
  [
    'BakeoffRunAccepted',
    keys({
      status: true,
      job_id: true,
      profile: true,
      datasets: true,
      models: true,
      warnings: true,
    } satisfies Record<keyof T.BakeoffRunAccepted, true>),
  ],
  [
    'JobProgress',
    keys({ done: true, total: true } satisfies Record<keyof T.JobProgress, true>),
  ],
  [
    'CompletedTask',
    keys({ dataset: true, model: true } satisfies Record<keyof T.CompletedTask, true>),
  ],
  [
    'FailedTask',
    keys({ stage: true, dataset: true, model: true, error: true } satisfies Record<
      keyof T.FailedTask,
      true
    >),
  ],
  [
    'BakeoffStatus',
    keys({
      schema_version: true,
      job_id: true,
      state: true,
      profile: true,
      datasets: true,
      models: true,
      started_at: true,
      finished_at: true,
      progress: true,
      completed: true,
      failed: true,
      error: true,
    } satisfies Record<keyof T.BakeoffStatus, true>),
  ],
  [
    'BakeoffRunRow',
    keys({
      job_id: true,
      state: true,
      profile: true,
      datasets: true,
      models: true,
      started_at: true,
      finished_at: true,
    } satisfies Record<keyof T.BakeoffRunRow, true>),
  ],
  ['BakeoffRunList', keys({ runs: true } satisfies Record<keyof T.BakeoffRunList, true>)],
  [
    'MetricBlock',
    keys({
      n_classes: true,
      map_50: true,
      map_50_95: true,
      map_75: true,
      ap_small: true,
      ap_medium: true,
      ap_large: true,
      precision: true,
      recall: true,
      f1: true,
      mean_iou: true,
      tp: true,
      fp: true,
      fn: true,
    } satisfies Record<keyof T.MetricBlock, true>),
  ],
  [
    'CommonMetricBlock',
    keys({
      n_classes: true,
      map_50: true,
      map_50_95: true,
      precision: true,
      recall: true,
      f1: true,
      tp: true,
      fp: true,
      fn: true,
    } satisfies Record<keyof T.CommonMetricBlock, true>),
  ],
  [
    'PerClassRow',
    keys({
      eval_class_id: true,
      name: true,
      n_gt: true,
      covered: true,
      model_class_ids: true,
      ap50: true,
      ap50_95: true,
      ap75: true,
      precision: true,
      recall: true,
      f1: true,
      tp: true,
      fp: true,
      fn: true,
    } satisfies Record<keyof T.PerClassRow, true>),
  ],
  [
    'Coverage',
    keys({
      n_eval_classes: true,
      n_covered: true,
      not_covered: true,
      unmapped_model_classes: true,
      predictions_outside_scored_classes: true,
    } satisfies Record<keyof T.Coverage, true>),
  ],
  [
    'RowClassMapping',
    keys({ method: true, warnings: true } satisfies Record<
      keyof T.RowClassMapping,
      true
    >),
  ],
  [
    'LatencyStats',
    keys({ mean: true, p50: true, p90: true, p99: true } satisfies Record<
      keyof T.LatencyStats,
      true
    >),
  ],
  [
    'StratumMetrics',
    keys({ n_images: true, map_50: true, precision: true, recall: true } satisfies Record<
      keyof T.StratumMetrics,
      true
    >),
  ],
  [
    'ComparisonRow',
    keys({
      rank: true,
      model: true,
      display_name: true,
      source: true,
      run_id: true,
      runtime: true,
      imgsz: true,
      training_data: true,
      overall: true,
      common: true,
      per_class: true,
      coverage: true,
      class_mapping: true,
      train_test_overlap: true,
      latency_ms: true,
      fps: true,
      size_mb: true,
      per_stratum: true,
    } satisfies Record<keyof T.ComparisonRow, true>),
  ],
  [
    'ComparisonDataset',
    keys({
      id: true,
      frozen_test_sha: true,
      test_label_sha: true,
      n_images: true,
      n_objects: true,
      n_background_images: true,
    } satisfies Record<keyof T.ComparisonDataset, true>),
  ],
  [
    'EvalClassGt',
    keys({ eval_class_id: true, name: true, n_gt: true } satisfies Record<
      keyof T.EvalClassGt,
      true
    >),
  ],
  [
    'FailedModel',
    keys({ model: true, error: true } satisfies Record<keyof T.FailedModel, true>),
  ],
  [
    'BakeoffComparison',
    keys({
      schema_version: true,
      job_id: true,
      profile: true,
      thresholds: true,
      dataset: true,
      eval_classes: true,
      common_classes: true,
      rank_by: true,
      rank_scope: true,
      models: true,
      failed: true,
      warnings: true,
      n_models: true,
    } satisfies Record<keyof T.BakeoffComparison, true>),
  ],
  [
    'MatrixDataset',
    keys({
      id: true,
      frozen_test_sha: true,
      test_label_sha: true,
      rank_scope: true,
      n_common_classes: true,
    } satisfies Record<keyof T.MatrixDataset, true>),
  ],
  [
    'MatrixModel',
    keys({ model: true, display_name: true, source: true } satisfies Record<
      keyof T.MatrixModel,
      true
    >),
  ],
  [
    'MatrixCell',
    keys({
      map_50: true,
      map_50_95: true,
      precision: true,
      recall: true,
      f1: true,
      latency_ms: true,
      size_mb: true,
      coverage: true,
      rank: true,
    } satisfies Record<keyof T.MatrixCell, true>),
  ],
  [
    'BakeoffMatrix',
    keys({
      schema_version: true,
      job_id: true,
      rank_by: true,
      datasets: true,
      models: true,
      metrics: true,
      cells: true,
      best: true,
    } satisfies Record<keyof T.BakeoffMatrix, true>),
  ],
];

/**
 * PENDING-REBASE allow-list, NOT a permanent exception (2026-09-26,
 * projects P1). The vendored OpenAPI snapshot is currently synced from
 * OpenProcessor's `cutover/projects-foundation` @ `dc2b4e0e` — a branch
 * cut before the upstream `main` commit that added these three fields
 * (OpenProcessor 3cd4ca87, already adopted on this frontend's `master`).
 * That branch will be rebased onto `main` and re-synced; when it is,
 * `npm run contract:sync` will pick these back up in the snapshot and
 * this allow-list should be deleted, not widened. Until then the fields
 * stay on the frontend types/fixtures (never deleted — they're real,
 * already-adopted server behavior) and this test just doesn't demand
 * the stale snapshot serve them too.
 */
const PENDING_REBASE_FIELDS: Record<string, string[]> = {
  EvalDatasetClass: ['registry_class_name'],
  BakeoffProfileRow: ['context_class_names'],
  ClassMapping: ['model_to_eval_names'],
};

describe('bake-off v2 types vs vendored OpenAPI', () => {
  it.each(CASES)('%s has exactly the served properties', (name, frontendKeys) => {
    const schema = schemas[name];
    expect(schema, `schema ${name} missing`).toBeDefined();
    const pending = new Set(PENDING_REBASE_FIELDS[name] ?? []);
    const expectedKeys = frontendKeys.filter((k) => !pending.has(k));
    expect(Object.keys(schema.properties ?? {}).sort()).toEqual(expectedKeys);
  });

  it('BakeoffMatrix.best is a list of winners per dataset per metric', () => {
    const best = schemas.BakeoffMatrix.properties?.best as {
      additionalProperties: { additionalProperties: { type: string } };
    };
    expect(best.additionalProperties.additionalProperties.type).toBe('array');
  });

  it('BakeoffRunRequest forbids extra fields', () => {
    const req = schemas.BakeoffRunRequest as Schema & { additionalProperties?: boolean };
    expect(req.additionalProperties).toBe(false);
  });
});
