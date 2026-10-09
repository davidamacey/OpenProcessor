/**
 * v2 bake-off wire fixtures (neutral widget domain). Shapes follow the
 * vendored OpenAPI; values are illustrative.
 */
import type {
  BakeoffComparison,
  BakeoffMatrix,
  BakeoffRunAccepted,
  BakeoffStatus,
  ComparisonRow,
  EvalDataset,
  MetricBlock,
  PerClassRow,
  TrainedModel,
} from '$lib/types_bakeoff';

export const DS_CURRENT = 'export:20260924T233203Z';
export const DS_OLDER = 'export:20260901T000000Z';
export const DS_EXTERNAL = 'external:curated/widget_set';

function dataset(over: Partial<EvalDataset> & Pick<EvalDataset, 'id'>): EvalDataset {
  return {
    source: 'export',
    group: null,
    name: over.id.split(':')[1] ?? over.id,
    path: `/exports/${over.id}`,
    is_current: false,
    dataset_kind: 'multi_class',
    nc: 5,
    classes: [
      {
        eval_class_id: 0,
        name: 'gear',
        registry_class_id: 10,
        registry_class_name: 'gear',
        n_objects: 5,
        n_images: 5,
      },
      {
        eval_class_id: 1,
        name: 'bolt',
        registry_class_id: 11,
        registry_class_name: 'bolt',
        n_objects: 4,
        n_images: 4,
      },
    ],
    n_images: 9,
    n_objects: 9,
    n_background_images: 0,
    frozen_test_sha: 'aaaaaaaaaaaaaaaa',
    test_label_sha: 'bbbbbbbbbbbbbbbb',
    sha_source: 'computed',
    dataset_sha: null,
    exported_at: '2026-09-24T23:32:11Z',
    unlabeled_items_on_exported_images: null,
    frozen_ok: null,
    ...over,
  };
}

export const EVAL_DATASETS: EvalDataset[] = [
  dataset({ id: DS_CURRENT, is_current: true }),
  dataset({ id: DS_OLDER }),
  dataset({
    id: DS_EXTERNAL,
    source: 'external',
    group: 'curated',
    name: 'widget_set',
    dataset_kind: 'external',
    frozen_ok: true,
  }),
];

export function trainedModel(over: Partial<TrainedModel> = {}): TrainedModel {
  return {
    run_id: 'run-a',
    display_name: 'run-a',
    model_family: 'yolo26',
    model_size: 'n',
    imgsz: 640,
    checkpoint_path: '/runs/run-a/weights/best.pt',
    finished_at: '2026-09-24T23:51:09Z',
    campaign_id: null,
    train_export_id: DS_CURRENT,
    dataset_sha: null,
    frozen_test_sha: null,
    class_names: ['gear', 'bolt'],
    single_cls: false,
    trainer_map50: 0.9356,
    trainer_map50_split: 'test',
    ...over,
  };
}

const metrics = (v: number | null): MetricBlock => ({
  n_classes: 2,
  map_50: v,
  map_50_95: v,
  map_75: v,
  ap_small: null,
  ap_medium: null,
  ap_large: null,
  precision: v,
  recall: v,
  f1: v,
  mean_iou: v,
  tp: 1,
  fp: 0,
  fn: 0,
});

function perClass(
  eval_class_id: number,
  name: string,
  v: number | null,
  covered = true,
): PerClassRow {
  return {
    eval_class_id,
    name,
    n_gt: 5,
    covered,
    model_class_ids: covered ? [eval_class_id] : [],
    ap50: covered ? v : null,
    ap50_95: covered ? v : null,
    ap75: covered ? v : null,
    precision: covered ? v : null,
    recall: covered ? v : null,
    f1: covered ? v : null,
    tp: covered ? 1 : null,
    fp: covered ? 0 : null,
    fn: covered ? 0 : null,
  };
}

function row(over: Partial<ComparisonRow> & Pick<ComparisonRow, 'model'>): ComparisonRow {
  return {
    rank: 1,
    display_name: over.model,
    source: 'run',
    run_id: null,
    runtime: 'ultralytics',
    imgsz: 640,
    training_data: null,
    overall: metrics(0.5),
    common: { ...metrics(0.5) },
    per_class: [perClass(0, 'gear', 0.5), perClass(1, 'bolt', 0.5)],
    coverage: {
      n_eval_classes: 2,
      n_covered: 2,
      not_covered: [],
      unmapped_model_classes: [],
      predictions_outside_scored_classes: 0,
    },
    class_mapping: { method: 'registry_ids', warnings: [] },
    train_test_overlap: { n_images: 0, fraction: 0 },
    latency_ms: { mean: 4.2, p50: 4, p90: 5, p99: 6 },
    fps: 238,
    size_mb: 5.4,
    per_stratum: {},
    ...over,
  };
}

/** Two models; the subset model covers only `gear` and maps an extra class. */
export const COMPARISON: BakeoffComparison = {
  schema_version: 2,
  job_id: 'job-1',
  profile: 'generic',
  thresholds: { conf_floor: 0.001, nms_iou: 0.7, op_conf: 0.25, op_iou: 0.45 },
  dataset: { id: DS_CURRENT, n_images: 9, n_objects: 9, n_background_images: 0 },
  eval_classes: [
    { eval_class_id: 0, name: 'gear', n_gt: 5 },
    { eval_class_id: 1, name: 'bolt', n_gt: 4 },
  ],
  common_classes: [0],
  rank_by: 'map_50_95',
  rank_scope: 'common',
  models: [
    row({ model: 'run:run-a', display_name: 'full model', run_id: 'run-a' }),
    row({
      model: 'run:run-b',
      display_name: 'subset model',
      run_id: 'run-b',
      rank: null,
      overall: metrics(0),
      per_class: [perClass(0, 'gear', 0), perClass(1, 'bolt', null, false)],
      coverage: {
        n_eval_classes: 2,
        n_covered: 1,
        not_covered: [{ eval_class_id: 1, name: 'bolt' }],
        unmapped_model_classes: [
          { model_class_id: 3, name: 'sprocket', n_predictions: 7 },
        ],
        predictions_outside_scored_classes: 0,
      },
      train_test_overlap: { n_images: 2, fraction: 0.2222 },
    }),
  ],
  failed: [],
  warnings: [],
  n_models: 2,
};

export const MATRIX: BakeoffMatrix = {
  schema_version: 2,
  job_id: 'job-1',
  rank_by: 'map_50_95',
  datasets: [
    {
      id: DS_CURRENT,
      frozen_test_sha: null,
      test_label_sha: null,
      rank_scope: 'common',
      n_common_classes: 1,
    },
  ],
  models: [
    { model: 'run:run-a', display_name: 'full model', source: 'run' },
    { model: 'run:run-b', display_name: 'subset model', source: 'run' },
    { model: 'baseline:ref', display_name: 'reference', source: 'baseline' },
  ],
  metrics: [
    'map_50',
    'map_50_95',
    'precision',
    'recall',
    'f1',
    'latency_ms',
    'size_mb',
    'coverage',
  ],
  cells: {
    'run:run-a': {
      [DS_CURRENT]: {
        map_50: 0.9,
        map_50_95: 0.6,
        precision: 0.8,
        recall: 0.7,
        f1: 0.75,
        latency_ms: 4.2,
        size_mb: 5.4,
        coverage: 1,
        rank: 1,
      },
    },
    'run:run-b': {
      [DS_CURRENT]: {
        map_50: 0.9,
        map_50_95: 0.6,
        precision: 0.5,
        recall: 0.4,
        f1: 0.45,
        latency_ms: 3.1,
        size_mb: 5.4,
        coverage: 0.5,
        rank: 1,
      },
    },
    'baseline:ref': {
      [DS_CURRENT]: {
        map_50: 0.3,
        map_50_95: 0.2,
        precision: null,
        recall: 0.1,
        f1: 0.1,
        latency_ms: 9,
        size_mb: 12,
        coverage: 0.5,
        rank: 3,
      },
    },
  },
  best: {
    [DS_CURRENT]: {
      map_50_95: ['run:run-a', 'run:run-b'],
      map_50: ['run:run-a', 'run:run-b'],
      precision: ['run:run-a'],
      latency_ms: ['run:run-b'],
    },
  },
};

export const ACCEPTED: BakeoffRunAccepted = {
  status: 'enqueued',
  job_id: 'job-1',
  profile: 'generic',
  datasets: [
    {
      id: DS_CURRENT,
      path: '/exports/x',
      frozen_test_sha: null,
      test_label_sha: null,
      n_eval_classes: 2,
    },
  ],
  models: [
    {
      model: 'run:run-b',
      display_name: 'subset model',
      source: 'run',
      class_mapping: {
        [DS_CURRENT]: {
          method: 'run_class_remap',
          model_to_eval: { '0': 0 },
          model_to_eval_names: { '0': 'gear' },
          unmapped_model_classes: [],
          not_covered_eval_classes: [{ eval_class_id: 1, name: 'bolt' }],
          warnings: [],
        },
      },
      train_test_overlap: { [DS_CURRENT]: { n_images: 0, fraction: 0 } },
    },
  ],
  warnings: [],
};

export function status(over: Partial<BakeoffStatus> = {}): BakeoffStatus {
  return {
    schema_version: 2,
    job_id: 'job-1',
    state: 'running',
    profile: 'generic',
    datasets: [DS_CURRENT],
    models: ['run:run-a', 'run:run-b'],
    started_at: '2026-09-25T01:02:03Z',
    finished_at: null,
    progress: { done: 1, total: 2 },
    completed: [{ dataset: DS_CURRENT, model: 'run:run-a' }],
    failed: [],
    error: null,
    ...over,
  };
}
