/**
 * Real served fixtures from the 2026-09-24T23-47-55_yolo26n live
 * train-smoke run — `GET {API_PREFIX}/train/status/{job_id}` and
 * `GET {API_PREFIX}/train/manifest/{job_id}`, captured verbatim
 * (`artifacts_local/cw-live/train-smoke/f_status.json` /
 * `f_manifest.json`, gitignored — this is the committed copy tests
 * import). Used by `RunResults.test.ts` / `trainResults.test.ts` so
 * the results-view tests exercise the actual wire shape rather than a
 * hand-typed guess.
 *
 * **OpenProcessor #34 W1 adoption (2026-09-25):** this run predates the
 * `last_epoch_metric`/`best_checkpoint_metric` split — re-fetched live
 * against the currently-deployed (pre-fix) backend, its status still
 * carries the OLD `best_metric`/`last_metric` keys AND a buggy
 * `best_checkpoint_metric` back-filled from the wrong (test-split) eval
 * numbers. Per the W1 backend fix (not yet deployed): a run whose
 * status.json predates these fields serves `null` for BOTH
 * `last_epoch_metric` and `best_checkpoint_metric` — no incorrect
 * back-fill. This fixture is hand-corrected to that post-fix shape
 * (nulls) rather than the live buggy response, so tests exercise the
 * real fixed behavior, not a bug being adopted on purpose. `eval` is
 * otherwise served verbatim (unaffected by the fix) and predates the
 * `split`/`head` fields too — the genuinely oldest captured shape.
 */
import type { TrainJobStatus, TrainManifest } from '$lib/types_train';

export const trainStatusFixture: TrainJobStatus = {
  job_id: '2026-09-24T23-47-55_yolo26n',
  campaign_id: null,
  state: 'finished',
  started_at: '2026-09-24T23:47:56Z',
  finished_at: '2026-09-24T23:51:09Z',
  current_epoch: 20,
  total_epochs: 20,
  epoch_time_s: null,
  last_epoch_metric: null,
  best_checkpoint_metric: null,
  mlflow_run_id: '724a9292103d4ec3b153068758be340d',
  mlflow_run_url:
    'http://op-mlflow:5000/#/experiments/1/runs/724a9292103d4ec3b153068758be340d',
  checkpoint_path:
    '/var/lib/openprocessor/training_runs/2026-09-24T23-47-55_yolo26n/weights/best.pt',
  gpu: [{ index: 0, util_pct: 2, mem_used_mb: 18155, mem_total_mb: 49140 }],
  eval: {
    map50: 0.9191,
    map50_95: 0.85096,
    per_class: [
      {
        class_id: 0,
        name: 'miata',
        precision: 1.0,
        recall: 0.5758532802586369,
        f1: 0.730846313514828,
        ap50: 0.755,
        support: 5,
      },
      {
        class_id: 1,
        name: 'minicooper',
        precision: 0.6392386650044177,
        recall: 1.0,
        f1: 0.7799214094339276,
        ap50: 0.9378571428571427,
        support: 5,
      },
      {
        class_id: 2,
        name: 'mustang',
        precision: 0.6440803613748901,
        recall: 1.0,
        f1: 0.7835144516126534,
        ap50: 0.995,
        support: 5,
      },
      {
        class_id: 3,
        name: 'porsche',
        precision: 0.6641866095965356,
        recall: 0.8,
        f1: 0.7257944912139916,
        ap50: 0.6300000000000001,
        support: 5,
      },
      {
        class_id: 4,
        name: 'vw',
        precision: 0.800825081032084,
        recall: 1.0,
        f1: 0.8893979648185678,
        ap50: 0.995,
        support: 5,
      },
    ],
    confusion_matrix_path:
      '/var/lib/openprocessor/training_runs/2026-09-24T23-47-55_yolo26n/confusion_matrix.png',
  },
  compare: null,
  error: null,
  heartbeat_at: '2026-09-24T23:51:09Z',
  class_remap_copy_failed: false,
};

export const trainManifestFixture: TrainManifest = {
  campaign_id: null,
  code_versions: {
    api_sha: null,
    trainer_sha: null,
    trainer_image_id: null,
    ultralytics_pkg: '8.4.48',
    ultralytics_sha: '8f5d355cd05b91503e7bf62429681dea4fa4b004',
  },
  created_at: '2026-09-24T23:51:09.494255+00:00',
  job_id: '2026-09-24T23-47-55_yolo26n',
  kind: 'train',
  lineage: {
    augmentation_seed: 42,
    class_remap: {
      include_classes: [38, 39, 44, 52, 79],
      names: ['miata', 'minicooper', 'mustang', 'porsche', 'vw'],
      new_to_original: { '0': 38, '1': 39, '2': 44, '3': 52, '4': 79 },
      original_to_new: { '38': 0, '39': 1, '44': 2, '52': 3, '79': 4 },
      single_cls: false,
    },
    dataset_sha: null,
    frozen_test_sha: null,
    test_label_sha: null,
    dataset_version_tag: null,
    deterministic: true,
    export_dir: '/exports/20260924T233203Z',
    include_classes: [38, 39, 44, 52, 79],
    registry_sha: '3aee444124c8fa161030c424cc891a40139cde7b96d5cf73d0363e00d7c7f671',
    single_cls: false,
    training_seed: 42,
  },
  promoted_to: null,
  results: {
    last_epoch_metric: null,
    best_checkpoint_metric: null,
    checkpoint_path:
      '/var/lib/openprocessor/training_runs/2026-09-24T23-47-55_yolo26n/weights/best.pt',
    checkpoint_sha256: 'cc5ffb75e020b54d87df6f534de2a7a74eaa02519c69658a9504e2fe42d15e81',
    compare: null,
    eval: {
      confusion_matrix_path:
        '/var/lib/openprocessor/training_runs/2026-09-24T23-47-55_yolo26n/confusion_matrix.png',
      map50: 0.9191,
      map50_95: 0.85096,
      per_class: [
        {
          ap50: 0.755,
          class_id: 0,
          f1: 0.730846313514828,
          name: 'miata',
          precision: 1.0,
          recall: 0.5758532802586369,
          support: 5,
        },
        {
          ap50: 0.9378571428571427,
          class_id: 1,
          f1: 0.7799214094339276,
          name: 'minicooper',
          precision: 0.6392386650044177,
          recall: 1.0,
          support: 5,
        },
        {
          ap50: 0.995,
          class_id: 2,
          f1: 0.7835144516126534,
          name: 'mustang',
          precision: 0.6440803613748901,
          recall: 1.0,
          support: 5,
        },
        {
          ap50: 0.6300000000000001,
          class_id: 3,
          f1: 0.7257944912139916,
          name: 'porsche',
          precision: 0.6641866095965356,
          recall: 0.8,
          support: 5,
        },
        {
          ap50: 0.995,
          class_id: 4,
          f1: 0.8893979648185678,
          name: 'vw',
          precision: 0.800825081032084,
          recall: 1.0,
          support: 5,
        },
      ],
    },
    final_state: 'finished',
    mlflow_run_id: '724a9292103d4ec3b153068758be340d',
    mlflow_run_url:
      'http://op-mlflow:5000/#/experiments/1/runs/724a9292103d4ec3b153068758be340d',
  },
  spec: {
    augmentation: {
      albumentations: {},
      enabled: true,
      multiplier: 3,
      per_class_multiplier: {},
      preset: 'balanced_default',
    },
    cuda_visible_devices: '2',
    hyperparameters: {
      batch: 64,
      epochs: 20,
      imgsz: 640,
      lr0: 0.005,
      momentum: 0.947,
      optimizer: 'MuSGD',
      patience: 10,
      weight_decay: 0.00064,
    },
    model_family: 'yolo26',
    model_size: 'n',
    profile: 'probe',
  },
};

/**
 * Hand-constructed (not live-captured) example of a genuinely post-W1
 * run — no live run exists yet whose `status.json` was written after
 * the fix, so this fills in realistic `last_epoch_metric`/
 * `best_checkpoint_metric` (with their own `epoch` numbers, one
 * coherent map50+map50_95 row each) and `eval.split`/`eval.head` on top
 * of `trainStatusFixture`'s otherwise-real shape, to exercise the
 * epoch-labelled metrics and eval-head rendering `trainStatusFixture`
 * alone (nulled per the fix) can't.
 */
export const trainStatusFixtureW1: TrainJobStatus = {
  ...trainStatusFixture,
  job_id: '2026-09-26T00-00-00_yolo26n',
  last_epoch_metric: { epoch: 20, map50: 0.9191, map50_95: 0.8459816666666667 },
  best_checkpoint_metric: { epoch: 17, map50: 0.9356, map50_95: 0.85096 },
  eval: {
    ...trainStatusFixture.eval,
    split: 'test',
    head: 'end2end',
  },
};
