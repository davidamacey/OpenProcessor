/**
 * Export wire types: split counts, export status and result, single-class
 * export, the dataset list and the test holdout.
 */
/** Instance counts per split — `GET {API_PREFIX}/export/status`'s
 *  `split_counts` (OpenProcessor `ExportSplitCounts`, df01309). */
export interface ExportSplitCounts {
  train: number;
  val: number;
  test: number;
}

/** One class's instance counts per split, as recorded in the export
 *  manifest (OpenProcessor `ExportClassSplitCounts`, df01309). */
export interface ExportClassSplitCounts {
  /** Registry class id. */
  class_id: number;
  /** Dense class id written into the label files. */
  export_id: number;
  class_name: string;
  train: number;
  val: number;
  test: number;
}

/**
 * Server response from `GET {API_PREFIX}/export/status` (OpenProcessor
 * `ExportStatusResponse`, df01309). `status` is `'idle' | 'unknown' |
 * 'success'` on the vendored df01309 backend — the GET endpoint now only
 * ever describes the *last completed* export (`idle` = none yet, every
 * other field null; `unknown` = the `current` manifest is missing/
 * unreadable). The `progress`/`job_id`/`error` fields below are NOT part
 * of that response; they're kept only because `/export`'s `runExport()`
 * synthesizes an `ExportStatus`-shaped object from `POST /export/yolo`'s
 * synchronous `ExportResult` response (a different endpoint, still
 * `running`/`failed`/`success`-capable) and assigns it to the same
 * `exportState` variable. Every field below `status` is optional/nullable
 * because the served `ExportStatusResponse` requires only `status` (an
 * idle status, with no export yet, carries none of them).
 */
export interface ExportStatus {
  status: string;
  last_run: string | null;
  /** 0..1 for active jobs (synthesized from `ExportResult` only — the GET
   *  response never carries this). */
  progress?: number;
  job_id?: string | null;
  /** On success: directory the manifest was written to. Same as `path`. */
  export_dir?: string | null;
  /** On failure: the error message (synthesized from `ExportResult`
   *  only — the GET response never carries this). */
  error?: string | null;
  /** Optional human-readable detail. */
  message?: string | null;
  /** Resolved export directory — same value as `export_dir`. */
  path?: string | null;
  version_tag?: string | null;
  dataset_sha?: string | null;
  seed?: number | null;
  /** Row attribute the split grouped on (`'image_id'`). */
  group_key?: string | null;
  /** Exported images (OpenProcessor 4c9499a: one image + one label file
   *  per source image, one line per object). */
  image_count?: number | null;
  /** Exported objects (label lines) across all images. */
  object_count?: number | null;
  class_count?: number | null;
  /** Of `class_count` registry classes, how many have >=1 object (#36
   *  item 6). `null` for an export written before it was recorded. */
  classes_with_objects?: number | null;
  /** Images per split. */
  split_counts?: ExportSplitCounts | null;
  /** Objects (label lines) per split. */
  split_object_counts?: ExportSplitCounts | null;
  /** Objects per class per split. `null` for an export written before
   *  this was recorded. */
  class_split_counts?: ExportClassSplitCounts[] | null;
  /** Whether `POST /export/yolo` would run now (an unfiltered export);
   *  `null` when the cohort count failed. Only `false` blocks. */
  can_export?: boolean | null;
  /** Why not, in the words the export 422 uses; empty when exportable. */
  blocking_reasons?: string[];
  /** Whether images with an unlabeled object were left out. */
  require_fully_labeled_images?: boolean | null;
  /** Objects on exported images the export did not label (unreviewed, or
   *  on a class it leaves out); learned as background. */
  unlabeled_items_on_exported_images?: number | null;
  /** Exported images holding at least one unlabeled object. */
  images_with_unlabeled_items?: number | null;
  /** Images left out by `require_fully_labeled_images` (0 when it was
   *  off). */
  images_dropped_not_fully_labeled?: number | null;
  /** Validated items the export couldn't write, by reason. Null for an
   *  export made before OpenProcessor 536e000. */
  skipped_items?: ExportSkippedItems | null;
}

/** `ExportSkippedItems` (OpenProcessor 536e000). */
export interface ExportSkippedItems {
  no_image_id: number;
  no_usable_box_or_class: number;
}

/**
 * Server response from `POST {API_PREFIX}/export/yolo` — synchronous
 * (verified live against OpenProcessor's `export_yolo`, `src/routers/curation/export.py`:
 * `service.export_yolo(...)` is `await`ed before the handler returns).
 * There is no `job_id`/queued state on this endpoint at all — the
 * response already carries the finished export's own fields. M13
 * (2026-09-24 interactive pass): the dashboard used to read `job_id`
 * (always undefined) and toast "Export job started", which never
 * matched what actually happened.
 */
export interface ExportResult {
  status: string;
  export_dir?: string | null;
  version_tag?: string | null;
  manifest_path?: string | null;
  dataset_sha?: string | null;
  /** Images per split. */
  split_counts?: Record<string, number> | null;
  /** Exported images. */
  image_count?: number | null;
  /** Exported objects (label lines) across all images. */
  object_count?: number | null;
  /** Objects (label lines) per split. */
  split_object_counts?: Record<string, number> | null;
  /** Echoed from the request — whether images with an unlabeled object
   *  were left out. */
  require_fully_labeled_images?: boolean | null;
  /** Objects on exported images the export did not label; learned as
   *  background. */
  unlabeled_items_on_exported_images?: number | null;
  /** Exported images holding at least one unlabeled object. */
  images_with_unlabeled_items?: number | null;
  /** Images left out by `require_fully_labeled_images`. */
  images_dropped_not_fully_labeled?: number | null;
  /** Validated items the export couldn't write, by reason. */
  skipped_items?: ExportSkippedItems | null;
  dedup?: number | null;
  started_at?: string | null;
  finished_at?: string | null;
  /** @deprecated Never served by POST /export/yolo — kept only so a
   *  caller written against the old shape still type-checks. */
  job_id?: string | null;
  message?: string | null;
}

/** Server response from `POST {API_PREFIX}/export/single_class`. */
export interface SingleClassExportResult {
  status: string;
  export_dir: string;
  version_tag?: string | null;
  manifest_path: string;
  data_yaml_path: string;
  dataset_sha: string;
  frozen_test_sha?: string | null;
  split_counts: Record<string, number>;
  image_count: number;
  class_count?: number | null;
  positive_images?: number | null;
  background_images?: number | null;
  positives_zero_warning?: boolean | null;
  current_symlink?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
}

/** Server response from `GET {API_PREFIX}/export/single_class/status`. */
export interface SingleClassExportStatus {
  status: 'idle' | 'unknown' | 'success' | string;
  profile_name: string;
  last_run: string | null;
  export_dir?: string | null;
  dataset_kind?: string | null;
  dataset_sha?: string | null;
  frozen_test_sha?: string | null;
  class_count?: number | null;
  class_names?: string[] | null;
  image_count?: number | null;
  positive_images?: number | null;
  background_images?: number | null;
  false_positive_background_images?: number | null;
  positives_zero_warning?: boolean | null;
  split_counts?: Record<string, number> | null;
}

/** One materialized dataset version from `GET {API_PREFIX}/export/datasets`. */
export interface ExportDataset {
  /** The `/methods` export-axis id that produced it: `yolo` (multi-class)
   *  or `single_class`. */
  kind: string;
  /** Narrowed exports only — which single-class profile wrote it. */
  profile_name?: string | null;
  export_dir: string;
  version_tag: string;
  image_count?: number | null;
  /** Exported objects (label lines) across all images. */
  object_count?: number | null;
  split_counts?: Record<string, number> | null;
  dataset_sha?: string | null;
  exported_at?: string | null;
  image_mode?: string | null;
  img_max_side?: number | null;
  sampling?: string | null;
  max_positive_images?: number | null;
  max_images?: number | null;
  class_count?: number | null;
  /** Of `class_count` registry classes, how many have >=1 object (#36
   *  item 6). */
  classes_with_objects?: number | null;
  is_current: boolean;
}

/** Server response from `GET {API_PREFIX}/export/datasets`. */
export interface ExportDatasetList {
  datasets: ExportDataset[];
  count: number;
}

/**
 * Server response from `POST {API_PREFIX}/test_holdout/freeze`. As of
 * OpenProcessor df01309 the request body is `{percent}` only — no
 * `seed` (selection is deterministic, SHA1-of-`crop_id` per class; an
 * unknown field like `seed` is now a 422, not silently ignored).
 */
export interface TestHoldoutFreezeResult {
  n_frozen: number;
  n_classes_covered: number;
  test_holdout_sha: string;
  per_class_counts: Record<string, number>;
  /** Selection method name — `'sha1_per_class'`. */
  selection: string;
  /** Target holdout percent per class, echoed from the request. */
  percent: number;
  /** Floor per class — all of a class smaller than this is frozen. */
  min_per_class: number;
}

/** Server response from `GET {API_PREFIX}/test_holdout/stats`. */
export interface TestHoldoutStats {
  total: number;
  by_class: Array<{ key: number; doc_count: number; deficient: boolean }>;
  /** The same class-adequacy threshold served on `/classes`/`/stats/classes`
   *  — the frontend's "below 5 test crops" copy reads this, never a
   *  hardcoded 5. */
  min_test_per_class: number;
}
