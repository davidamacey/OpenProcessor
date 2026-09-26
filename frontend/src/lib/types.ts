/**
 * TypeScript types mirroring the OpenSearch indexes defined in Wave 1d
 * (the items, classes and clusters indexes) and the OpenProcessor
 * `{API_PREFIX}/...` endpoint responses defined in Phase 2D.
 *
 * These shapes are forward-tolerant: we accept extra fields silently so a
 * server-side schema bump won't break the app.
 */

import type { SlotKey, SlotData } from './annotations/types';

/** Who wrote a crop's current label. Same vocabulary as `class_source`
 *  (curation_api_contract.md "class_source values"): `human*`, the fixed
 *  VLM writer values, or an ingest detector's config-derived value — so
 *  it stays an open string. */
export type LabelSource =
  'human' | 'human_confirmed' | 'vlm' | 'vlm_human_confirmed' | (string & {});

export type ClassSource = 'registry' | 'derived' | 'imported';

export interface RegistryClass {
  id: number;
  name: string;
  group: string | null;
  count: number;
  validated_count: number;
  /**
   * Size of the FAISS cluster bucket whose id == this class id. Includes
   * unlabeled candidates the operator hasn't triaged yet — i.e. matches
   * what the operator sees when they open /clusters/{id}. The sidebar
   * chip displays this; the cluster-page banner breaks out validated /
   * labeled / total side-by-side.
   */
  cluster_size: number;
  /** Single-character keyboard shortcut, persisted in the registry. */
  hotkey_letter?: string | null;
  added_at: string;
  /** Hex color hint or null. */
  color?: string | null;
  /** True when the class has been merged into another and should be hidden by default. */
  deprecated?: boolean;
  /** OpenProcessor 51b05d7: the class this one was merged into (its crops
   *  now carry that class). Null/absent for a class never merged. A merged
   *  class can't be restored (restore 409s `class_merged`). */
  merged_into?: number | null;
  /** Server-computed adequacy tier from `GET {API_PREFIX}/classes`
   *  (`block` | `warn` | `ok`, against the served `thresholds`). Never
   *  recomputed client-side from `validated_count`. */
  adequacy?: string;
  /** `'item'` (an ordinary item class) or `'region'` (a slot-bound
   *  region class, i.e. the served profile's region_class_name) — served on
   *  `GET {API_PREFIX}/classes`/`/stats/classes` (OpenProcessor #36 item 1,
   *  X2/R1). The single source of truth for excluding a region class from
   *  an item-class picker; `isSlotBoundClass`/`isItemClassTarget`
   *  (`$lib/classVisibility`) read this first, falling back to the slot
   *  registry only for a class an older backend doesn't tag. */
  kind?: 'item' | 'region';
  /** Validated crops usable for training — `sample_count`/`validated_count`
   *  no longer include region counts as of #36 (X2), so this is the
   *  server's own trainable count for the class, not client math. */
  trainable?: number;
  /** On `GET {API_PREFIX}/classes`: the shortfall of `trainable` against
   *  the served per-class hard minimum (`thresholds.block_below`), floored
   *  at 0 — served directly. > 0 means the class blocks preflight. */
  trainable_gap?: number;
}

/** `thresholds` served on `GET {API_PREFIX}/classes`, `GET {API_PREFIX}/stats/classes`
 *  and `{API_PREFIX}/train/preflight` — the single source of truth for the
 *  adequacy tiers, augmentation target range and test-holdout minimum. */
export interface ClassThresholds {
  block_below: number;
  warn_below: number;
  min_test_per_class: number;
  aug_target_min: number;
  aug_target_max: number;
}

/** `GET {API_PREFIX}/classes` response envelope. */
export interface ClassesResponse {
  classes: RegistryClass[];
  thresholds: ClassThresholds;
  /** Every single-character combo reserved for a labeling action —
   *  core keys plus every registered queue-capable slot's keymap. */
  reserved_hotkeys: string[];
}

/** Payload for `POST {API_PREFIX}/classes`. */
export interface RegistryClassCreate {
  name: string;
  group: string;
  notes?: string;
}

/** Payload for `PUT {API_PREFIX}/classes/{id}`. Any subset of fields may be supplied. */
export interface RegistryClassUpdate {
  name?: string;
  group?: string;
  /** Pass an empty string to clear the binding, or omit to leave unchanged. */
  hotkey_letter?: string;
}

/** Payload for `POST {API_PREFIX}/classes/merge`. */
export interface RegistryClassMerge {
  source_id: number;
  target_id: number;
}

/** `POST {API_PREFIX}/classes/merge?dry_run=true` response — reports
 *  counts and writes nothing. A real merge 409s when `holdout_blocking > 0`. */
export interface ClassMergeDryRun {
  dry_run: true;
  source_id: number;
  target_id: number;
  would_relabel: number;
  /** OpenProcessor 51b05d7: human validations KEPT on the relabeled crops
   *  (a merge now carries validations over; was `would_unvalidate`). */
  validations_carried_over: number;
  holdout_blocking: number;
  blocked: boolean;
}

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
 * so a pre-df01309 backend's GET response (missing all of them) renders
 * exactly as it did before — no page break on a missing field.
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
   *  item 6). `null` for an export written before it was recorded, or a
   *  backend that predates the field entirely. */
  classes_with_objects?: number | null;
  /** Images per split. */
  split_counts?: ExportSplitCounts | null;
  /** Objects (label lines) per split. */
  split_object_counts?: ExportSplitCounts | null;
  /** Objects per class per split. `null` for an export written before
   *  this was recorded. */
  class_split_counts?: ExportClassSplitCounts[] | null;
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
 * unknown field like `seed` is now a 422, not silently ignored) — and
 * the response gained `selection`/`min_per_class` (both required on
 * df01309; optional here so a pre-df01309 backend's response, which
 * doesn't serve them, still type-checks and renders without them).
 */
export interface TestHoldoutFreezeResult {
  n_frozen: number;
  n_classes_covered: number;
  test_holdout_sha: string;
  per_class_counts: Record<string, number>;
  /** Selection method name — `'sha1_per_class'` on df01309. */
  selection?: string;
  /** Target holdout percent per class, echoed from the request. */
  percent?: number;
  /** Floor per class — all of a class smaller than this is frozen. */
  min_per_class?: number;
}

/** Server response from `GET {API_PREFIX}/test_holdout/stats`. */
export interface TestHoldoutStats {
  total: number;
  by_class: Array<{ key: number; doc_count: number; deficient?: boolean }>;
  /** The same class-adequacy threshold served on `/classes`/`/stats/classes`
   *  — the frontend's "below 5 test crops" copy reads this, never a
   *  hardcoded 5. */
  min_test_per_class?: number;
}

export interface BBoxNorm {
  cx: number;
  cy: number;
  w: number;
  h: number;
}

/** One OCR text line on the item crop (wire `ItemTextLine` —
 *  `box_norm` is already normalized in the item-crop frame, `rel_height`
 *  is line height / crop height). Populates `Crop.item_text_lines`. */
export interface ItemTextLine {
  text: string | null;
  confidence: number | null;
  box_norm: number[] | null;
  rel_height: number | null;
}

export interface Crop {
  id: string;
  source_image_path: string;
  source_image_sha256?: string;
  bbox_norm: BBoxNorm;
  class_id: number | null;
  class_name: string | null;
  /** Where the class assignment came from — `human*`, the fixed VLM
   *  writer values (`vlm`, `vlm_unmatched`, …), or an ingest detector's
   *  config-derived value (`{primary}_proposal`, `{secondary}_model`, …).
   *  Drives the per-source filter chip in the cluster view. */
  class_source: string | null;
  label_source: LabelSource;
  /** `class_validated OR region_validated` (wire's `label_validated`,
   *  `wire.py:105`). "Anything on this crop was validated" — NOT the
   *  same as the class label being trustworthy. Read `class_validated`
   *  for any class-label display/eligibility check; keep this only for
   *  an intentional "anything validated" meaning (G2). */
  label_validated: boolean;
  /** Whether the *class* assignment specifically was human-validated.
   *  Independent of region_validated — a crop can be
   *  `label_validated=true` (region validated) while `class_validated`
   *  is still false, e.g. a v6-model class label on a region a human
   *  confirmed has no visible region. */
  class_validated: boolean;
  label_confidence: number | null;
  /** Confidence of whoever set the *class label* (wire `class_confidence`,
   *  dq-queues cutover 2026-09-24) — distinct from `label_confidence`
   *  (the detector/classifier score). VLM high/medium/low map to
   *  0.92/0.70/0.40 server-side; classifier labels carry their own
   *  score; human labels are null. */
  class_confidence?: number | null;
  /** `'vlm' | 'model' | null` — which pipeline produced `class_confidence`. */
  class_confidence_source?: string | null;
  /** The VLM's verbatim class answer, even when unmatched/unapplied. */
  vlm_raw_class?: string | null;
  vlm_class_attempted_at?: string | null;
  /** `no_answer | no_match | invalid_index | unparseable | null`. */
  vlm_class_empty_reason?: string | null;
  /** The VLM's registry-matched class for this crop when it did not
   *  auto-apply it (wire `vlm_proposed_class_id`/`_name`). Drives the
   *  accept-suggestion chip and the `G`/`Shift+Enter` keys. */
  vlm_suggested_class_id?: number | null;
  vlm_suggested_class_name?: string | null;
  /** The VLM's categorical confidence: `high` | `medium` | `low`. */
  vlm_confidence?: string | null;
  /** Cluster + similarity-to-centroid (0..1), served verbatim as
   *  `cluster_similarity` — no client 1−cosine-distance computation. */
  cluster_id: number | null;
  similarity_to_centroid: number | null;
  /** Distance to this crop's cluster centroid, null when the item has
   *  left the cluster it's measured against (dq-queues cutover). */
  cluster_distance?: number | null;
  /** Nearest cluster centroid id, only meaningful together with
   *  cluster_distance/similarity_to_centroid/cluster_is_core. */
  cluster_nearest_id?: number | null;
  /** Server-computed: this crop's `cluster_similarity` is at/above the
   *  cluster response's `core_similarity_min`. Null when the backend
   *  hasn't computed a similarity for this crop yet. Drives the
   *  cluster-detail "core" cut line — never a client-side 0.75 constant. */
  cluster_is_core?: boolean | null;
  /** AHC sub-cluster id (string, e.g. "47a"), populated only after
   *  refine ran. Cleared by the backend whenever cluster_id changes
   *  (move / batch_label / class_merge / residual recluster) — the
   *  subid is meaningful only inside its origin cluster. */
  cluster_subid: string | null;
  /** Per-slot capability data, keyed by SlotKey, produced by mapRawCrop
   *  via mapCropSlots(). Only slots with actual evidence on the row
   *  appear (slotIsPresent). Optional because many test fixtures
   *  construct Crop literals directly; read it through
   *  `slotOf(crop, spec)` (annotations/cropSlots.ts), never `?.[...]`
   *  by hand, so the accessor can be tightened later. */
  slots?: Record<SlotKey, SlotData>;
  // -- Class provenance (Wave 1) -----------------------------------------
  class_detector?: string | null;
  class_detector_version?: string | null;
  class_labeled_at?: string | null;
  class_labeler?: string | null;
  /** Ingest source tag (`GET {API_PREFIX}/review/{tab}?source=` /
   *  `GET {API_PREFIX}/crops?source=`). Replaces the old `hdd_source` field the
   *  backend never actually populated (2026-09-24 logic-moves cutover) —
   *  `source` is the live wire key. `?hdd_source=` itself was removed from
   *  `GET {API_PREFIX}/crops` by the OpenProcessor 1327181 naming sweep (F9); every
   *  query-param site uses `?source=` now too (see `CropFilter.source` below). */
  source?: string | null;
  /** The backend's confirmable suggestion for this crop — what
   *  Enter/Confirm assigns. Served on every crop-shaped item, not just
   *  review-queue rows (undo/restore, `/crops/{id}`, diverse selection
   *  hydration all carry it too), so it lives on `Crop` rather than only
   *  `ReviewItem`. */
  proposed_class_id: number | null;
  proposed_class_name: string | null;
  test_holdout: boolean;
  // -- Exclude / Ignore (G7) ----------------------------------------------
  /** True when the crop is in the excluded/"Ignored" bucket
   *  (`cluster_id=-2`) — set by `POST /crops/batch_exclude`, cleared by
   *  `POST /crops/batch_unexclude`. */
  class_excluded?: boolean;
  /** Free-text tag recorded at exclude time (`'ignore'`, `'blurry'`, …). */
  excluded_reason?: string | null;
  excluded_at?: string | null;
  /** Item-crop-frame OCR text lines (G-series item text; wire
   *  `item_text_lines`, `ItemTextLine[]`). Empty array when the item has
   *  no detected text, never absent. */
  item_text_lines?: ItemTextLine[];
  outlier_flagged?: boolean;
  outlier_score?: number | null;
  // -- Primary-subject rank + blur quality -------------------------------
  /** 1 = largest-area crop in its source photo, 2 = second, … Drives the
   *  "Largest / Largest + 2nd" subject toggle. */
  crop_rank_in_image?: number | null;
  /** Normalized bbox area (w*h in [0,1]). */
  crop_area_norm?: number | null;
  /** Crop/full Laplacian ratio (higher = clearer).
   *  Drives the clarity slider. */
  blur_lap_ratio?: number | null;
  /** Raw v6 detection confidence (recorded even below the 0.75 floor). */
  classifier_raw_confidence?: number | null;
  /** Coarse COCO class hint for coco_yolo11_proposal blind spots. */
  proposal_name?: string | null;
  // -- Curation scores (Phase 3 review-queue strategies, 2026-09) --------
  // Provenance quad mirroring the region_detector/region_detector_version
  // pattern (docs/curation-strategy-plan-2026-09.md §4). `mistakenness`
  // is the only curation score that cleared the full validation gate as
  // of OpenProcessor docs/design/curation_scores.md — representativeness/
  // atypicality/uncertainty_entropy are pre-existing fields (cluster
  // distance, probe entropy) exposed as named sorts, not new score
  // fields, so they don't need their own Crop fields. Everything else
  // in the plan (uniqueness, near-dup) is still shadow/pending and has
  // no field here yet — added when/if it clears validation.
  mistakenness_score?: number | null;
  mistakenness_method?: string | null;
  mistakenness_version?: string | null;
  mistakenness_scored_at?: string | null;
  /**
   * F8 D1 (OpenProcessor 51b05d7): the probe's opinion on this item.
   * `probe_in_scope`: null = not scored yet, true = the item's class is
   * one the probe knows (a real opinion), false = outside the probe's
   * classes (no opinion; `probe_disagreement` is then null).
   * `probe_disagreement`: true/false only when the probe has an opinion.
   */
  probe_disagreement?: boolean | null;
  probe_in_scope?: boolean | null;
  probe_model_version?: string | null;
  /**
   * OpenProcessor main 8990ede: `true` only when the probe disagrees AND
   * the item is in scope AND `probe_pred_confidence` cleared the
   * backend's own threshold (`OP_PROBE_ACTIONABLE_MIN_CONFIDENCE`,
   * echoed read-only as `ProbeStatusResponse.actionable_min_confidence`
   * — no client-side threshold anywhere). `false` when scored but not
   * actionable (including every non-disagreeing item); `null` when not
   * scored at all.
   */
  probe_actionable?: boolean | null;
  updated_at: string;
}

/** Class clusters mirror class_id (0..80); candidate clusters land at
 *  10000+ from the residual AHC pass; unassigned is < 0. The region view
 *  also emits "false_positive" for the permanent FP bucket (-100). The
 *  backend derives this from cluster_id; the frontend NEVER recomputes it. */
export type ClusterKind = 'class' | 'candidate' | 'unassigned' | 'false_positive';

/** Backend's purity banding (`purity_thresholds`: pure_min 0.85 / mixed_min
 *  0.6) — served per cluster, never recomputed against a client constant. */
export type PurityTier = 'pure' | 'mixed' | 'noisy';

export interface Cluster {
  id: number;
  /** Backend-derived: "class" | "candidate" | "unassigned". */
  cluster_kind: ClusterKind;
  size: number;
  /** class_validated=true count. */
  validated_count: number;
  dominant_class_id: number | null;
  dominant_class_name: string | null;
  /** Served `label_purity` — the dominant class's share of the
   *  LABELLED members. Not the geometry `purity`. */
  dominant_pct: number | null;
  /** Served member counts behind `dominant_pct` (cluster-scoped, include
   *  any test-holdout members). Optional: absent on older responses. */
  dominant_count?: number | null;
  labelled_count?: number | null;
  /** DQ-M2 fix (dq-queues cutover, 2026-09-24): nearest-centroid geometry
   *  purity — the share of `purity_n` measured members whose nearest
   *  cluster centroid is this cluster's own. Independent of labels, so a
   *  class cluster is no longer 1.0 by construction. 0..1, null when no
   *  member has been measured. */
  purity: number | null;
  /** How many members `purity` was measured over — show alongside
   *  `purity` (it's noisy at low n). */
  purity_n?: number | null;
  /** Always `'nearest_centroid'` today — served, never hardcoded. */
  purity_basis?: string | null;
  /** Server-banded purity — drives the card's pure/mixed/noisy badge and
   *  border color. Null only for the client-only "cluster card lookup
   *  failed" stub in `getCluster`. */
  purity_tier: PurityTier | null;
  /** Largest-class share among labelled members (the pre-cutover
   *  label-based "purity" — 1.0 for a class cluster by construction).
   *  `promotable` is gated on this, not the geometry-based `purity`. */
  label_purity?: number | null;
  /** Share of this cluster's members that carry any label at all. */
  labelled_share?: number | null;
  /** Server's auto-promote eligibility for this cluster (purity +
   *  member-count + labelled-share gate — `purity_thresholds` on the
   *  `{API_PREFIX}/clusters` response). */
  promotable: boolean;
  /** `{API_PREFIX}/clusters`' `core_similarity_min` (0.75), copied onto every
   *  cluster in that response so a consumer doesn't need the raw list
   *  response's top-level field. Null when the lookup that would have
   *  supplied it failed (see `getCluster`'s stub). */
  core_similarity_min: number | null;
  /** True when no member has a class_name — pure candidate. */
  is_unlabeled: boolean;
  representative_crop_ids: string[]; // up to 4
  // Optional explicit thumbnail URLs (one per representative_crop_ids
  // entry, same order). Used by the synthetic region card so its
  // tiles show region close-ups (API_PREFIX-relative /crops/{id}/region_thumbnail) rather
  // than the default item-crop thumbnail. Regular clusters leave
  // this undefined; the grid then falls back to getThumbUrl().
  representative_thumb_urls?: string[];
  has_subclusters: boolean;
  /** Distinct cluster_subid count from the backend. */
  n_subclusters: number;
  /** Legacy alias for n_subclusters — kept until callers migrate. */
  sub_clusters?: number;
  centroid_sha?: string;
  updated_at: string | null;
  /** Set only on the client-built region inventory entry pinned
   *  atop `/clusters` (M4, docs/design/interactive-pass-2026-09-24.md) —
   *  it is not a real cluster (no purity, no cluster_kind), so the grid
   *  must not draw a pure/mixed/noisy badge for it or key it against a
   *  real cluster id. Absent (not false) on every server-served Cluster. */
  isSlotCard?: boolean;
  /** The slot's served display name (e.g. its tab label), set only on
   *  that inventory entry (F8 D7): the card is titled by it, not by the
   *  raw class id. */
  slotDisplayName?: string;
}

export interface StatsSummary {
  /** Set when `/stats/dataset` failed; the totals below are then zeros,
   *  not real counts. `per_class` comes from `/stats/classes` and stays. */
  dataset_error: string | null;
  total_crops: number;
  validated_crops: number;
  test_holdout_crops: number;
  ingestion: {
    images_processed: number;
    images_pending: number;
    last_run_at: string | null;
  };
  per_class: Array<{
    class_id: number;
    class_name: string;
    count: number;
    validated_count: number;
    /** Server-computed adequacy tier (`block`/`warn`/`ok`) — see
     *  `RegistryClass.adequacy`. */
    adequacy?: string;
    /** Server-computed YOLO augmentation target for this class. */
    aug_target?: number;
    /** `aug_target - validated_count`, served directly. */
    aug_gap?: number;
    /** Validated crops usable for training (region counts excluded, #36
     *  X2) — the server's own trainable count, not client math. */
    trainable?: number;
    /** `aug_target - trainable`, served directly (#36 item 1). */
    trainable_gap?: number;
  }>;
  /** Served alongside `per_class` on `/stats/classes` — same shape as
   *  `ClassesResponse.thresholds`. */
  thresholds?: ClassThresholds;
}

/** `GET {API_PREFIX}/health`. `degraded` means a non-critical
 *  dependency (e.g. the VLM) is down; labeling still works. */
export interface ApiHealth {
  status: 'ok' | 'degraded' | 'down';
  triton?: { reachable: boolean; detail?: string };
  opensearch?: { reachable: boolean; indexes?: Record<string, boolean> };
  vlm?: { reachable: boolean; model?: string | null };
  registry?: { path?: string; exists?: boolean; mtime?: string | null };
  /** The backend's active region profile, or `null` when none is
   *  configured (then every region route answers 409). Absent on a
   *  backend older than OpenProcessor naming-w2, which the UI treats the
   *  same as `null`. The only signal region features key on. */
  region_profile?: ServedRegionProfile | null;
}

/** `RegionProfileSummary` on `GET {API_PREFIX}/health` and
 *  `GET {API_PREFIX}/regions/vocabulary`. */
export interface ServedRegionProfile {
  /** Profile id; the region slot's key. */
  name: string;
  /** The region noun shown to operators (tab label, gallery copy). May be
   *  empty when the profile doesn't set one. */
  display_name: string;
  /** The singular form of the region noun, for a singular-context label
   *  ("Confirm <Region>", "<Region> score"). May be empty — falls back
   *  to the generic "Region" (OpenProcessor #36 item 10, 2026-09-25). */
  display_name_singular: string;
  /** The class whose items ARE regions (e.g. the class a region export
   *  writes). May be empty. */
  region_class_name: string;
  /** How region text is read (`vlm` / `ocr` / `vlm_then_ocr` / `both`),
   *  or `'none'` for a text-free profile (OpenProcessor W1,
   *  2026-09-26) — a profile that detects/segments a region but never
   *  reads text off it. */
  text_reader: string;
  /** False for a text-free profile: no region text is read, stored or
   *  editable — `region_text*` stays null and a `region_meta` PATCH
   *  carrying `region_text` 422s (OpenProcessor W1). Optional/undefined
   *  on a pre-W1 backend; callers fall back to `text_reader` being
   *  non-empty and not `'none'` (`profileReadsText()`,
   *  `servedRegionSlot.ts`). */
  reads_text?: boolean;
  /** Whether the OCR text-hint re-pass (after a segmenter miss) is
   *  enabled for this profile. Optional/undefined on a pre-W1 backend;
   *  informational only today — no UI reads it yet (see CLAUDE.md). */
  text_hint_enabled?: boolean;
}

// 'outliers' was retired from the UI in the 2026-09 tab consolidation
// (3 live rows, functionally identical to the `atypicality` sort already
// offered via the strategy bar) — its backend {API_PREFIX}/review/outliers query
// is untouched, but nothing in the frontend calls it anymore, so the
// literal is gone from this union too. 'mismatches' / 'vlm_low_conf' /
// 'primary_low_conf' are no longer top-level UI tabs but still real
// values here — they're driven by the All-tab preset chips instead (see
// $lib/reviewTabs.ts's resolveEffectiveTab).
// A queue-capable slot's tab id is the structural `slot:${SlotKey}`
// template (SlotReviewTab), not a hand-maintained literal per slot. The
// slot's own `urlId` is the bookmark value (see `$lib/reviewTabs.ts`'s
// `tabFromUrlId`), not an internal id.
export type SlotReviewTab = `slot:${string}`;

export type CoreReviewTab =
  | 'all'
  | 'mismatches'
  | 'vlm_low_conf'
  | 'uncertainty'
  | 'model_disagreements'
  | 'primary_low_conf'
  | 'classifier_blind_spots'
  // New-class-proposal queue (2026-09-24 logic-moves W5): crops the VLM
  // couldn't match to any registered class (`class_source:
  // 'vlm_new_class_pending'`). Backed by the same `{API_PREFIX}/review/{tab}`
  // shape as every other core tab; `/classes`'s Proposals section reads
  // the separate `.../summary` aggregate instead (see api.ts).
  | 'new_class_proposals';

export type ReviewTab = CoreReviewTab | SlotReviewTab;

export interface ReviewItem extends Crop {
  /** m2 (2026-09-24 interactive pass): null after an undo re-insert when
   *  the controller's removed-item cache doesn't have the original
   *  served reason (e.g. a page reload between removal and undo) —
   *  never a frontend-invented string like "restored by undo". */
  reason: string | null;
  // proposed_class_id/name live on Crop now (served on every crop-shaped
  // item, not just review rows) — not re-declared here.
  /** The freshly-promoted model's predicted class id for this crop
   *  (`model_disagreements` tab). Lets "Accept model's class" call
   *  `assign()` directly instead of needing a name→id lookup. */
  probe_pred_class_id?: number | null;
  /** Set when a human (or the VLM) flagged this crop as needing a class
   *  the registry doesn't have yet — drives the `new_class_proposals`
   *  queue and the note shown inline. */
  needs_new_class?: boolean;
  needs_new_class_note?: string | null;
  // Phase 5 — populated for the model_disagreements tab. The new model's
  // prediction for this crop, plus how confident it was. Lets the
  // labeler render "human said X, model said Y" inline.
  probe_pred_class?: string | null;
  probe_pred_entropy?: number | null;
}

/**
 * A crop returned by `GET {API_PREFIX}/search/text` (P2-14 semantic text search).
 * Same shape as `Crop` plus the query-similarity score. Field named
 * `similarity_score` to match this repo's existing `<name>_score`
 * convention for per-crop curation scores (`mistakenness_score`), not a
 * bare `score` — kept distinct from those since it's query-relative, not
 * an absolute crop property.
 */
export interface SearchCrop extends Crop {
  similarity_score: number;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  /** Set by `{API_PREFIX}/review/{tab}` when the requested `?sort=` couldn't be
   *  honored (e.g. the field isn't backfilled yet) and the server fell
   *  back to the default ordering. Rendered as an inline note, never a
   *  toast — this isn't an error, just a degraded request. Absent on
   *  every other endpoint and on a request that didn't ask for a sort. */
  sort_fallback_reason?: string | null;
  /** The sort id `{API_PREFIX}/review/{tab}` actually used — the tab's own
   *  default when no `?sort=` was sent, or the requested id when it was
   *  honored. Shown in the StrategyBar summary chip so the operator can
   *  see what's actually ordering the queue, not just what they last
   *  picked (2026-09-24 logic-moves W5). Absent on endpoints that don't
   *  report it. */
  sort_applied?: string | null;
  /** Set by `{API_PREFIX}/review/{tab}` when `total === 0` — the server's
   *  own explanation for why this queue is empty right now (e.g. "no
   *  probe predictions — run a probe"), distinct from and more direct
   *  than `sort_fallback_reason` (OpenProcessor #36 item 9). Absent when
   *  the queue isn't empty, or on a backend that predates the field. */
  empty_reason?: string | null;
  /** Provenance for a pool-scale overlay ordering (curation-strategy plan
   *  Phase 4 — currently only `{API_PREFIX}/crops?order=diverse`): which
   *  overlay/version produced this selection, and how large the pool it
   *  drew from was. `n_pool` lets the UI note when diverse selection is
   *  sampling a much larger cohort than what's on screen, rather than
   *  silently truncating. All three are optional/tolerant — absent on
   *  every other order value and on any backend that hasn't shipped this
   *  yet (Phase 0/3 backend, or `OP_SELECT_DIVERSE_ENABLED` off). */
  order_method?: string | null;
  order_version?: string | null;
  n_pool?: number | null;
  /** m7 (2026-09-24 interactive pass): `{API_PREFIX}/clusters`' own
   *  `purity_thresholds` — the border-color legend on `/clusters` used
   *  to hardcode "≥80% / ≥60% / <60%", which drifted from the server's
   *  real `pure_min`/`mixed_min` (0.85/0.6). Absent on every other
   *  endpoint. */
  purity_thresholds?: { pure_min: number; mixed_min: number } | null;
}

/**
 * `POST {API_PREFIX}/select/diverse`'s scope object (P2-10, `/review`'s diverse
 * overlay — docs/CLAUDE.md's curation-strategy-selector-bar section).
 * Distinct from `/clusters/[id]`'s `GET {API_PREFIX}/crops?order=diverse&k=N` —
 * that path is a small, synchronous, cluster-scoped selection; this one
 * scopes to a review-tab cohort (`review_tab` reuses the backend's
 * existing tab-query builder) which can be pool-scale (the `all` tab is
 * ~320k crops), hence the job/poll contract below. `filters` only
 * supports term/terms filters server-side (`class_id`, `source`) —
 * NOT `conf_min`/`conf_max`/`min_blur_ratio`/`max_rank`/region `text`.
 */
export interface SelectDiverseScope {
  cluster_id?: number | null;
  review_tab?: string | null;
  filters?: Record<string, unknown>;
}

/**
 * Immediate (200) result of `POST {API_PREFIX}/select/diverse` for a small pool.
 * Forward-tolerant per this file's convention — an unrecognized `method`
 * string is still carried through as-is.
 */
export interface DiverseSelection {
  crop_ids: string[];
  method: string;
  version: string;
  n_pool: number;
}

/** Background-job state for a large-pool diverse selection (202 path),
 *  polled via `GET {API_PREFIX}/select/status`. CONFIRMED against the real backend
 *  (`selection/job.py`'s `_JobState`, 2026-09-12): `status` is one of
 *  `'idle' | 'running' | 'completed' | 'failed' | 'cancelled'`, and the
 *  finished selection is nested under `result` (the same
 *  `{crop_ids, method, version, n_pool}` shape the sync 200 path returns
 *  directly) — NOT flattened onto the job status object. `status` values
 *  this build doesn't recognize are carried through as-is rather than
 *  normalized — callers should treat anything other than `'running'` as
 *  "stop polling." */
export interface SelectJobStatus {
  job_id?: string | null;
  status: string;
  result?: DiverseSelection | null;
  error?: string | null;
}

export interface CropFilter {
  class_id?: number | null;
  cluster_id?: number | null;
  label_source?: LabelSource;
  /** Original label source (where the class came from). Matches the
   *  {API_PREFIX}/crops?class_source= query param. */
  class_source?: string;
  label_validated?: boolean;
  /** `GET {API_PREFIX}/crops?source=` — renamed off the removed `?hdd_source=` param
   *  by the OpenProcessor 1327181 naming sweep (F9). */
  source?: string;
  conf_min?: number;
  conf_max?: number;
  sort?: string;
  limit?: number;
  page?: number;
  // -- Primary-subject filters -------------------------------------------
  /** Keep only crops with crop_rank_in_image <= max_rank (1 or 2). */
  max_rank?: number | null;
  /** Clarity slider: keep crops with blur_lap_ratio >= this (null-safe). */
  min_blur_ratio?: number | null;
  /** Mine the low-confidence pool: classifier_raw_confidence < this OR no v6 box. */
  classifier_conf_lt?: number | null;
  /** Crops permanently dismissed from every /review queue via {API_PREFIX}/crops/{id}/review_dismiss. */
  review_dismissed?: boolean;
  /** Include excluded ("Ignored") crops — off by default server-side, so
   *  the "Ignored" bucket view is the only caller that sets this. */
  include_excluded?: boolean;
  /** Free-text OCR search over `item_text_lines` (G-series item text).
   *  A query with no letter/digit 400s server-side — see `getCrops`. */
  item_text?: string;
}

/** `GET {API_PREFIX}/crops/{id}/history` — the item's class history, oldest
 *  first. Each entry is the item's class state *before* one write, plus
 *  who made it (`writer`) and when (`at`). Untyped on the wire
 *  (`additionalProperties: true`) beyond `writer`/`at`, so every other key
 *  is read tolerantly. */
export interface CropHistoryEntry {
  writer: string | null;
  at: string | null;
  class_id?: number | null;
  class_name?: string | null;
  class_source?: string | null;
  label_source?: string | null;
  confidence?: number | null;
  class_detector?: string | null;
  class_detector_version?: string | null;
  class_labeler?: string | null;
  class_labeled_at?: string | null;
  class_validated?: boolean | null;
  cluster_id?: number | null;
  cluster_subid?: string | null;
  review_dismissed_at?: string | null;
  review_dismissed_by?: string | null;
}

export interface CropHistoryResponse {
  crop_id: string;
  entries: CropHistoryEntry[];
}

/** `GET {API_PREFIX}/crops/{id}/image` — the shared source image plus every
 *  item cropped from it (siblings, including the requested crop itself). */
export interface CropImageMeta {
  image_id: string;
  image_path: string;
  width: number | null;
  height: number | null;
  source: string | null;
  indexed_at: string | null;
}

export interface CropContextResponse {
  image: CropImageMeta;
  items: Crop[];
}

export interface ClusterFilter {
  class_id?: number | null;
  /** DQ-M4 (docs/design/data-quality-pass-2026-09-24.md): restrict to
   *  exactly one cluster, so its representatives can be fetched
   *  individually in the frontend's own (client-sorted) display order —
   *  the server has no `cluster_ids`/batch-by-id param, only a
   *  size-desc-ordered `offset`/`limit` window (see
   *  representatives_offset/_limit below), which is why representatives
   *  used to fill in server order regardless of what sort the operator
   *  picked. */
  cluster_id?: number | null;
  /** DQ-M4 live-verified follow-up: the backend's `offset`/`limit`
   *  representatives window rejects `limit=0` (422, "Input should be
   *  greater than or equal to 1") — confirmed live against real data,
   *  not just the vendored OpenAPI's `minimum: 1`. To genuinely skip
   *  representative computation on the main card-list call (superseded
   *  entirely by per-card display-order fetches, see
   *  displayOrderRepresentatives.ts), set this to 0 instead — `0`
   *  representative crops per card is valid (`minimum: 0`) and just as
   *  cheap, without touching offset/limit at all. */
  per_cluster?: number;
  min_size?: number;
  sort?: 'purity_asc' | 'purity_desc' | 'size_desc' | 'size_asc' | 'dominant_class';
  page?: number;
  page_size?: number;
  // Primary-subject grid filters: card stats reflect only crops that pass.
  max_rank?: number | null;
  min_blur_ratio?: number | null;
  class_source?: string | null;
  /** D-4 (docs/design/curation_query_performance_audit.md): which window
   *  of the (already-fully-returned, size-desc-ordered) card list gets a
   *  populated `representatives` array — cards outside the window come
   *  back with `representatives: []`, not omitted. Independent of
   *  `page`/`page_size` above, which the endpoint ignores entirely (every
   *  card up to `max_clusters` is always returned in one response); the
   *  caller windows representatives to whatever's actually visible. */
  representatives_offset?: number;
  representatives_limit?: number;
}

export interface BulkLabelConflict {
  crop_id: string;
  current_source: string | null;
}

export interface BulkLabelResult {
  // Matches the FastAPI handler at src/routers/curation/crops.py:
  //   batch_label_crops -> { updated: int, updated_ids: [...], conflicts: [...] }.
  // `move` (POST {API_PREFIX}/crops/move) returns the same shape.
  updated: number;
  updated_ids: string[];
  conflicts: BulkLabelConflict[];
}

/** `create` payload on `POST {API_PREFIX}/review/new_class_proposals/resolve` —
 *  register a brand-new registry class before resolving the term. Same
 *  slug rule as `RegistryClassCreate.name` (`class_name` here, to match
 *  the term the VLM proposed rather than an internal field name). */
export interface ResolveNewClassCreate {
  class_name: string;
  group?: string;
  notes?: string | null;
}

/** Body for `POST {API_PREFIX}/review/new_class_proposals/resolve` — bulk-
 *  resolve every pending `vlm_new_class_pending` item proposing `label`.
 *  Exactly one of `class_id` (map to an existing registry class) /
 *  `create` (register a new one first) must be set — the backend 422s
 *  otherwise. */
export interface ResolveNewClassRequest {
  label: string;
  class_id?: number | null;
  create?: ResolveNewClassCreate | null;
  label_source?: 'human' | 'human_confirmed' | 'new_class_proposal';
}

/** `POST {API_PREFIX}/review/new_class_proposals/resolve` response.
 *  `class_id` is null only under `?dry_run=true` when `create` was
 *  given (nothing was created yet to have an id). `?dry_run=true` also
 *  leaves `updated`/`updated_ids`/`conflicts`/`skipped` at their empty
 *  defaults — only `matched`/`matched_ids` are meaningful for a dry run. */
export interface ResolveNewClassResponse {
  class_id: number | null;
  class_name: string;
  created: boolean;
  label: string;
  matched: number;
  matched_ids: string[];
  updated: number;
  updated_ids: string[];
  conflicts: BulkLabelConflict[];
  skipped: string[];
}

/** `POST {API_PREFIX}/crops/label/undo_batch` response — batch form of
 *  `undoCropLabel`. Each crop is restored independently to its own state
 *  before its most recent human class write; `items` carries only the
 *  crops actually restored. */
export interface CropUndoBatchResult {
  items: Crop[];
  undone: number;
  nothing_to_undo: string[];
  conflicts: string[];
  not_found: string[];
}

/** `POST {API_PREFIX}/crops/region/undo_batch` response (M6) — same shape
 *  as `CropUndoBatchResult`, kept as its own type since it's a distinct
 *  wire contract (region writes, not class writes), not because the
 *  fields differ. */
export type CropRegionUndoBatchResult = CropUndoBatchResult;

export interface ToastMessage {
  id: string;
  kind: 'info' | 'success' | 'warn' | 'error';
  text: string;
  ttl_ms?: number;
}

export interface KeyboardShortcut {
  /** Every combo bound to this action/registration, in document order. */
  keys: string[];
  scope: string;
  description: string;
}

export type ModelStatus =
  'ready' | 'not_ready' | 'unavailable' | 'not_configured' | 'not_installed';
export type ModelKind = 'triton' | 'external';

export interface ModelInfo {
  name: string;
  friendly_name: string;
  role: string;
  kind: ModelKind;
  model_type: string;
  status: ModelStatus;
  version: string | null;
  inference_count: number | null;
  exec_count: number | null;
  inference_failed: number | null;
  avg_latency_ms: number | null;
  last_error: string | null;
  endpoint: string | null;
  /**
   * Follow-up gap 2 (docs/design/audit-remediation-plan-2026-09.md
   * Appendix D item 3, 2026-09-11): `DELETE {API_PREFIX}/models/{name}` guard
   * flags, mirrored from the same checks the endpoint enforces
   * server-side so the UI never has to re-derive them (and can't drift
   * out of sync with the real guard).
   */
  /** Region-detection pipeline model (backend `_is_region_protected_model`) — never unloadable through the UI, no override. */
  is_region_protected?: boolean;
  /** The active production model or another core pipeline model — unload requires force=true. */
  requires_force_to_unload?: boolean;
  /** Served directly (2026-09-25 follow-up to #36 item 5) — `false` for
   *  every external-service entry (the segmenter, the VLM) and any model
   *  the active config hard-blocks (`is_region_protected`, now also
   *  covering the ingest primary proposer/secondary classifier and the
   *  OCR det/rec pair, not just the region detector). `unloadButtonState`
   *  reads this FIRST, ahead of `kind`/`is_region_protected` — the
   *  server's own verdict, never re-derived from the other flags. */
  unloadable?: boolean;
  /** Served (OpenProcessor ba88751): true only for a model the pipeline
   *  can run without — the region profile's detector when a segmenter is
   *  configured. Pairs with status `not_installed`. Absent on older backends. */
  optional?: boolean;
  /** Present (with job_id/version) only for models promoted through this pipeline. */
  job_id?: string | null;
  promoted_at?: string | null;
}

export interface ModelsStatus {
  models: ModelInfo[];
}

export interface UnloadModelResponse {
  triton_name: string;
  triton_unloaded: boolean;
  directory_removed: boolean;
  forced: boolean;
  warning: string | null;
}

/** One undoable user action — a single confirmed write, however many
 *  crops it touched. What it restores is the backend's business: Z calls
 *  `POST /crops/{id}/label/undo` (one id) or
 *  `POST /crops/label/undo_batch` (several) and renders the result(s). */
export interface UndoEntry {
  crop_ids: string[];
  at: number;
  /**
   * Which backend undo route this entry reverses (M6,
   * docs/design/interactive-pass-2026-09-24.md). Defaults to `'label'`
   * when absent (every entry pushed before this field existed). Kept as
   * ONE ring buffer with a kind tag, not a separate region/vlm_dismiss
   * stack, so Z on `/clusters/[id]` — where a label write and a
   * Reject-VLM (`vlm_dismiss`) can interleave in the same session —
   * always reverses whatever the operator *actually did last*,
   * chronologically, rather than "the last label write" while silently
   * skipping a more recent dismiss. A page-scoped ignore/un-ignore
   * history (`clusterController`'s `lastExcludedIds`) stays its own
   * thing on purpose: it's bound to its own `X`/`U` keys, never `Z`, so
   * there's no ordering question to get right by sharing a stack.
   */
  kind?: 'label' | 'region' | 'vlm_dismiss';
}

// -- ingest --
// docs/design/ingest-ui-and-acceptance-plan-2026-09-24.md §B.1. Wire
// shapes for OpenProcessor's `{API_PREFIX}/ingest/*` router
// (`src/routers/curation/ingest.py`, `_common_models.py` on the
// backend). BA-1..BA-7 (OpenProcessor #36, c5c606f) landed: upload bytes
// are persisted (`source_identifier` alongside `image_path`), `error_kind`
// is a stable code on every failed result, `GET {API_PREFIX}/ingest/config`
// is real and typed, and `GET {API_PREFIX}/ingest/region_drain` serves a
// server-computed `drained` verdict.

export type IngestItemStatus = 'success' | 'duplicate' | 'failed';

// BA-7: stable machine codes served alongside `error`'s free-text message
// on a failed result. Not an exhaustive union on the wire (a future
// failure mode may still surface a message without extending this list),
// so this type stays a documented `string`, not a literal union.
export type IngestErrorKind =
  | 'empty'
  | 'unservable_path'
  | 'unsupported_type'
  | 'too_large'
  | 'decode_failed'
  | 'detector_infer'
  | 'bulk_index';

export interface IngestImageResult {
  status: IngestItemStatus;
  image_id: string;
  image_path: string;
  imohash: string;
  n_crops: number;
  n_regions: number;
  error: string | null;
  /** BA-7. Always present when `status === 'failed'`, null otherwise. */
  error_kind: IngestErrorKind | string | null;
  /**
   * BA-1: the client-supplied identifier for a byte-upload ingest, where
   * `image_path` is now the server-persisted path. Null for a
   * server-path ingest (`image_path` already IS the client-meaningful
   * identifier).
   */
  source_identifier: string | null;
  /**
   * OpenProcessor d72cc63: set when the image itself ingested but the
   * optional secondary detector failed on it (so it carries only the
   * primary detector's crops). Null otherwise; absent on an older backend.
   */
  secondary_detector_error?: string | null;
}

export interface BatchIngestSummary {
  successful: number;
  duplicates: number;
  failed: number;
  mismatches: number;
  missed_labels: number;
  unmatched_detections: number;
  labels_imported: number;
  crops_indexed: number;
  /** d72cc63: how many results carry a `secondary_detector_error`. */
  secondary_detector_failures?: number;
}

export interface BatchIngestResponse {
  status: 'success' | 'partial' | 'error';
  summary: BatchIngestSummary;
  results: IngestImageResult[];
  disagreements: Record<string, unknown>[];
}

export interface IngestStatusBucket {
  key: string;
  doc_count: number;
  key_as_string?: string;
}

export interface IngestStatus {
  total: number;
  by_source: IngestStatusBucket[];
  by_day: IngestStatusBucket[];
}

export interface RegionDrain {
  pending_detection: number;
  pending_verification: number;
  total_unfinished: number;
  /**
   * BA-3 (landed, c5c606f): true once `total_unfinished` has read 0 for
   * `IngestConfig.region_drain.stable_polls` consecutive polls of this
   * endpoint — computed server-side, not a client-invented window.
   */
  drained: boolean;
  /** Seconds since the last non-zero reading (0 while still non-zero). */
  stable_for_s: number;
  /** ISO timestamp of this poll. */
  observed_at: string;
  /**
   * V-1 (OpenProcessor d72cc63): the active region profile's Triton
   * dependencies and whether each is ready now; empty with no profile.
   */
  region_dependencies?: RegionDependencyStatus[];
  /**
   * V-1: a served, human-readable line when items are pending AND a
   * dependency is down; null otherwise. Rendered verbatim.
   */
  stall_reason?: string | null;
}

export interface RegionDependencyStatus {
  role: string;
  model: string;
  ready: boolean;
  detail: string;
  unavailable_since: string | null;
}

export interface IngestPathLookupResponse {
  known_paths: Record<string, string>;
}

export interface IngestBatchItem {
  path: string;
  source?: string;
  label_txt_path?: string | null;
}

export interface IngestBatchRequest {
  items: IngestBatchItem[];
  label_source?: string;
  detect_mismatches?: boolean;
}

export interface IngestUploadRequest {
  files: File[];
  identifiers: string[];
  source: string;
}

/**
 * BA-2 (landed, c5c606f): `GET {API_PREFIX}/ingest/config`'s served
 * shape — real, enforced limits, not client-guessed constants. Field
 * names/nesting match `IngestConfigResponse` in
 * `src/routers/curation/_common_models.py` on the backend exactly.
 */
export interface IngestConfig {
  upload: {
    enabled: boolean;
    max_images_per_request: number;
    max_bytes_per_request: number;
    accepted_extensions: string[];
    persists_bytes: boolean;
  };
  batch: { enabled: boolean; max_items: number; source_roots: string[] };
  region_drain: { poll_interval_s: number; stable_polls: number };
}
