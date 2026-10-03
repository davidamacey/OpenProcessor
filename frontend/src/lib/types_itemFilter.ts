/**
 * The shared item filter (OpenProcessor v0.4.0): the enum vocabularies,
 * the query-parameter form every list/stats route accepts, and the
 * `ItemFilter` / `ItemSelection` body schemas `selection` and
 * `export/yolo item_filter` take. Owned by track C after the v0.4.0
 * Step 0 prelude (docs/design/v040-backend-deltas-ui-plan-2026-10-03.md).
 *
 * Pinned to the vendored OpenAPI by
 * `src/lib/contract/itemFilterContract.test.ts`. The backend serves no
 * vocabulary for these enums, so the const arrays are the contract's.
 */

/** `ItemDoc.embedding_state`. */
export const EMBEDDING_STATES = [
  'embedded',
  'not_selected',
  'deferred',
  'failed',
] as const;
export type EmbeddingState = (typeof EMBEDDING_STATES)[number];

/** `ItemFilter.origin`: which actor produced the item. */
export const ITEM_ORIGINS = ['detector', 'sam3', 'human', 'import'] as const;
export type ItemOrigin = (typeof ITEM_ORIGINS)[number];

/** `ItemFilter.review_status`. */
export const REVIEW_STATUSES = ['pending', 'validated', 'dismissed', 'excluded'] as const;
export type ReviewStatus = (typeof REVIEW_STATUSES)[number];

/** `ItemSelection.sample` / `ReprocessTargets.sample`. */
export const SELECTION_SAMPLES = ['random', 'largest'] as const;
export type SelectionSample = (typeof SELECTION_SAMPLES)[number];

/**
 * The shared filter as query parameters (`GET /crops` and the other list
 * and stats routes). Array values are sent as repeated keys by `qs()`.
 * `null` / `undefined` scalars are omitted. `open_vocab_set` /
 * `source_prompt` are declared on `GET /crops` only.
 */
export interface ItemFilterQuery {
  class_name?: string[];
  exclude_class_name?: string[];
  conf_min?: number | null;
  conf_max?: number | null;
  min_area?: number | null;
  max_area?: number | null;
  max_rank?: number | null;
  origin?: ItemOrigin[];
  embedding_state?: EmbeddingState[];
  review_status?: ReviewStatus[];
  open_vocab_set?: string;
  source_prompt?: string;
}

/** `ItemFilter` (request body schema). */
export interface ItemFilter {
  class_id?: number | null;
  class_names?: string[];
  class_source?: string | null;
  classifier_conf_lt?: number | null;
  cluster_id?: number | null;
  conf_max?: number | null;
  conf_min?: number | null;
  dataset_split?: string | null;
  embedding_state?: EmbeddingState[];
  exclude_class_names?: string[];
  import_id?: string | null;
  item_text?: string | null;
  label_source?: string | null;
  label_validated?: boolean | null;
  max_area?: number | null;
  max_rank?: number | null;
  min_area?: number | null;
  min_blur_ratio?: number | null;
  needs_new_class?: boolean | null;
  on_negative_frame?: boolean | null;
  open_vocab_set?: string | null;
  origin?: ItemOrigin[];
  proposed_by_import?: boolean | null;
  region_gate_skipped?: boolean | null;
  review_dismissed?: boolean | null;
  review_status?: ReviewStatus[];
  source?: string | null;
  source_prompt?: string | null;
}

/** `ItemSelection` (request body schema). */
export interface ItemSelection {
  crop_ids?: string[] | null;
  filter?: ItemFilter | null;
  include_excluded?: boolean;
  include_test?: boolean;
  limit?: number | null;
  sample?: SelectionSample | null;
  seed?: number;
}

/** What a `dry_run: true` selection write returns: the count the write
 *  would change, and nothing written (`SelectionDryRunResponse`). */
export interface SelectionDryRun {
  dry_run: true;
  selected: number;
}

/** `BatchExcludeResponse` / `BatchUnexcludeResponse`: `updated_ids` is the
 *  undo target. */
export interface SelectionExcludeResult {
  excluded?: number;
  unexcluded?: number;
  updated_ids: string[];
  errors: number;
}

/** `VectorRefresh` on every region write: boxes embedded now, and boxes
 *  still without a valid vector (retry with a reprocess `embed`). */
export interface VectorRefresh {
  embedded: number;
  pending: number;
}
