/**
 * Review wire types: tabs, queue items, search hits, diverse selection and
 * new-class resolution.
 */
import type { BulkLabelConflict, Crop } from '$lib/types';

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
  // New-class-proposal queue: crops the VLM
  // couldn't match to any registered class (`class_source:
  // 'vlm_new_class_pending'`). Backed by the same `{API_PREFIX}/review/{tab}`
  // shape as every other core tab; `/classes`'s Proposals section reads
  // the separate `.../summary` aggregate instead (see api.ts).
  | 'new_class_proposals'
  // Labels written by a dataset import (`GET {prefix}/review/imported`);
  // only offered when `GET /review/tabs` serves an `imported` entry.
  | 'imported'
  // Machine labels whose class differs from the detector's own answer
  // (label confirmation, #119); only offered when `GET /review/tabs`
  // serves a `detector_disagreements` entry.
  | 'detector_disagreements';

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
