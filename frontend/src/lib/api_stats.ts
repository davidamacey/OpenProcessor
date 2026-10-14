/**
 * Health and dataset-statistics reads (`/health`, `/stats/*`).
 */
import { apiFetch, globalApi, qs, scoped } from '$lib/api';
import { apiErrorText } from '$lib/api_errors';
import type { ClassThresholds } from '$lib/types_classes';
import type { ApiHealth, GlobalHealth, StatsSummary } from '$lib/types_health';
import type { ItemFilterQuery } from '$lib/types_itemFilter';

// -- endpoints -----------------------------------------------------------

export function getHealth(signal?: AbortSignal): Promise<ApiHealth> {
  return apiFetch<ApiHealth>(`${scoped()}/health`, {}, signal);
}

/** `GET {globalApi()}/health` — unscoped, no project bound. Feeds only
 *  the top-bar API status chip. */
export function getGlobalHealth(signal?: AbortSignal): Promise<GlobalHealth> {
  return apiFetch<GlobalHealth>(`${globalApi()}/health`, {}, signal, { global: true });
}

/**
 * Pipeline-dashboard payload from `GET {API_PREFIX}/stats/dataset`. Contract
 * defined by `src/routers/curation/stats.py` — every nested key is
 * always present, numeric counters are always integers >= 0, and
 * `clusters.last_run_at` / `clusters.method` may be null when no
 * auto_label run has ever completed.
 */
export interface DatasetStats {
  as_of: string;
  total_crops: number;
  validated: number;
  test_holdout: number;
  by_source: Array<{ key: string; doc_count: number }>;
  labeled: {
    by_human: number;
    by_vlm: number;
    by_classifier: number;
    other: number;
  };
  regions: {
    /** Crops with a region_bbox_norm right now — the honest "crops with a
     *  region" count (matches the region cluster view). */
    boxed: number;
    /** Crops the verifier confirmed carry a real region (region_status='detected'). */
    confirmed: number;
    /** Sum of region_detector credit — includes rejected/failed attempts,
     *  so it OVERSTATES real regions. Not the headline. */
    total_detected: number;
    by_detector: number;
    by_segmenter: number;
    /** Crops where the operator drew a fresh region bbox from scratch. */
    by_human_drew: number;
    /** Crops whose region was verified by a human (the Confirm button). */
    verified_by_human: number;
    /** Crops whose region was verified by the VLM verifier (auto-verify). */
    verified_by_vlm: number;
    /**
     * Union: any region the operator touched — drew the bbox OR
     * confirmed an AI-proposed one. The dashboard surfaces this as
     * the honest "you reviewed N regions" number.
     */
    validated_by_human: number;
  };
  unlabeled: {
    pending_detection: number;
    pending_verification: number;
    /** Crops with no `class_id` at all, whatever their `label_source`
     *  (was miscounted as "labeled" — D1, visual audit 2026-09-24, before
     *  OpenProcessor #36 restricted `labeled.*` to docs with a real
     *  `class_id`). */
    no_label_source: number;
    /** Subset of `no_label_source` the VLM looked at but couldn't (or
     *  didn't) resolve to a class (#36 item 2). */
    vlm_no_class: number;
    /** crops a detector proposed but
     *  nothing has classified yet — a subset of `no_label_source`, like
     *  `vlm_no_class`. */
    by_proposal: number;
  };
  in_progress: {
    region_drain_total_unfinished: number;
    /** V-1 (OpenProcessor d72cc63): a served, human-readable line naming
     *  why the region drain can't progress (a region-profile dependency
     *  is down, and since when). Null when nothing is pending or every
     *  dependency is ready. Rendered verbatim. */
    region_stall_reason: string | null;
  };
  clusters: {
    last_run_at: string | null;
    /** Current total of distinct non-noise clusters in the index. */
    cluster_count: number;
    /** How many clusters the last auto-label run itself made (null: no run). */
    last_run_cluster_count?: number | null;
    residual_count: number;
    noise_count: number;
    method: string | null;
  };
  /** v0.4.0: items with and without a vector, and the served per-state
   *  breakdown (legacy items report under `unknown`). Required on the wire. */
  embedding: {
    embedded: number;
    not_embedded: number;
    by_state: Record<string, number>;
  };
}

/**
 * `GET /stats/dataset?<item filter>` `total_crops` — how many items the
 * served filter matches (the `/export` "Matching items" line). `null` when
 * the response carries no total: nothing is counted client-side.
 */
export async function getMatchingItemCount(
  filter: ItemFilterQuery,
  signal?: AbortSignal,
): Promise<number | null> {
  const raw = await apiFetch<{ total_crops?: number | null }>(
    `${scoped()}/stats/dataset${qs({ ...filter })}`,
    {},
    signal,
  );
  return typeof raw.total_crops === 'number' ? raw.total_crops : null;
}

export async function getStats(signal?: AbortSignal): Promise<StatsSummary> {
  // The API returns
  //   {API_PREFIX}/stats/dataset:  {total_crops, validated, test_holdout, by_source}
  //   {API_PREFIX}/stats/classes:  {classes:[{class_id, class_name, count, validated_count}, ...]}
  // The labeler dashboard expects StatsSummary which uses validated_crops /
  // test_holdout_crops / per_class — fold the two server
  // payloads into that shape so the dashboard can render directly.
  type RawDataset = {
    total_crops?: number;
    validated?: number;
    test_holdout?: number;
    by_source?: Array<{ key: string; doc_count: number }>;
  };
  type RawClasses = {
    classes: StatsSummary['per_class'];
    thresholds?: ClassThresholds;
  };
  // allSettled, not Promise.all: /stats/dataset can 503 (G1 — the live
  // op_items region_status mapping isn't aggregatable) while
  // /stats/classes is healthy. A dataset failure must still let
  // /export render its class table from per_class, so a rejection here
  // falls back to an empty RawDataset rather than sinking both calls.
  const [dsResult, clsResult] = await Promise.allSettled([
    apiFetch<RawDataset>(`${scoped()}/stats/dataset`, {}, signal),
    apiFetch<RawClasses>(`${scoped()}/stats/classes`, {}, signal),
  ]);
  const ds: RawDataset = dsResult.status === 'fulfilled' ? dsResult.value : {};
  const cls: RawClasses =
    clsResult.status === 'fulfilled' ? clsResult.value : { classes: [] };
  return {
    dataset_error: dsResult.status === 'rejected' ? apiErrorText(dsResult.reason) : null,
    total_crops: ds.total_crops ?? 0,
    validated_crops: ds.validated ?? 0,
    test_holdout_crops: ds.test_holdout ?? 0,
    per_class: cls.classes.map((c) => ({
      class_id: c.class_id,
      class_name: c.class_name,
      count: c.count,
      validated_count: c.validated_count,
      adequacy: c.adequacy,
      aug_target: c.aug_target,
      aug_gap: c.aug_gap,
      trainable: c.trainable,
      trainable_gap: c.trainable_gap,
    })),
    thresholds: cls.thresholds,
  };
}
