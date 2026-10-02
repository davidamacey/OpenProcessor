import type { DatasetStats } from '$lib/api';

/** A full served `GET {API_PREFIX}/stats/dataset` body, with top-level overrides. */
export function datasetStatsFixture(overrides: Partial<DatasetStats> = {}): DatasetStats {
  return {
    as_of: '2026-09-24T00:00:00Z',
    total_crops: 1000,
    validated: 500,
    test_holdout: 50,
    by_source: [{ key: 'nas1', doc_count: 1000 }],
    labeled: { by_human: 100, by_vlm: 200, by_classifier: 50, other: 5 },
    regions: {
      boxed: 0,
      confirmed: 0,
      total_detected: 0,
      by_detector: 0,
      by_segmenter: 0,
      by_human_drew: 0,
      verified_by_human: 0,
      verified_by_vlm: 0,
      validated_by_human: 0,
    },
    unlabeled: {
      pending_detection: 10,
      pending_verification: 5,
      no_label_source: 2,
      vlm_no_class: 0,
      by_proposal: 0,
    },
    in_progress: { region_drain_total_unfinished: 0, region_stall_reason: null },
    clusters: {
      last_run_at: null,
      cluster_count: 0,
      residual_count: 0,
      noise_count: 0,
      method: null,
    },
    ...overrides,
  };
}
