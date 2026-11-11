import type { DatasetStats } from '$lib/api_stats';

const LABELED_ROWS = [
  ['by_human', 'Human'],
  ['by_import', 'Imported'],
  ['by_vlm', 'VLM suggestions'],
  ['by_classifier', 'Classifier'],
  ['other', 'Other'],
] as const;

/** A served `labeled` block: the provenance counts, their total and one row each. */
export function labeledStatsFixture(
  counts: Partial<Record<(typeof LABELED_ROWS)[number][0], number>> = {},
): DatasetStats['labeled'] {
  const count = (key: (typeof LABELED_ROWS)[number][0]) => counts[key] ?? 0;
  const total = LABELED_ROWS.reduce((n, [key]) => n + count(key), 0);
  return {
    by_human: count('by_human'),
    by_import: count('by_import'),
    by_vlm: count('by_vlm'),
    by_classifier: count('by_classifier'),
    other: count('other'),
    total,
    rows: LABELED_ROWS.map(([key, label]) => ({
      key,
      label,
      count: count(key),
      share: total === 0 ? 0 : count(key) / total,
    })),
  };
}

/** A full served `GET {API_PREFIX}/stats/dataset` body, with top-level overrides. */
export function datasetStatsFixture(overrides: Partial<DatasetStats> = {}): DatasetStats {
  return {
    as_of: '2026-09-24T00:00:00Z',
    total_crops: 1000,
    validated: 500,
    test_holdout: 50,
    by_source: [{ key: 'nas1', doc_count: 1000 }],
    labeled: labeledStatsFixture({
      by_human: 100,
      by_vlm: 200,
      by_classifier: 50,
      other: 5,
    }),
    regions: {
      boxed: 0,
      coverage_share: 0,
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
    embedding: { embedded: 1000, not_embedded: 0, by_state: { embedded: 1000 } },
    ...overrides,
  };
}
