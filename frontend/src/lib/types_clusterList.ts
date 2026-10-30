/**
 * The cluster list (`GET {P}/clusters`) and cluster facets
 * (`GET {P}/clusters/{id}/facets`) wire shapes. The sort ids and the
 * validation states are pinned to the contract by
 * `contract/clusterListContract.test.ts`; the sort labels, the flag
 * vocabulary and the facet counts are served and rendered verbatim.
 */

/** `GET /clusters` `sort`. */
export const CLUSTER_SORT_IDS = [
  'needs_review_desc',
  'size_desc',
  'size_asc',
  'label_agreement_asc',
  'purity_asc',
  'purity_desc',
  'detector_conf_asc',
  'unvalidated_desc',
  'updated_desc',
  'dominant_class',
] as const;
export type ClusterSortId = (typeof CLUSTER_SORT_IDS)[number];

/** `ClusterCard.validation_state` and the `GET /clusters` filter of that name. */
export const VALIDATION_STATES = ['none', 'partial', 'complete'] as const;
export type ValidationState = (typeof VALIDATION_STATES)[number];

/** `ClusterSortOption`. */
export interface ClusterSortOption {
  id: string;
  label: string;
}

/** `ClusterFlag`: one entry of the served flag vocabulary. */
export interface ClusterFlag {
  id: string;
  label: string;
  description: string;
}

/** The card filters the list sends as query parameters. */
export interface ClusterListFilter {
  dominant_class: string[];
  detector_class: string[];
  min_size: number | null;
  max_size: number | null;
  validation_state: ValidationState | null;
  flag: string[];
  needs_review: boolean;
}

/** `FacetCount`. */
export interface FacetCount {
  value: string;
  count: number;
}

/** `ConfidenceBin`: `min` and `max` are the values to send as
 *  `detector_conf_min` (inclusive) and `detector_conf_max` (exclusive). */
export interface ConfidenceBin {
  min: number;
  max: number | null;
  count: number;
}

/** `ClusterFacets`. */
export interface ClusterFacets {
  cluster_id: number;
  total: number;
  detector_classes: FacetCount[];
  class_sources: FacetCount[];
  validation: { validated: number; unvalidated: number };
  review_tabs: Record<string, number>;
  detector_confidence: ConfidenceBin[];
  detector_confidence_missing: number;
}
