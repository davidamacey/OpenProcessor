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
  'detector_conf_asc',
  'unvalidated_desc',
  'updated_desc',
  'dominant_class',
] as const;
export type ClusterSortId = (typeof CLUSTER_SORT_IDS)[number];

/** `ClusterCard.validation_state` and the `GET /clusters` filter of that name. */
export const VALIDATION_STATES = ['none', 'partial', 'complete'] as const;
export type ValidationState = (typeof VALIDATION_STATES)[number];

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
  cluster_uid: string;
  total: number;
  detector_classes: FacetCount[];
  class_sources: FacetCount[];
  validation: { validated: number; unvalidated: number };
  review_tabs: Record<string, number>;
  detector_confidence: ConfidenceBin[];
  detector_confidence_missing: number;
}

/** `ClusterRunRequest.target` / `ClusterActivationRequest.target`. */
export type ClusterTarget = 'item' | 'region_box';

/** `ClusterRunAccepted`: the 202 of `POST /clusters/runs`. Follow `job_id`; its
 *  result is `{status, partition_id, n_items, n_clusters, activated, stability}`. */
export interface ClusterRunAccepted {
  job_id: string;
}

/** `ClusterPartitionView`: one fitted partition. Only the serving one names the
 *  clusters the pages show; a `ready` one is activated with `activateClusterPartition`. */
export interface ClusterPartitionView {
  id: string;
  target: string;
  level: string;
  method: string;
  embedding_field: string;
  status: 'building' | 'ready' | 'failed' | 'discarded';
  serving: boolean;
  parent_partition_id: string | null;
  n_clusters: number | null;
  n_items: number | null;
  n_review: number | null;
  n_unassigned: number | null;
  metrics: Record<string, unknown>;
  diff: Record<string, unknown> | null;
  created_at: string;
}

/** `ClusterPartitionList`. */
export interface ClusterPartitionList {
  partitions: ClusterPartitionView[];
}

/** `ClusterActivationRequest`: the expected serving partition and revision make a
 *  stale activation a 409 `partition_conflict` instead of a silent overwrite. */
export interface ClusterActivationRequest {
  partition_id: string;
  target: ClusterTarget;
  level?: string;
  expected_serving_partition_id?: string | null;
  expected_revision?: number | null;
}

/** `ClusterActivationAccepted`: the 202 of `PUT /clusters/activation`; follow `job_id`. */
export interface ClusterActivationAccepted {
  job_id: string;
  pending_partition_id: string;
  revision: number;
}
