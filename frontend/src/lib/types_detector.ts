// Owner: Track B (v0.4.0 generic detector, ingest policy, embedding).
// Wire types for the `IngestPolicy*`, `IngestDetector*`, `SeedFromDetector*`
// and `DetectionsSummary` schemas of the vendored contract
// (`contracts/openprocessor/openapi/curation.json`), pinned key for key by
// `contract/detectorContract.test.ts`.
import type { ReprocessRequest } from '$lib/types_import';

/** `IngestDetectorLabel`: one class the deployment's detector can emit. */
export interface IngestDetectorLabel {
  class_id: number;
  name: string;
  slug: string;
}

/** `IngestDetectorInfo`: the `detector` block of `GET /ingest/config`. */
export interface IngestDetectorInfo {
  model: string;
  version: string;
  input_size: number;
  assigns_class: boolean;
  confidence_floor_applies: boolean;
  n_labels: number;
  labels: IngestDetectorLabel[];
}

export const CLASS_RESOLUTIONS = ['proposal', 'by_name'] as const;
export type ClassResolution = (typeof CLASS_RESOLUTIONS)[number];

export const EMBEDDING_MODES = ['all', 'selected', 'lazy'] as const;
export type EmbeddingMode = (typeof EMBEDDING_MODES)[number];

/** `DetectFilter`: which detections an ingest keeps. */
export interface DetectFilter {
  class_resolution?: ClassResolution;
  classes?: string[] | null;
  exclude_classes?: string[];
  max_per_image?: number | null;
  min_box_area_frac?: number | null;
  min_confidence?: number | null;
}

/** `EmbeddingPolicy`: which kept detections get a vector at ingest. */
export interface EmbeddingPolicy {
  classes?: string[];
  max_per_image?: number | null;
  min_box_area_frac?: number | null;
  min_confidence?: number | null;
  mode?: EmbeddingMode;
}

/** `DetectorOverride`: a per-project detector instead of the deployment's. */
export interface DetectorOverride {
  input_size?: number | null;
  labels_path?: string;
  model: string;
  version?: string;
}

/** `IngestPolicy`: the served policy (a GET). */
export interface IngestPolicy {
  detect?: DetectFilter;
  detector?: DetectorOverride | null;
  embedding?: EmbeddingPolicy;
  revision?: number;
}

/** `IngestPolicyBody`: the preview request. */
export interface IngestPolicyBody {
  detect?: DetectFilter;
  detector?: DetectorOverride | null;
  embedding?: EmbeddingPolicy;
}

/** `IngestPolicyUpdate`: the PUT request. */
export interface IngestPolicyUpdate extends IngestPolicyBody {
  expected_revision: number;
}

/** `IngestPolicyPutResponse`. */
export interface IngestPolicyPutResponse extends IngestPolicyBody {
  revision?: number;
  unknown_names?: string[];
}

/** `PolicyPreviewClass`. */
export interface PolicyPreviewClass {
  name: string;
  would_embed: number;
  would_not_embed: number;
}

/** `IngestPolicyPreview`. */
export interface IngestPolicyPreview {
  total_items: number;
  scanned: number;
  truncated: boolean;
  would_embed: number;
  would_not_embed: number;
  /** Of `would_embed`, items that embed only because a human or validated
   *  label always embeds (counted for items stored with a vector). */
  embedded_because_labeled: number;
  estimated_vector_mb: number;
  by_class: PolicyPreviewClass[];
}

/** `SeedFromDetectorRequest`. */
export interface SeedFromDetectorRequest {
  dry_run?: boolean;
  group?: string;
  names?: string[] | null;
}

export interface SeededClass {
  class_id: number | null;
  detector_label: string;
  name: string;
}

export const SEED_SKIP_REASONS = ['exists', 'deprecated'] as const;
export type SeedSkipReason = (typeof SEED_SKIP_REASONS)[number];

export interface SeedSkipped {
  detector_label: string;
  name: string;
  reason: SeedSkipReason;
}

export const SEED_CONFLICT_REASONS = ['duplicate_slug', 'unnamed_label'] as const;
export type SeedConflictReason = (typeof SEED_CONFLICT_REASONS)[number];

export interface SeedConflict {
  class_id_in_detector: number;
  detector_label: string;
  reason: SeedConflictReason;
}

export interface SeedFromDetectorResponse {
  conflicts: SeedConflict[];
  created: SeededClass[];
  detector_model: string;
  dry_run: boolean;
  skipped: SeedSkipped[];
}

/** `EmbeddingByState`: always complete in a served response. */
export interface EmbeddingByState {
  deferred?: number;
  embedded?: number;
  failed?: number;
  not_selected?: number;
  unknown?: number;
}

export interface EmbeddingBreakdown {
  by_state: EmbeddingByState;
  embedded: number;
  not_embedded: number;
}

export interface LabelSummary {
  count: number;
  embedding: EmbeddingBreakdown;
  name: string;
}

export interface DetectionsSummary {
  by_label: LabelSummary[];
  embedding: EmbeddingBreakdown;
  labels_truncated: boolean;
  suggested_reprocess: ReprocessRequest | null;
  total: number;
}
