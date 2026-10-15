/**
 * The raw item wire (`RawCrop`, `RAW_CROP_KEYS`) and its mapping onto the
 * labeler's `Crop`. A leaf: imports no other `api*` module.
 */
import type { EmbeddingState } from '$lib/types_itemFilter';
import type { Crop, ItemTextLine } from '$lib/types';
import { mapCropSlots } from '$lib/annotations/cropSlots';
import type { XYXY } from '$lib/annotations/types';
import type { ItemAssist } from '$lib/types_assist';

/** Convert API's [x1, y1, x2, y2] to the labeler's BBoxNorm {cx, cy, w, h}. */
function xyxyToBBoxNorm(bb: number[]): import('$lib/types').BBoxNorm {
  const [x1 = 0, y1 = 0, x2 = 0, y2 = 0] = bb;
  return {
    cx: (x1 + x2) / 2,
    cy: (y1 + y2) / 2,
    w: Math.max(0, x2 - x1),
    h: Math.max(0, y2 - y1),
  };
}

/** Raw crop shape from the {API_PREFIX}/crops API. */
export type RawCrop = {
  crop_id: string;
  image_id?: string;
  image_path: string;
  bbox_norm: number[];
  class_id?: number | null;
  class_name?: string | null;
  class_source?: string;
  confidence?: number;
  /** Confidence of whoever set the label: VLM high/medium/low map to
   *  0.92/0.70/0.40, classifier labels carry their own score, human
   *  labels are null (dq-queues cutover, 2026-09-24). Distinct from
   *  `confidence`, which is always the detector/classifier score. */
  class_confidence?: number | null;
  /** `'vlm' | 'model' | null` — which pipeline set `class_confidence`. */
  class_confidence_source?: string | null;
  /** The VLM's verbatim class answer, even when it didn't match a
   *  registry class or wasn't auto-applied. */
  vlm_raw_class?: string | null;
  vlm_class_attempted_at?: string | null;
  /** `no_answer | no_match | invalid_index | unparseable | null`. */
  vlm_class_empty_reason?: string | null;
  cluster_id?: number | null;
  cluster_distance?: number | null;
  /** Nearest cluster centroid id when it differs from `cluster_id` (the
   *  item has left its cluster) — null together with cluster_distance/
   *  cluster_similarity/cluster_is_core in that case. */
  cluster_nearest_id?: number | null;
  /** Server-computed cosine similarity to this crop's cluster centroid
   *  (0..1) — the served replacement for the old client `1 -
   *  cluster_distance` estimate. Null when the backend hasn't computed
   *  it for this crop. */
  cluster_similarity?: number | null;
  /** Server-computed: `cluster_similarity >= core_similarity_min`. Drives
   *  the cluster-detail cut line — see `Crop.cluster_is_core`. */
  cluster_is_core?: boolean | null;
  cluster_subid?: string | null;
  label_validated?: boolean;
  /** G2: the class-label-specific validation flag. `label_validated` is
   *  `class_validated OR region_validated` server-side and over-reports
   *  a validated class. */
  class_validated?: boolean;
  label_source?: string;
  class_detector?: string | null;
  class_detector_version?: string | null;
  class_labeled_at?: string | null;
  class_labeler?: string | null;
  /** Ingest source tag. */
  source?: string | null;
  test_holdout?: boolean;
  crop_rank_in_image?: number | null;
  crop_area_norm?: number | null;
  blur_lap_ratio?: number | null;
  classifier_raw_confidence?: number | null;
  proposal_name?: string | null;
  detector_class_name?: string | null;
  /** Accuracy audit state: drawn into a sample, the machine label it carried
   *  when drawn, and the verdict once a human labeled it. */
  audit_sample?: boolean | null;
  audit_label_name?: string | null;
  audit_label_source?: string | null;
  audit_outcome?: string | null;
  audit_human_name?: string | null;
  detector_class_id?: number | null;
  detector_confidence?: number | null;
  vlm_confidence?: string | null;
  vlm_proposed_class_id?: number | null;
  vlm_proposed_class_name?: string | null;
  /** The backend's confirmable suggestion — served on every crop-shaped
   *  item (item 11, 2026-09-24), not just review-queue rows. */
  proposed_class_id?: number | null;
  proposed_class_name?: string | null;
  // Curation scores (Phase 3, docs/curation-strategy-plan-2026-09.md §4).
  // Optional/forward-tolerant: an un-backfilled pool just omits these.
  mistakenness_score?: number | null;
  mistakenness_method?: string | null;
  mistakenness_version?: string | null;
  mistakenness_scored_at?: string | null;
  // F8 D1 (OpenProcessor 51b05d7): the probe's opinion on this item.
  probe_disagreement?: boolean | null;
  probe_in_scope?: boolean | null;
  probe_model_version?: string | null;
  // OpenProcessor main 8990ede: server-computed actionability, folding
  // in scope + disagreement + the server's own confidence threshold.
  probe_actionable?: boolean | null;
  thumbnail_url?: string;
  updated_at?: string;
  class_excluded?: boolean;
  excluded_reason?: string | null;
  excluded_at?: string | null;
  item_text_lines?: unknown;
  // VLM / import / combine item provenance (`ItemDoc`); each is
  // served on every item, a missing key maps to null / false / [].
  vlm_endpoint?: string | null;
  vlm_model?: string | null;
  vlm_prompt_pack?: string | null;
  label_locked?: boolean;
  import_ids?: string[];
  dataset_split?: string | null;
  imported_at?: string | null;
  proposed_by_import?: string | null;
  on_negative_frame?: boolean;
  import_standalone_region?: boolean;
  proposal_chain?: string[];
  origin_project?: string | null;
  origin_item_id?: string | null;
  origin_image_id?: string | null;
  origin_split?: string | null;
  combine_conflict?: boolean;
  combine_conflict_origins?: string[];
  combine_merged_origins?: string[];
  // v0.4.0 embedding state and open-vocabulary provenance (contract
  // fce17771 `ItemDoc`); a missing key maps to null.
  embedding_state?: EmbeddingState | null;
  source_prompt?: string | null;
  open_vocab_set?: string | null;
  open_vocab_revision?: number | null;
  mask_polygon?: number[][] | null;
  region_gate_skip?: string | null;
  /** The newest assist run's suggestions (attributes, the VLM's class answer). */
  assist?: ItemAssist | null;
};

/**
 * Every key `RawCrop` declares, as a runtime value. `src/lib/test/makeItem.ts`
 * builds a fixture from it that can't silently miss a field, and
 * `src/lib/contract/wireKeys.test.ts` checks it against the backend's
 * vendored item-wire key list. The `satisfies` + exhaustiveness check below
 * makes an added or removed `RawCrop` key a compile error here.
 */
export const RAW_CROP_KEYS = [
  'crop_id',
  'image_id',
  'image_path',
  'bbox_norm',
  'class_id',
  'class_name',
  'class_source',
  'confidence',
  'class_confidence',
  'class_confidence_source',
  'vlm_raw_class',
  'vlm_class_attempted_at',
  'vlm_class_empty_reason',
  'cluster_id',
  'cluster_distance',
  'cluster_nearest_id',
  'cluster_similarity',
  'cluster_is_core',
  'cluster_subid',
  'label_validated',
  'class_validated',
  'label_source',
  'class_detector',
  'class_detector_version',
  'class_labeled_at',
  'class_labeler',
  'source',
  'test_holdout',
  'crop_rank_in_image',
  'crop_area_norm',
  'blur_lap_ratio',
  'classifier_raw_confidence',
  'proposal_name',
  'detector_class_name',
  'audit_sample',
  'audit_label_name',
  'audit_label_source',
  'audit_outcome',
  'audit_human_name',
  'detector_class_id',
  'detector_confidence',
  'vlm_confidence',
  'vlm_proposed_class_id',
  'vlm_proposed_class_name',
  'proposed_class_id',
  'proposed_class_name',
  'mistakenness_score',
  'mistakenness_method',
  'mistakenness_version',
  'mistakenness_scored_at',
  'probe_disagreement',
  'probe_in_scope',
  'probe_model_version',
  'probe_actionable',
  'thumbnail_url',
  'updated_at',
  'class_excluded',
  'excluded_reason',
  'excluded_at',
  'item_text_lines',
  'vlm_endpoint',
  'vlm_model',
  'vlm_prompt_pack',
  'label_locked',
  'import_ids',
  'dataset_split',
  'imported_at',
  'proposed_by_import',
  'on_negative_frame',
  'import_standalone_region',
  'proposal_chain',
  'origin_project',
  'origin_item_id',
  'origin_image_id',
  'origin_split',
  'combine_conflict',
  'combine_conflict_origins',
  'combine_merged_origins',
  'embedding_state',
  'source_prompt',
  'open_vocab_set',
  'open_vocab_revision',
  'mask_polygon',
  'region_gate_skip',
  'assist',
] as const satisfies readonly (keyof RawCrop)[];
// Compile error if RAW_CROP_KEYS drops (or never gains) a RawCrop key.
type _RawCropKeysExhaustive =
  Exclude<keyof RawCrop, (typeof RAW_CROP_KEYS)[number]> extends never
    ? true
    : [
        'RAW_CROP_KEYS is missing',
        Exclude<keyof RawCrop, (typeof RAW_CROP_KEYS)[number]>,
      ];
const _rawCropKeysExhaustive: _RawCropKeysExhaustive = true;

/** Parses the wire `ItemTextLine[]` tolerantly — a malformed/absent entry
 *  is dropped rather than throwing, since this is OCR output the backend
 *  may not have backfilled for every item. */
function asItemTextLines(v: unknown): ItemTextLine[] {
  if (!Array.isArray(v)) return [];
  const out: ItemTextLine[] = [];
  for (const raw of v) {
    if (typeof raw !== 'object' || raw === null) continue;
    const r = raw as Record<string, unknown>;
    out.push({
      text: typeof r.text === 'string' ? r.text : null,
      confidence: typeof r.confidence === 'number' ? r.confidence : null,
      box_norm: Array.isArray(r.box_norm) ? (r.box_norm as number[]) : null,
      rel_height: typeof r.rel_height === 'number' ? r.rel_height : null,
    });
  }
  return out;
}

/** A served string list, tolerantly: anything else (absent, null) is `[]`. */
function asStringArray(v: unknown): string[] {
  return Array.isArray(v) ? v.filter((x): x is string => typeof x === 'string') : [];
}

export function mapRawCrop(c: RawCrop): Crop {
  const bb = c.bbox_norm ?? [0, 0, 0, 0];
  const out: Crop = {
    id: c.crop_id,
    source_image_path: c.image_path,
    image_id: c.image_id || undefined,
    bbox_norm: xyxyToBBoxNorm(bb),
    class_id: c.class_id ?? null,
    class_name: c.class_name ?? null,
    detector_class_name: c.detector_class_name ?? null,
    audit_sample: c.audit_sample ?? null,
    audit_label_name: c.audit_label_name ?? null,
    audit_label_source: c.audit_label_source ?? null,
    audit_outcome: c.audit_outcome ?? null,
    audit_human_name: c.audit_human_name ?? null,
    detector_class_id: c.detector_class_id ?? null,
    detector_confidence: c.detector_confidence ?? null,
    class_source: c.class_source ?? null,
    label_source: c.label_source || 'unknown',
    label_validated: !!c.label_validated,
    class_validated: !!c.class_validated,
    label_confidence: c.confidence ?? null,
    class_confidence: c.class_confidence ?? null,
    class_confidence_source: c.class_confidence_source ?? null,
    vlm_raw_class: c.vlm_raw_class ?? null,
    vlm_class_attempted_at: c.vlm_class_attempted_at ?? null,
    vlm_class_empty_reason: c.vlm_class_empty_reason ?? null,
    cluster_id: c.cluster_id ?? null,
    cluster_distance: c.cluster_distance ?? null,
    cluster_nearest_id: c.cluster_nearest_id ?? null,
    // Served directly — no client 1-cosine-distance estimate.
    similarity_to_centroid: c.cluster_similarity ?? null,
    cluster_is_core: c.cluster_is_core ?? null,
    cluster_subid: c.cluster_subid ?? null,
    class_detector: c.class_detector ?? null,
    class_detector_version: c.class_detector_version ?? null,
    class_labeled_at: c.class_labeled_at ?? null,
    class_labeler: c.class_labeler ?? null,
    source: c.source ?? null,
    proposed_class_id: c.proposed_class_id ?? null,
    proposed_class_name: c.proposed_class_name ?? null,
    test_holdout: !!c.test_holdout,
    crop_rank_in_image: c.crop_rank_in_image ?? null,
    crop_area_norm: c.crop_area_norm ?? null,
    blur_lap_ratio: c.blur_lap_ratio ?? null,
    classifier_raw_confidence: c.classifier_raw_confidence ?? null,
    proposal_name: c.proposal_name ?? null,
    vlm_confidence: c.vlm_confidence ?? null,
    vlm_suggested_class_id: c.vlm_proposed_class_id ?? null,
    vlm_suggested_class_name: c.vlm_proposed_class_name ?? null,
    mistakenness_score: c.mistakenness_score ?? null,
    mistakenness_method: c.mistakenness_method ?? null,
    mistakenness_version: c.mistakenness_version ?? null,
    mistakenness_scored_at: c.mistakenness_scored_at ?? null,
    probe_disagreement: c.probe_disagreement ?? null,
    probe_in_scope: c.probe_in_scope ?? null,
    probe_model_version: c.probe_model_version ?? null,
    probe_actionable: c.probe_actionable ?? null,
    class_excluded: !!c.class_excluded,
    excluded_reason: c.excluded_reason ?? null,
    excluded_at: c.excluded_at ?? null,
    item_text_lines: asItemTextLines(c.item_text_lines),
    vlm_endpoint: c.vlm_endpoint ?? null,
    vlm_model: c.vlm_model ?? null,
    vlm_prompt_pack: c.vlm_prompt_pack ?? null,
    label_locked: !!c.label_locked,
    import_ids: asStringArray(c.import_ids),
    dataset_split: c.dataset_split ?? null,
    imported_at: c.imported_at ?? null,
    proposed_by_import: c.proposed_by_import ?? null,
    on_negative_frame: !!c.on_negative_frame,
    import_standalone_region: !!c.import_standalone_region,
    proposal_chain: asStringArray(c.proposal_chain),
    origin_project: c.origin_project ?? null,
    origin_item_id: c.origin_item_id ?? null,
    origin_image_id: c.origin_image_id ?? null,
    origin_split: c.origin_split ?? null,
    combine_conflict: !!c.combine_conflict,
    combine_conflict_origins: asStringArray(c.combine_conflict_origins),
    combine_merged_origins: asStringArray(c.combine_merged_origins),
    embedding_state: c.embedding_state ?? null,
    source_prompt: c.source_prompt ?? null,
    open_vocab_set: c.open_vocab_set ?? null,
    open_vocab_revision: c.open_vocab_revision ?? null,
    mask_polygon: c.mask_polygon ?? null,
    region_gate_skip: c.region_gate_skip ?? null,
    assist: c.assist ?? null,
    slots: mapCropSlots(c as unknown as Record<string, unknown>, bb as XYXY),
    // Preserve server-side updated_at — overriding it client-side breaks
    // ordering and lets the same crop key appear twice in keyed each blocks
    // (Svelte each_key_duplicate).
    updated_at:
      typeof (c as Record<string, unknown>).updated_at === 'string'
        ? ((c as Record<string, unknown>).updated_at as string)
        : '',
  };
  return out;
}
