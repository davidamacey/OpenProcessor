/**
 * The legacy license_plate slot — the one fully-working configured
 * instance of the annotation-slot mechanism. Decomposes the ~30
 * `plate_*` fields currently hardcoded across api.ts / SlotCard.svelte /
 * PlateEditor.svelte / review/+page.svelte / clusters/+page.svelte into
 * the five independent capability blocks (§2.1 of the plan).
 *
 * Mirrors openprocessor's `PlateStatus` (src/config/plate_state.py, all 8
 * values) and `HUMAN_PLATE_STATUS_VALUES`
 * (src/routers/legacy/_common.py:324, the 4 human-writable ones).
 *
 * Not yet bound to any route — this is Phase 1's additive foundation.
 */

import type { SlotSpec } from '../types';
import { PLATE_SHAPE_ENVELOPE } from '../../shapeGate';

export const licensePlateSlot: SlotSpec = {
  key: 'license_plate',
  bind: { className: 'license_plate' },
  label: { singular: 'plate', plural: 'plates', title: 'Plate' },

  capabilities: {
    subBox: {
      bboxField: 'plate_bbox_norm',
      storedFrame: 'source',
      frameField: 'plate_bbox_frame',
      scoreField: 'plate_score',
      visibleField: 'plate_visible',
      envelope: PLATE_SHAPE_ENVELOPE,
      thumbnail: {
        path: (id, size) =>
          `/crops/${encodeURIComponent(id)}/region_thumbnail?size=${size}`,
        aspect: '2 / 1',
        defaultSize: 160,
      },
      ring: {
        confirmed: 'border-green-400 shadow-[0_0_0_1px_rgba(34,197,94,0.45)]',
        proposed: 'border-yellow-400 shadow-[0_0_0_1px_rgba(250,204,21,0.45)]',
        rejected: 'border-zinc-600 shadow-none',
      },
      editor: { thumbSize: 512, viewPadding: 2.5, nudgeStep: 1 / 512 },
    },

    text: {
      valueField: 'plate_text',
      rawField: 'plate_text_raw',
      sourceField: 'plate_text_source',
      confidenceField: 'plate_text_confidence',
      engineVersionField: 'plate_text_engine_version',
      label: 'Plate text',
      placeholder: 'ABC123',
      transform: 'uppercase',
      monospace: true,
    },

    provenance: {
      detectorField: 'plate_detector',
      detectorVersionField: 'plate_detector_version',
      chainField: 'plate_detector_chain',
      verifierField: 'plate_verifier',
      verifierVersionField: 'plate_verifier_version',
      verifiedAtField: 'plate_verified_at',
      detectedAtField: 'plate_detected_at',
      showChainOnCard: true,
    },

    lifecycle: {
      statusField: 'plate_status',
      verifiedField: 'plate_verified',
      rejectionReasonField: 'plate_rejection_reason',
      states: [
        {
          value: 'pending_detection',
          label: 'pending detection',
          humanWritable: false,
          role: 'pending',
        },
        {
          value: 'pending_verification',
          label: 'pending verification',
          humanWritable: false,
          role: 'pending',
        },
        {
          value: 'detected',
          label: 'detected (plate visible)',
          humanWritable: true,
          role: 'proposed',
        },
        {
          value: 'verify_rejected',
          label: 'rejected (bad detection)',
          humanWritable: true,
          role: 'rejected',
        },
        {
          // Wave 2 C14 (docs/design/slot-generic-crop-mapping-plan-2026-09-21.md
          // §8.4(ii)): OpenProcessor renamed this status value
          // no_plate_box -> no_region_box (merged at b3f928d). `value`
          // is what the UI now writes; `aliases` keeps the old value
          // resolving to this same state on read forever, since
          // existing OpenSearch documents may still carry it (the
          // 347k-doc reindex this would otherwise require was WONTFIX'd).
          value: 'no_region_box',
          aliases: ['no_plate_box'],
          label: 'no box found',
          humanWritable: false,
          role: 'absent',
        },
        {
          // no_plate_visible -> no_region_visible, same rename/alias story.
          value: 'no_region_visible',
          aliases: ['no_plate_visible'],
          label: 'no plate visible',
          humanWritable: true,
          role: 'absent',
        },
        {
          value: 'detection_failed',
          label: 'detection failed',
          humanWritable: false,
          role: 'pending',
        },
        {
          value: 'false_positive',
          label: 'false positive (keep box)',
          humanWritable: true,
          role: 'falsePositive',
          dim: true,
          badge: 'false pos',
        },
      ],
      confirmState: 'detected',
      rejectState: 'no_region_visible',
      falsePositiveState: 'false_positive',
    },

    queue: {
      // Wave 2 C14: the review-tab endpoint id 'plates' -> 'regions'
      // (GET /curation/review/regions; the old 'plates' id now 400s
      // server-side). urlId stays 'plates' FOREVER — it's the bookmark
      // contract (reviewTabs.ts's tabFromUrlId), decoupled from the
      // wire route name by design; do not rename it.
      endpointId: 'regions',
      urlId: 'plates',
      tabLabel: 'Plates',
      browsePath: '/regions',
      keymap: {
        confirm: ['enter'],
        reject: ['d'],
        markFalsePositive: ['f'],
        editBox: ['e'],
        back: ['arrowleft', 'b'],
      },
      textFilter: { param: 'text', label: 'Text', placeholder: 'e.g. S14' },
      alwaysVisible: true,
    },

    // P2.13 (docs/genericization-plan-2026-09-13.md §9.2.3): the five
    // LPR training-candidate modes, moved here verbatim from the old
    // train/+page.svelte:535-557 PLATE_COHORTS literal. Every query is
    // the same GET with the same params, so the compiled URL is
    // byte-identical to what getTrainingCandidates(mode, {class_id})
    // produced — proven in cohorts.test.ts. `false_positives` is the
    // 5th mode: implemented server-side (op_plates.py:266-283), typed
    // in the old TrainingCohortMode, and unreachable from the UI until
    // this registration (§9.1's live defect #1).
    //
    // Wave 2 C13/C14 (docs/design/slot-generic-crop-mapping-plan-2026-09-21.md
    // §8.4(iii)): `id` (internal — a render key + suppressDerived match
    // target) was split from `params.mode` (the wire value the
    // backend's ?mode= query string actually sends) in C13; C14 flips
    // `mode` itself, now that both sides agree: 'lpr_blind_spots' ->
    // 'detector_blind_spots', 'lpr_low_conf_correct' -> 'low_conf_correct'.
    trainingCohorts: {
      // license_plate's own hand-tuned detector_blind_spots/low_conf_correct
      // are strictly better than the generic derived blind_spots/low_conf
      // (they additionally require Gemma verification / a specific
      // detector-chain tag neither of which is derivable from capability
      // shape alone) — suppress the generic ones outright rather than
      // showing both under different ids for the same underlying slot.
      suppressDerived: ['blind_spots', 'low_conf'],
      cohorts: [
        {
          id: 'detector_blind_spots',
          label: 'LPR blind spots',
          description:
            'SAM3 found the plate, Gemma confirmed, LPR missed — high-signal training examples',
          query: {
            kind: 'endpoint',
            path: '/regions/training_candidates',
            params: { mode: 'detector_blind_spots', class_id: '{classId}' },
          },
          rowKind: 'slot',
          reviewTarget: 'slotQueue',
        },
        {
          id: 'low_conf_correct',
          label: 'LPR low confidence',
          description: 'LPR + Gemma agreed but LPR score < 0.6 — high-loss training rows',
          query: {
            kind: 'endpoint',
            path: '/regions/training_candidates',
            params: { mode: 'low_conf_correct', class_id: '{classId}' },
          },
          rowKind: 'slot',
          reviewTarget: 'slotQueue',
        },
        {
          id: 'disagreement',
          label: 'Model disagreements',
          description: 'LPR + SAM3 both fired; review for IoU disagreement',
          query: {
            kind: 'endpoint',
            path: '/regions/training_candidates',
            params: { mode: 'disagreement', class_id: '{classId}' },
          },
          rowKind: 'slot',
          reviewTarget: 'slotQueue',
        },
        {
          id: 'human_corrected',
          label: 'Human corrected',
          description: 'Human reviewed and corrected a model output — gold standard',
          query: {
            kind: 'endpoint',
            path: '/regions/training_candidates',
            params: { mode: 'human_corrected', class_id: '{classId}' },
          },
          rowKind: 'slot',
          reviewTarget: 'slotQueue',
        },
        {
          id: 'false_positives',
          label: 'False positives',
          description: 'Human marked a detector box as a false positive (box retained)',
          query: {
            kind: 'endpoint',
            path: '/regions/training_candidates',
            params: { mode: 'false_positives', class_id: '{classId}' },
          },
          rowKind: 'slot',
          reviewTarget: 'slotQueue',
        },
      ],
    },
  },

  // The single-class plate dataset, built by OpenProcessor's generic
  // narrowed export from each crop's region sub-box. `kind` is the
  // `/methods` export-axis id that gates the panel. Consumed by /train's
  // capability gate (P2.15); options[] is still rendered as bound
  // controls rather than a generic form — follow-up.
  extras: {
    datasetExport: {
      kind: 'single_class',
      label: 'LPR plate dataset',
      buildPath: '/export/single_class',
      statusPath: '/export/single_class/status',
      datasetKind: 'license_plate',
      profileName: 'license_plate',
      boxSource: 'region',
      regionClassName: 'license_plate',
      classIds: [],
      singleClass: true,
      blurb:
        'Single-class plate dataset (positives + human FP hard-negatives + a sample of plate-free backgrounds).',
      options: [
        {
          key: 'image_mode',
          label: 'image mode',
          kind: 'select',
          choices: ['whole_frame', 'item_crop'],
          default: 'whole_frame',
        },
        {
          key: 'img_max_side',
          label: 'image size',
          kind: 'select',
          choices: [640, 1280],
          default: 1280,
        },
        { key: 'max_positive_images', label: 'sample N positives', kind: 'number' },
        {
          key: 'dedup_threshold',
          label: 'dedup near-dup frames',
          kind: 'toggle',
          onValue: 0.98,
        },
      ],
    },
  },

  // Wave 2 C14: the crop-scoped writes rename too —
  // PUT/DELETE /curation/crops/{id}/plate -> /region,
  // PATCH /curation/crops/{id}/plate_meta -> /region_meta — confirmed by the
  // backend session alongside the collection-route rename (plan §8.5
  // Q2, resolved "yes" rather than the plan's conservative default).
  endpoints: {
    setBox: (id) => `/crops/${encodeURIComponent(id)}/region`,
    clearBox: (id) => `/crops/${encodeURIComponent(id)}/region`,
    patchMeta: (id) => `/crops/${encodeURIComponent(id)}/region_meta`,
    batchStatus: () => `/regions/batch_status`,
  },

  stats: {
    key: 'plates',
    panelTitle: 'Plate detections',
    coverageTitle: 'Plate coverage',
  },
};
