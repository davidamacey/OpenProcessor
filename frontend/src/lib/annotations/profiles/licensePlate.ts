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
          `/crops/${encodeURIComponent(id)}/plate_thumbnail?size=${size}`,
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
          value: 'no_plate_box',
          label: 'no box found',
          humanWritable: false,
          role: 'absent',
        },
        {
          value: 'no_plate_visible',
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
      rejectState: 'no_plate_visible',
      falsePositiveState: 'false_positive',
    },

    queue: {
      endpointId: 'plates',
      urlId: 'plates',
      tabLabel: 'Plates',
      browsePath: '/plates',
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
  },

  endpoints: {
    setBox: (id) => `/crops/${encodeURIComponent(id)}/plate`,
    clearBox: (id) => `/crops/${encodeURIComponent(id)}/plate`,
    patchMeta: (id) => `/crops/${encodeURIComponent(id)}/plate_meta`,
    batchStatus: () => `/plates/batch_status`,
  },

  stats: {
    key: 'plates',
    panelTitle: 'Plate detections',
    coverageTitle: 'Plate coverage',
  },
};
