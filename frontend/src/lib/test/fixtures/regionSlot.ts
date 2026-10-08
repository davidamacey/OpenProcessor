/**
 * The neutral region-slot test fixture (docs/design/
 * domain-neutral-audit-2026-09-24.md §4.4): items are widgets, and the
 * region is a tag on a widget, read by OCR as `TAG-001`-style text.
 *
 * Tests that need "a region slot" use this rather than a specific example
 * profile, so they exercise the generic slot path, not one domain.
 *
 * - `WIDGET_TAG_PROFILE` is what a backend configured for this domain
 *   serves on `/health.region_profile`.
 * - `widgetTagServedSlot` is the slot the app synthesizes from it — the
 *   production path (`regionSlotFromServedProfile`), untouched.
 * - `widgetTagSlot` is that slot as a deployment would customize it with a
 *   tier-2 entry keyed on the profile name: domain copy (a singular noun,
 *   text label and placeholder), hand-declared training cohorts and stats
 *   titles. Its wire half is the served slot's, by construction.
 */
import { regionSlotFromServedProfile } from '$lib/annotations/servedRegionSlot';
import type { SlotSpec } from '$lib/annotations/types';
import type { ServedRegionProfile } from '$lib/types';

export const WIDGET_TAG_CLASS = 'widget_tag';

export const WIDGET_TAG_PROFILE: ServedRegionProfile = {
  name: 'widget_tag',
  display_name: 'Widget tags',
  display_name_singular: 'Widget tag',
  region_class_name: WIDGET_TAG_CLASS,
  text_reader: 'ocr',
  reads_text: true,
  text_hint_enabled: false,
  limits: { max_boxes_per_write: 500 },
};

export const widgetTagServedSlot: SlotSpec =
  regionSlotFromServedProfile(WIDGET_TAG_PROFILE);

/** Text-free variant (OpenProcessor W1, 2026-09-26): a profile that
 *  detects/segments the region but never reads text off it —
 *  `text_reader: 'none'`, `reads_text: false`. */
export const WIDGET_TAG_PROFILE_NO_TEXT: ServedRegionProfile = {
  name: 'widget_tag',
  display_name: 'Widget tags',
  display_name_singular: 'Widget tag',
  region_class_name: WIDGET_TAG_CLASS,
  text_reader: 'none',
  reads_text: false,
  text_hint_enabled: false,
  limits: { max_boxes_per_write: 500 },
};

const served = widgetTagServedSlot;
const wire = served.capabilities;

function trainingCohort(id: string, label: string, description: string) {
  return {
    id,
    label,
    description,
    query: {
      kind: 'endpoint' as const,
      path: '/regions/training_candidates',
      params: { mode: id, class_id: '{classId}' },
    },
    rowKind: 'slot' as const,
    reviewTarget: 'slotQueue' as const,
  };
}

export const widgetTagSlot: SlotSpec = {
  ...served,
  label: { singular: 'tag', plural: 'tags', title: 'Tag' },

  capabilities: {
    ...wire,
    text: {
      ...wire.text!,
      label: 'Tag text',
      placeholder: 'TAG-001',
    },
    lifecycle: {
      ...wire.lifecycle!,
      states: wire.lifecycle!.states.map((s) => ({
        ...s,
        label: s.value.replace(/_/g, ' '),
      })),
    },
    queue: {
      ...wire.queue!,
      textFilter: { param: 'text', label: 'Tag text', placeholder: 'e.g. TAG-001' },
    },
    trainingCohorts: {
      suppressDerived: ['blind_spots', 'low_conf'],
      cohorts: [
        trainingCohort(
          'detector_blind_spots',
          'Detector blind spots',
          'The segmenter found the tag and the verifier confirmed it; the detector missed it',
        ),
        trainingCohort(
          'low_conf_correct',
          'Detector low confidence',
          'Detector and verifier agreed but the detector score was low',
        ),
        trainingCohort(
          'disagreement',
          'Model disagreements',
          'Detector and segmenter both fired',
        ),
        trainingCohort(
          'human_corrected',
          'Human corrected',
          'A human corrected a model output',
        ),
        trainingCohort(
          'false_positives',
          'False positives',
          'A human marked a detector box as a false positive (box retained)',
        ),
      ],
    },
  },

  extras: {
    datasetExport: {
      ...(served.extras!.datasetExport as Record<string, unknown>),
      label: 'Widget tag dataset',
      blurb:
        'Single-class widget-tag dataset (positives + hard negatives + backgrounds).',
    },
  },

  stats: {
    key: 'regions',
    panelTitle: 'Widget tag detections',
    coverageTitle: 'Widget tag coverage',
  },
};
