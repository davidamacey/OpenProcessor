/**
 * The neutral region-slot test fixture (docs/design/
 * domain-neutral-audit-2026-09-24.md §4.4): items are widgets, and the
 * region is a tag on a widget, read by OCR as `TAG-001`-style text.
 *
 * Tests that need "a region slot" use this rather than a specific example
 * profile, so they exercise the generic slot path, not one domain.
 *
 * The `region_*` wire capability map (field names, endpoints, thumbnail
 * path) is reused from the built-in example profile rather than copied —
 * it is the one place those wire names live until the served-profile
 * synthesis (`regionSlotFromServedProfile`, audit step 8) replaces this
 * hand-built spec. Everything user-visible (labels, tab, cohorts, export,
 * stats) is the widget domain.
 */
import { licensePlateSlot as regionWireSource } from '$lib/annotations/profiles/licensePlate';
import type { SlotSpec } from '$lib/annotations/types';

const wire = regionWireSource.capabilities;

export const WIDGET_TAG_CLASS = 'widget_tag';

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
  key: 'widget_tag',
  bind: { className: WIDGET_TAG_CLASS },
  label: { singular: 'tag', plural: 'tags', title: 'Tag' },

  capabilities: {
    subBox: wire.subBox,
    text: {
      ...wire.text!,
      label: 'Tag text',
      placeholder: 'TAG-001',
    },
    provenance: wire.provenance,
    lifecycle: {
      ...wire.lifecycle!,
      states: wire.lifecycle!.states.map((s) => ({
        ...s,
        label: s.value.replace(/_/g, ' '),
      })),
    },
    queue: {
      ...wire.queue!,
      endpointId: 'regions',
      urlId: 'regions',
      tabLabel: 'Widget tags',
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
      ...regionWireSource.extras!.datasetExport!,
      label: 'Widget tag dataset',
      datasetKind: WIDGET_TAG_CLASS,
      profileName: WIDGET_TAG_CLASS,
      regionClassName: WIDGET_TAG_CLASS,
      blurb:
        'Single-class widget-tag dataset (positives + hard negatives + backgrounds).',
    },
  },

  endpoints: regionWireSource.endpoints,

  stats: {
    key: 'regions',
    panelTitle: 'Widget tag detections',
    coverageTitle: 'Widget tag coverage',
  },
};
