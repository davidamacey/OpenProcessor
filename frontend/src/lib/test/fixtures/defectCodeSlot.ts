/**
 * Test fixture, part B — proves the capability model degrades correctly
 * with NO geometry at all (docs/genericization-plan-2026-09-13.md §5.4,
 * Slot B). The region slot and `aircraft_tail_number` both use `subBox`;
 * this one deliberately omits it, which is the specific case that breaks
 * an over-fitted "every slot has a box" abstraction.
 *
 * Not bound to any real class or route. Its operator-facing JSON form is
 * `examples/annotation-profiles/defect-code.json`.
 */

import type { SlotSpec } from '$lib/annotations/types';

export const defectCodeSlot: SlotSpec = {
  key: 'defect_code',
  bind: { className: 'part_surface' },
  label: { singular: 'defect code', plural: 'defect codes', title: 'Defect code' },

  capabilities: {
    // NO subBox: no geometry, no thumbnail, no envelope, no editor.
    text: {
      valueField: 'defect_code',
      sourceField: 'defect_code_source',
      confidenceField: 'defect_code_confidence',
      label: 'Defect',
      transform: 'none',
      monospace: false,
      vocabulary: [
        { value: 'none', label: 'No defect' },
        { value: 'scratch', label: 'Scratch', description: 'Linear surface mark' },
        { value: 'dent', label: 'Dent / deformation' },
        { value: 'corrosion', label: 'Corrosion' },
        { value: 'weld', label: 'Weld defect' },
      ],
    },
    provenance: {
      detectorField: 'defect_model',
      detectorVersionField: 'defect_model_version',
      showChainOnCard: false,
    },
    lifecycle: {
      statusField: 'defect_status',
      verifiedField: 'defect_verified',
      states: [
        {
          value: 'proposed',
          label: 'model proposed',
          humanWritable: false,
          role: 'proposed',
        },
        {
          value: 'confirmed',
          label: 'confirmed',
          humanWritable: true,
          role: 'confirmed',
        },
        {
          value: 'corrected',
          label: 'corrected',
          humanWritable: true,
          role: 'confirmed',
        },
      ],
      confirmState: 'confirmed',
      rejectState: 'corrected',
    },
    queue: {
      endpointId: 'defects',
      urlId: 'defects',
      tabLabel: 'Defects',
      browsePath: '/defects',
      keymap: { confirm: ['enter'], reject: ['d'], back: ['arrowleft'] },
      alwaysVisible: true,
    },
  },
  endpoints: { patchMeta: (id) => `/crops/${encodeURIComponent(id)}/defect_meta` },
};
