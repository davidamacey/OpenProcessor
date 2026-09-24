/**
 * Example second slot — proves the capability model generalizes beyond
 * license_plate (docs/genericization-plan-2026-09-13.md §5.4, Slot A).
 *
 * Stresses every capability license_plate uses, with deliberately
 * DIFFERENT values everywhere, so any place a plate-shaped default or
 * assumption leaked into the (not-yet-built) generic components would
 * be caught immediately: `storedFrame: 'parent'` (plates use `'source'`),
 * `showChainOnCard: false` (plates: true), and no `falsePositiveState`
 * (plates have one).
 *
 * Not bound to any real class or route — this is a proof-of-concept
 * profile, not a shipped deployment.
 */

import type { SlotSpec } from '../types';

export const aircraftTailNumberSlot: SlotSpec = {
  key: 'aircraft_tail_number',
  bind: { className: 'aircraft' },
  label: { singular: 'tail number', plural: 'tail numbers', title: 'Tail number' },

  capabilities: {
    subBox: {
      bboxField: 'tail_bbox_norm',
      // Different from plates on purpose: proves storedFrame is read,
      // not assumed.
      storedFrame: 'parent',
      scoreField: 'tail_score',
      thumbnail: {
        path: (id, s) => `/crops/${encodeURIComponent(id)}/tail_thumbnail?size=${s}`,
        aspect: '1 / 2',
        defaultSize: 192,
      },
      ring: {
        confirmed: 'border-green-400 shadow-[0_0_0_1px_rgba(34,197,94,0.45)]',
        proposed: 'border-yellow-400 shadow-[0_0_0_1px_rgba(250,204,21,0.45)]',
        rejected: 'border-zinc-600 shadow-none',
      },
      editor: { thumbSize: 640, viewPadding: 3.0, nudgeStep: 1 / 640 },
    },
    text: {
      valueField: 'tail_number',
      sourceField: 'tail_number_source',
      confidenceField: 'tail_number_confidence',
      label: 'Tail number',
      placeholder: 'N123AB',
      transform: 'uppercase',
      monospace: true,
      pattern: /^[A-Z]\d{1,5}[A-Z]{0,2}$/,
      maxLength: 8,
    },
    provenance: {
      detectorField: 'tail_detector',
      chainField: 'tail_detector_chain',
      verifierField: 'tail_verifier',
      showChainOnCard: false, // differs from plates
    },
    lifecycle: {
      statusField: 'tail_status',
      verifiedField: 'tail_verified',
      states: [
        { value: 'detected', label: 'detected', humanWritable: true, role: 'proposed' },
        {
          value: 'not_visible',
          label: 'no tail number visible',
          humanWritable: true,
          role: 'absent',
        },
        {
          value: 'obscured',
          label: 'obscured / partial',
          humanWritable: true,
          role: 'rejected',
        },
      ],
      confirmState: 'detected',
      rejectState: 'not_visible',
      // No falsePositiveState -> the markFalsePositive action/hotkey
      // must not be offered for this slot.
    },
    queue: {
      endpointId: 'tail_numbers',
      urlId: 'tails',
      tabLabel: 'Tail numbers',
      browsePath: '/tail_numbers',
      keymap: { confirm: ['enter'], reject: ['d'], editBox: ['e'], back: ['arrowleft'] },
      textFilter: { param: 'text', label: 'Tail #', placeholder: 'N12' },
      alwaysVisible: false,
    },
  },
  endpoints: {
    setBox: (id) => `/crops/${encodeURIComponent(id)}/tail`,
    clearBox: (id) => `/crops/${encodeURIComponent(id)}/tail`,
    patchMeta: (id) => `/crops/${encodeURIComponent(id)}/tail_meta`,
  },
  stats: {
    key: 'tail_numbers',
    panelTitle: 'Tail-number detections',
    coverageTitle: 'Tail-number coverage',
  },
};
