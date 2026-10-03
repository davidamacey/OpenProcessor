/**
 * The region slot, synthesized from the backend's served region profile
 * (`GET {API_PREFIX}/health` `region_profile`, OpenProcessor naming-w2;
 * docs/design/domain-neutral-audit-2026-09-24.md §5.3).
 *
 * The backend supports exactly one region profile, and every region wire
 * name (`region_*` item keys, `/regions`, `/crops/{id}/region*`) is the
 * same whatever that profile detects. So the wire half of the slot is a
 * constant, `REGION_WIRE_CAPABILITIES`, and the only per-deployment parts
 * come from the served profile: the slot key (`name`), the bound class
 * (`region_class_name`) and the noun shown to operators (`display_name`).
 *
 * A deployment that wants richer copy (a singular noun, a text
 * placeholder, hand-tuned cohorts) registers a tier-2 profile whose `key`
 * equals the served `name`; it replaces this slot wholesale (see
 * `examples/annotation-profiles/`).
 */

import type { ServedRegionProfile } from '$lib/types';
import { keymapStore } from '$stores/keymap.svelte';
import type { SlotSpec, SlotState, SubBoxCapability, TextCapability } from './types';

const encode = encodeURIComponent;

/**
 * Wire field names for the region sub-box, identical for every profile.
 *
 * Multi-box list only (docs/design/w8-multibox-frontend-plan-2026-09-26.md):
 * the backend serves every region box as an element of `region_boxes`
 * and no per-box scalar item key exists. `readSlot` never runs the
 * scalar-box path for a capability that declares `listField`.
 */
export const REGION_SUB_BOX: SubBoxCapability = {
  listField: 'region_boxes',
  countField: 'region_count',
  rejectedCountField: 'region_rejected_count',
  maxScoreField: 'region_max_score',
  setCompleteField: 'region_set_complete',
  revisionField: 'region_revision',
  thumbnail: {
    path: (id, boxId, size) =>
      `/crops/${encode(id)}/region_thumbnail?box_id=${encode(boxId)}&size=${size}`,
    aspect: '2 / 1',
    defaultSize: 160,
  },
  ring: {
    confirmed: 'border-green-400 shadow-[0_0_0_1px_rgba(34,197,94,0.45)]',
    proposed: 'border-yellow-400 shadow-[0_0_0_1px_rgba(250,204,21,0.45)]',
    rejected: 'border-zinc-600 shadow-none',
  },
  editor: { thumbSize: 512, viewPadding: 2.5, nudgeStep: 1 / 512 },
};

/** The region text capability. A region's text is per box
 *  (`SlotBox.text`, written through `PATCH /crops/{id}/regions/{box_id}`),
 *  so no item-level wire field is declared. `label`/`placeholder` are
 *  generic; the served profile carries no text noun. The placeholder is an
 *  instruction, never a sample reading (visual audit R9: a sample value in
 *  an empty field read as a VLM reading). */
export const REGION_TEXT: TextCapability = {
  label: 'Text',
  placeholder: 'type the text…',
  transform: 'none',
  monospace: true,
};

/**
 * The backend's `RegionStatus` values (contracts/openprocessor/ts/
 * regionStatus.ts) with neutral labels. The served `/regions/statuses`
 * vocabulary is the primary source for labels and human-writable flags
 * (`$lib/review/slotPanel.ts`); this list is the synchronous fallback
 * `readSlot` uses for a status's role/dim/badge.
 */
export const REGION_STATES: SlotState[] = [
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
  { value: 'detected', label: 'detected', humanWritable: true, role: 'proposed' },
  {
    value: 'verify_rejected',
    label: 'rejected (bad detection)',
    humanWritable: true,
    role: 'rejected',
  },
  { value: 'no_region_box', label: 'no box found', humanWritable: false, role: 'absent' },
  {
    value: 'no_region_visible',
    label: 'none visible',
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
];

/** The region capabilities that do not depend on the served profile. */
export const REGION_WIRE_CAPABILITIES: Pick<
  SlotSpec['capabilities'],
  'subBox' | 'text' | 'provenance' | 'lifecycle'
> = {
  subBox: REGION_SUB_BOX,
  text: REGION_TEXT,
  provenance: {
    chainField: 'region_detector_chain',
    verifierField: 'region_verifier',
    verifierVersionField: 'region_verifier_version',
    verifiedAtField: 'region_verified_at',
    detectedAtField: 'region_detected_at',
    showChainOnCard: true,
  },
  lifecycle: {
    statusField: 'region_status',
    verifiedField: 'region_verified',
    validatedField: 'region_validated',
    autoConfirmedField: 'region_auto_confirmed',
    rejectionReasonField: 'region_rejection_reason',
    labelSourceField: 'region_label_source',
    states: REGION_STATES,
    confirmState: 'detected',
    rejectState: 'no_region_visible',
    falsePositiveState: 'false_positive',
  },
};

/**
 * Box geometry/state/text writes (`putRegionBoxes`/`patchRegionBox`/
 * `putBatchRegions`/`postBatchBoxState`, `api.ts`) are called directly by
 * `multiBoxRegionController`, never through `SlotSpec.endpoints`;
 * `patchMeta`/`batchStatus` are the whole-set status paths.
 */
export const REGION_ENDPOINTS: SlotSpec['endpoints'] = {
  patchMeta: (id) => `/crops/${encode(id)}/region_meta`,
  batchStatus: () => `/regions/batch_status`,
};

/** The review-queue / tab id the backend serves region items under. */
export const REGION_TAB_ID = 'regions';

/** Fallback noun when the profile sets no `display_name` (the backend's
 *  own `/review/tabs` fallback label is the same word). */
const GENERIC_NOUN = 'Regions';

/** Fallback singular title when the profile sets no `display_name_singular`. */
const GENERIC_NOUN_SINGULAR = 'Region';

export function regionSlotFromServedProfile(p: ServedRegionProfile): SlotSpec {
  const noun = p.display_name.trim() || GENERIC_NOUN;
  // A profile without a region class still gets its review tab; binding
  // falls back to the profile name so the slot is never keyless.
  const className = p.region_class_name.trim() || p.name;
  const hasText = p.reads_text;
  // `title` reads as a singular in the UI ("Confirm Region", "Region
  // score"); the served plural noun ("Widget tags") is wrong there, so it's
  // its own served field, falling back to the generic "Region" (#36 item 10).
  const singularTitle = p.display_name_singular.trim() || GENERIC_NOUN_SINGULAR;

  return {
    key: p.name,
    bind: { className },
    label: {
      singular: singularTitle.toLowerCase(),
      plural: noun,
      title: singularTitle,
    },
    capabilities: {
      ...REGION_WIRE_CAPABILITIES,
      subBox: { ...REGION_SUB_BOX, maxBoxesPerWrite: p.limits.max_boxes_per_write },
      text: hasText ? REGION_TEXT : undefined,
      queue: {
        endpointId: REGION_TAB_ID,
        urlId: REGION_TAB_ID,
        tabLabel: noun,
        browsePath: '/regions',
        // The region keys are the keymap's `review.region.*` actions
        // (keymapFallback.ts), not a second literal here.
        keymap: {
          confirm: keymapStore.keysFor('review.region.confirm'),
          reject: keymapStore.keysFor('review.region.reject'),
          markFalsePositive: keymapStore.keysFor('review.region.false_positive'),
          editBox: keymapStore.keysFor('review.region.edit_box'),
          back: keymapStore.keysFor('review.region.back'),
        },
        textFilter: hasText
          ? { param: 'text', label: 'Text', placeholder: '' }
          : undefined,
        alwaysVisible: true,
      },
    },
    // The single-class export of this profile's region boxes. `/train`
    // still gates the panel on `/methods` advertising `single_class`.
    // `profileName` is the served profile name, so the export's output
    // root (and every past export under it) stays where it was.
    extras: {
      datasetExport: {
        kind: 'single_class',
        label: `${noun} dataset`,
        buildPath: '/export/single_class',
        statusPath: '/export/single_class/status',
        datasetKind: p.name,
        profileName: p.name,
        boxSource: 'region',
        regionClassName: className,
        classIds: [],
        singleClass: true,
        blurb: `Single-class ${noun} dataset: confirmed region boxes, human-marked false positives as hard negatives, and a sample of items with no region as backgrounds.`,
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
        ],
      },
    },
    endpoints: REGION_ENDPOINTS,
    stats: { key: 'regions', panelTitle: noun, coverageTitle: `${noun} coverage` },
  };
}
