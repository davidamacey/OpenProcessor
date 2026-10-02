/**
 * Annotation-slot type model.
 *
 * A "slot" is one secondary annotation attached to a crop — a sub-bbox,
 * a text attribute, a triage state, or any combination (e.g. a text
 * region on an item). The shape below is a *decomposition* into
 * independent capabilities, not one domain's struct. See
 * docs/genericization-plan-2026-09-13.md §2.5.
 *
 * Design constraint that drives almost every decision in this file:
 * **the frontend reads whatever field names the backend already emits.**
 * Every capability names its wire fields explicitly rather than assuming
 * a canonical schema, so this ships with zero backend change.
 *
 * This module is intentionally free of imports from application code
 * (api.ts, routes, other components) so it can never break anything by
 * existing — it is pure types + pure functions, consumed opt-in.
 */

// Type-only — erased at compile time, so this doesn't create a real
// runtime cycle with cohorts.ts importing SlotSpec from this module.
import type { TrainingCohortsCapability } from './cohorts';

/* ------------------------------------------------------------------ */
/* Primitives                                                          */
/* ------------------------------------------------------------------ */

/** Stable identifier for a slot within a deployment. Lowercase snake_case
 *  by convention; used in URLs (`?tab=slot:widget_tag`), storage keys,
 *  and `slots` indexing on a mapped crop, so it must never change once
 *  shipped. */
export type SlotKey = string;

/**
 * A field name on the raw crop JSON.
 *
 * Intentionally a plain `string`, NOT `keyof RawCrop`. A deployment
 * profile loaded from `static/annotation-profiles.json` must be able to
 * name fields this build has never heard of — forward tolerance is the
 * entire point of the adapter. Reads go through `pick()` in
 * `readSlot.ts`, which is `unknown`-typed and narrows at the boundary.
 */
export type WireField = string;

/**
 * Which coordinate frame a stored sub-bbox is expressed in.
 *
 * - `'source'`  — normalized to the full source image (what every current
 *                 built-in write produces).
 * - `'parent'`  — normalized to the parent crop's box ([0,1]^4 inside it).
 */
export type SlotFrame = 'source' | 'parent';

/** `[x1, y1, x2, y2]`, normalized. The wire format for every bbox. */
export type XYXY = [number, number, number, number];

/** `{cx, cy, w, h}`, normalized. Matches `BBoxNorm` in `../types`. */
export interface BBoxNormLike {
  cx: number;
  cy: number;
  w: number;
  h: number;
}

/** Tailwind class triple for a chip/ring. Static strings only — Tailwind
 *  cannot see dynamically-interpolated class names, so these may never be
 *  built with template literals at runtime. */
export interface Palette {
  border: string;
  bg: string;
  text: string;
}

/* ------------------------------------------------------------------ */
/* Capability: subBox                                                  */
/* ------------------------------------------------------------------ */

export interface SubBoxRing {
  /** A human has confirmed this box. */
  confirmed: string;
  /** A model proposed it; no human has ruled. */
  proposed: string;
  /** Human said "this is not a <slot>" but the box was retained. */
  rejected: string;
}

/**
 * One element of a multi-box list (W8, `RegionBoxWire`). Additive to the
 * existing scalar `SubBoxCapability` fields below — a slot that declares
 * `listField` gets `SlotData.subBoxes` populated by `readSlot`; slots that
 * don't (every non-region slot today) are unaffected. See
 * docs/design/w8-multibox-frontend-plan-2026-09-26.md.
 */
export interface SlotBox {
  /** Stable id within the item, e.g. `"b1"`. Null only for a not-yet-saved local box. */
  boxId: string | null;
  state: string;
  /** Source-frame geometry, `[x1,y1,x2,y2]` normalized to the source image. */
  rawXyxy: XYXY | null;
  /** Crop-local geometry, server-projected. Null = not drawable in the crop view. */
  parent: BBoxNormLike | null;
  score: number | null;
  detector: string | null;
  detectorVersion: string | null;
  source: string | null;
  /** The verifier's own box-correctness verdict (`false` = "model said
   *  wrong box"); null when no verdict was given. */
  bboxCorrect: boolean | null;
  confidence: string | null;
  rejectionReason: string | null;
  /** A human created, verdicted or edited this box (or it was imported
   *  from a labeled dataset): automated stages never touch it. */
  locked: boolean | null;
  /** The chosen text reading (only served when the profile reads text). */
  text: string | null;
  textRaw: string | null;
  textSource: string | null;
  textConfidence: number | null;
  textEngineVersion: string | null;
  /** The VLM's / OCR engine's own candidate readings. */
  textVlm: string | null;
  textOcr: string | null;
  /** True when the two readers differ. */
  textDisagreement: boolean | null;
  /** Why the chosen reading won — `region_text_choice` vocabulary. */
  textChoice: string | null;
  /** Why the VLM's own reading was rejected as not text, when it was. */
  textVlmInvalid: string | null;
  clusterId: number | null;
  clusterSubid: string | null;
  clusterDistance: number | null;
  detectedAt: string | null;
  /** Served per-box thumbnail path (already carries `box_id`). */
  thumbnailUrl: string | null;
}

/** Served `GET /regions/statuses` `box_states` entry (W8.7). */
export interface BoxStateInfo {
  value: string;
  label: string;
  role: 'proposed' | 'accepted' | 'rejected' | 'false_positive';
  humanWritable: boolean;
  exported: boolean;
  dashed: boolean;
  dim: boolean;
  badge: string | null;
}

export interface SubBoxCapability {
  /**
   * EITHER `listField` (a multi-box list: the served region slot) OR
   * `bboxField` (a read-only single scalar box: a tier-2 non-region slot).
   * Never both — `readSlot` skips the scalar block when `listField` is
   * set. The backend has no write route for a scalar-box slot, so a
   * `bboxField` slot is display-only.
   *
   * Wire field holding the box as `[x1,y1,x2,y2]`. */
  bboxField?: WireField;
  /** Wire field holding the multi-box list on the item
   *  (`ItemDoc.region_boxes`) — `readSlot` populates `SlotData.subBoxes`
   *  from it (element keys are the fixed `RegionBoxWire` shape, see
   *  `SlotBox`), always an array (`[]` when none). */
  listField?: WireField;
  /** Frame the stored scalar box uses when `frameField` is absent or
   *  unreadable. Only meaningful with `bboxField`. */
  storedFrame?: SlotFrame;
  /** Optional wire field carrying the frame per-row. */
  frameField?: WireField;
  /** Detector confidence 0..1 (scalar-box slots). */
  scoreField?: WireField;
  /** Boolean "the thing is visible in this crop" (scalar-box slots). */
  visibleField?: WireField;
  /** Item-level summary of a `listField` set: number of boxes, number of
   *  rejected ones, best score, whether the set is complete (false = the
   *  VLM reported visible regions missing from the list) and the
   *  optimistic-concurrency revision every box write echoes back as
   *  `expected_region_revision`. */
  countField?: WireField;
  rejectedCountField?: WireField;
  maxScoreField?: WireField;
  setCompleteField?: WireField;
  revisionField?: WireField;
  /** Server-side crop of one box, used as the gallery card image. The
   *  box id is required: the backend 422s a region thumbnail without one. */
  thumbnail?: {
    path: (cropId: string, boxId: string, size: number) => string;
    aspect: string;
    defaultSize: number;
  };
  ring: SubBoxRing;
  editor: {
    thumbSize: number;
    viewPadding: number;
    nudgeStep: number;
  };
  /** The served `region_profile.limits.max_boxes_per_write` for a
   *  `listField` capability — the only real limit on adding a box (no
   *  client-guessed cap). */
  maxBoxesPerWrite?: number;
}

/* ------------------------------------------------------------------ */
/* Capability: text                                                    */
/* ------------------------------------------------------------------ */

export interface TextCapability {
  /** Item-level wire field for a scalar-box slot's reading. The served
   *  region slot declares none: its text is per box (`SlotBox.text`,
   *  written through `PATCH /crops/{id}/regions/{box_id}`). */
  valueField?: WireField;
  rawField?: WireField;
  sourceField?: WireField;
  confidenceField?: WireField;
  engineVersionField?: WireField;
  label: string;
  placeholder?: string;
  transform?: 'none' | 'uppercase' | 'lowercase' | 'trim';
  pattern?: RegExp;
  maxLength?: number;
  monospace?: boolean;
  /** Closed vocabulary. When present the editor renders a <select> /
   *  combobox instead of a free-text input, and `pattern` is ignored. */
  vocabulary?: Array<{ value: string; label: string; description?: string }>;
}

/* ------------------------------------------------------------------ */
/* Capability: provenance                                              */
/* ------------------------------------------------------------------ */

export interface ProvenanceCapability {
  /** Item-level detector id (scalar-box slots). The served region slot's
   *  detector is per box (`SlotBox.detector`), so it declares none. */
  detectorField?: WireField;
  detectorVersionField?: WireField;
  chainField?: WireField;
  verifierField?: WireField;
  verifierVersionField?: WireField;
  verifiedAtField?: WireField;
  detectedAtField?: WireField;
  showChainOnCard: boolean;
}

/* ------------------------------------------------------------------ */
/* Capability: lifecycle                                               */
/* ------------------------------------------------------------------ */

export interface SlotState {
  value: string;
  label: string;
  humanWritable: boolean;
  role?: 'proposed' | 'confirmed' | 'rejected' | 'falsePositive' | 'absent' | 'pending';
  dim?: boolean;
  badge?: string;
  /** Read-tolerance half of a wire-vocabulary migration: additional raw
   *  status values that resolve to this same state. `value` is always
   *  what the UI *writes*; `aliases` is what it *accepts* on read, so a
   *  backend enum rename (or old OpenSearch documents that still carry
   *  the pre-rename value) never resolves to a null state. See
   *  docs/design/slot-generic-crop-mapping-plan-2026-09-21.md §8.4(ii). */
  aliases?: string[];
}

export interface LifecycleCapability {
  statusField: WireField;
  /** Boolean "a verification pass ran" (human OR the VLM verifier).
   *  Regions: `region_verified`. This — not a status value — is the
   *  correct predicate for the confirmed ring (Finding C.3). Distinct
   *  from `validatedField` below (dq-region, 2026-09-24): a
   *  machine-auto-confirmed region is verified but not validated. */
  verifiedField?: WireField;
  /** Boolean "a HUMAN confirmed (or drew/rejected) this region" — never
   *  set by a machine verdict (dq-region, 2026-09-24). Regions:
   *  `region_validated`. Use this, not `verifiedField`, to badge "human
   *  reviewed" vs `autoConfirmedField`'s "machine accepted,
   *  unreviewed". */
  validatedField?: WireField;
  /** Boolean "the worker's auto-confirm policy accepted this box
   *  without a human" (dq-region, 2026-09-24) — an accepted-but-
   *  unreviewed region that still sits in the human review queue.
   *  Regions: `region_auto_confirmed`. */
  autoConfirmedField?: WireField;
  rejectionReasonField?: WireField;
  /** Who made a human write (e.g. `region_label_source`). Sent as
   *  `'human'` on batch status writes when declared. */
  labelSourceField?: WireField;
  states: SlotState[];
  confirmState: string;
  rejectState: string;
  falsePositiveState?: string;
}

/* ------------------------------------------------------------------ */
/* Capability: queue                                                   */
/* ------------------------------------------------------------------ */

export type SlotAction =
  'confirm' | 'reject' | 'markFalsePositive' | 'editBox' | 'back' | 'skip' | 'undo';

export interface QueueCapability {
  /** Backend cohort id — the `{id}` in `GET {API_PREFIX}/review/{id}`. */
  endpointId: string;
  /** URL value for `?tab=`. */
  urlId: string;
  tabLabel: string;
  browsePath: string;
  /** Keymap. A letter appearing here is automatically added to the
   *  derived reserved-hotkey set (see `../classHotkey.ts`). */
  keymap: Partial<Record<SlotAction, string[]>>;
  textFilter?: { param: string; label: string; placeholder: string };
  alwaysVisible: boolean;
}

/* ------------------------------------------------------------------ */
/* Write surface                                                       */
/* ------------------------------------------------------------------ */

export interface SlotEndpoints {
  patchMeta?: (cropId: string) => string;
  batchStatus?: () => string;
}

/* ------------------------------------------------------------------ */
/* The slot itself                                                     */
/* ------------------------------------------------------------------ */

export interface SlotSpec {
  key: SlotKey;
  /** Which class this slot hangs off. `className` is matched
   *  case-insensitively against `RegistryClass.name`; `classId` wins when both
   *  are given. At least one is required. */
  bind: { className?: string; classId?: number };
  label: {
    singular: string;
    plural: string;
    title: string;
  };
  capabilities: {
    subBox?: SubBoxCapability;
    text?: TextCapability;
    provenance?: ProvenanceCapability;
    lifecycle?: LifecycleCapability;
    queue?: QueueCapability;
    /** Optional — see `../cohorts.ts` (§9.2 of the plan's addendum).
     *  Declared here rather than in `cohorts.ts` to avoid a cycle
     *  (`cohorts.ts` imports `SlotSpec`, not the reverse). */
    trainingCohorts?: TrainingCohortsCapability;
  };
  endpoints: SlotEndpoints;
  stats?: {
    key: string;
    panelTitle: string;
    coverageTitle: string;
  };
  /** Profile-private extension bag for capabilities this pass deliberately
   *  does NOT generalize (secondary clustering, dataset export, training
   *  cohorts). Typed as `unknown` on purpose — escape hatch, not an
   *  extension point. */
  extras?: Record<string, unknown>;
}

/* ------------------------------------------------------------------ */
/* Runtime data                                                        */
/* ------------------------------------------------------------------ */

export interface SlotData {
  key: SlotKey;
  /** W8 multi-box list, present when the slot's `subBox.listField` is
   *  set and the backend serves it — always an array, `[]` when the item
   *  has no boxes, in stored (display) order. See `SlotBox`. */
  subBoxes?: SlotBox[];
  /** The item-level summary of `subBoxes` — see
   *  `SubBoxCapability.countField`. Present with `subBoxes`. */
  boxSet?: {
    count: number | null;
    rejectedCount: number | null;
    maxScore: number | null;
    setComplete: boolean | null;
    revision: number | null;
  };
  subBox?: {
    parent: BBoxNormLike | null;
    rawXyxy: XYXY | null;
    frame: SlotFrame;
    score: number | null;
    visible: boolean | null;
  };
  text?: {
    value: string | null;
    raw: string | null;
    source: string | null;
    confidence: number | null;
    engineVersion: string | null;
  };
  provenance?: {
    detector: string | null;
    detectorVersion: string | null;
    chain: string[] | null;
    verifier: string | null;
    verifierVersion: string | null;
    verifiedAt: string | null;
    detectedAt: string | null;
  };
  lifecycle?: {
    status: string | null;
    state: SlotState | null;
    verified: boolean | null;
    /** A human confirmed/drew/rejected this region — see
     *  `LifecycleCapability.validatedField`. */
    validated: boolean | null;
    /** The worker's auto-confirm policy accepted this box without a
     *  human — see `LifecycleCapability.autoConfirmedField`. */
    autoConfirmed: boolean | null;
    rejectionReason: string | null;
  };
}

/** True when the crop carries any evidence of this slot at all — i.e. at
 *  least one capability block has a non-null value on the wire, not
 *  merely that the capability is *configured* (every configured
 *  capability always produces a block, with fields null when absent on
 *  the row). */
export function slotIsPresent(d: SlotData | undefined | null): boolean {
  if (!d) return false;
  // W8: a genuinely populated multi-box list is evidence; an empty list
  // ([] — "no boxes on this item yet") is not, same as the legacy
  // single-box null case below.
  if (d.subBoxes && d.subBoxes.length > 0) return true;
  if (d.subBox && d.subBox.rawXyxy != null) return true;
  if (d.text && d.text.value != null) return true;
  if (d.provenance && d.provenance.detector != null) return true;
  if (d.lifecycle && d.lifecycle.status != null) return true;
  return false;
}
