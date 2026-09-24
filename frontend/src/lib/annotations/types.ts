/**
 * Annotation-slot type model.
 *
 * A "slot" is one secondary annotation attached to a crop — a sub-bbox,
 * a text attribute, a triage state, or any combination. `license_plate`
 * is the only slot configured today (`profiles/licensePlate.ts`); the
 * shape below is the *decomposition* of what it uses, not a
 * plate-shaped struct. See docs/genericization-plan-2026-09-13.md §2.5.
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
 *  by convention; used in URLs (`?tab=slot:license_plate`), storage keys,
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

export interface SubBoxCapability {
  /** Wire field holding the box as `[x1,y1,x2,y2]`. Plates: `region_bbox_norm`. */
  bboxField: WireField;
  /** Frame the stored box uses when `frameField` is absent or unreadable. */
  storedFrame: SlotFrame;
  /** Optional wire field carrying the frame per-row (plates: `region_bbox_frame`).
   *  When present and parseable it overrides `storedFrame` for that row. */
  frameField?: WireField;
  /** Detector confidence 0..1. Plates: `region_score`. */
  scoreField?: WireField;
  /** Boolean "the thing is visible in this crop". Plates: `region_visible`. */
  visibleField?: WireField;
  /** Optional wire field carrying the box already projected into the
   *  PARENT (crop-local) frame, server-computed (plates:
   *  `region_bbox_in_parent`). When present, `readSlot` renders from it
   *  directly instead of projecting `bboxField` through `parentXyxy`
   *  itself — preferring the server's own projection over a client one. */
  bboxInParentField?: WireField;
  /** Server-side crop of the sub-bbox region, used as the gallery card image. */
  thumbnail?: {
    path: (cropId: string, size: number) => string;
    aspect: string;
    defaultSize: number;
  };
  ring: SubBoxRing;
  editor: {
    thumbSize: number;
    viewPadding: number;
    nudgeStep: number;
  };
}

/* ------------------------------------------------------------------ */
/* Capability: text                                                    */
/* ------------------------------------------------------------------ */

export interface TextCapability {
  valueField: WireField;
  rawField?: WireField;
  sourceField?: WireField;
  confidenceField?: WireField;
  engineVersionField?: WireField;
  /** Wire field carrying the VLM's own reading, independent of `valueField`
   *  (the backend's chosen reading). Plates: `region_text_vlm`. */
  vlmValueField?: WireField;
  /** Wire field carrying the OCR engine's own reading. Plates: `region_text_ocr`. */
  ocrValueField?: WireField;
  /** Wire field: boolean, true when `vlmValueField` and `ocrValueField`
   *  disagree. Plates: `region_text_disagreement`. */
  disagreementField?: WireField;
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
  detectorField: WireField;
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
  /** Boolean "a human signed off". Plates: `region_verified`. This — not a
   *  status value — is the correct predicate for the confirmed ring
   *  (Finding C.3). */
  verifiedField?: WireField;
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
  | 'confirm'
  | 'reject'
  | 'markFalsePositive'
  | 'editBox'
  | 'back'
  | 'skip'
  | 'undo';

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
  setBox?: (cropId: string) => string;
  clearBox?: (cropId: string) => string;
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
    /** The VLM's own reading, when the slot declares `vlmValueField`. */
    vlmValue: string | null;
    /** The OCR engine's own reading, when the slot declares `ocrValueField`. */
    ocrValue: string | null;
    /** True when `vlmValue` and `ocrValue` disagree ("readers disagree"). */
    disagreement: boolean | null;
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
    rejectionReason: string | null;
  };
}

/** True when the crop carries any evidence of this slot at all — i.e. at
 *  least one capability block has a non-null value on the wire, not
 *  merely that the capability is *configured* (every configured
 *  capability always produces a block, with fields null when absent on
 *  the row). Replaces the hand-written `hasPlate` predicate at
 *  `CropMetaPanel.svelte:26-30`. */
export function slotIsPresent(d: SlotData | undefined | null): boolean {
  if (!d) return false;
  if (d.subBox && d.subBox.rawXyxy != null) return true;
  if (d.text && d.text.value != null) return true;
  if (d.provenance && d.provenance.detector != null) return true;
  if (d.lifecycle && d.lifecycle.status != null) return true;
  return false;
}
