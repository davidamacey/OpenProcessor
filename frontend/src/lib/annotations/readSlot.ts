/**
 * Reads a `SlotSpec`'s capabilities off a raw crop JSON object.
 *
 * This is the field-mapping adapter (docs/genericization-plan-2026-09-13.md
 * §2.2/§2.6): each capability names the wire field(s) it reads, so this
 * function works for any slot whose fields exist on the raw payload,
 * without the caller needing a canonical schema.
 *
 * Not yet wired into `api.ts`'s `mapRawCrop` — this is Phase 1's
 * additive foundation (types + adapter + profile), landed independently
 * of the (larger, higher-risk) `Crop.slots` wiring and component
 * migration described in the plan's Phase 2. See the plan for the full
 * migration; this module is what that migration will call.
 */

import type { SlotSpec, SlotData, SlotBox, XYXY, SlotFrame, BBoxNormLike } from './types';

function pick(raw: Record<string, unknown>, field: string | undefined): unknown {
  return field == null ? undefined : raw[field];
}

function asXyxy(v: unknown): XYXY | null {
  if (!Array.isArray(v) || v.length !== 4) return null;
  const [a, b, c, d] = v;
  if (![a, b, c, d].every((n) => typeof n === 'number')) return null;
  return [a, b, c, d] as XYXY;
}

function asString(v: unknown): string | null {
  return typeof v === 'string' ? v : null;
}

function asNumber(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function asBoolean(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

function asStringArray(v: unknown): string[] | null {
  if (!Array.isArray(v)) return null;
  return v.every((x) => typeof x === 'string') ? (v as string[]) : null;
}

/** Projects a stored xyxy box (in `frame`) into the parent-crop frame. */
function projectToParent(
  childSourceXyxy: XYXY,
  parentSourceXyxy: XYXY,
  frame: SlotFrame,
): BBoxNormLike | null {
  if (frame === 'parent') {
    const [x1, y1, x2, y2] = childSourceXyxy;
    return { cx: (x1 + x2) / 2, cy: (y1 + y2) / 2, w: x2 - x1, h: y2 - y1 };
  }
  // frame === 'source': project through the parent box.
  const [vx1, vy1, vx2, vy2] = parentSourceXyxy;
  const vw = vx2 - vx1;
  const vh = vy2 - vy1;
  if (!(vw > 1e-9) || !(vh > 1e-9)) return null;
  const [px1, py1, px2, py2] = childSourceXyxy;
  return {
    cx: ((px1 + px2) / 2 - vx1) / vw,
    cy: ((py1 + py2) / 2 - vy1) / vh,
    w: (px2 - px1) / vw,
    h: (py2 - py1) / vh,
  };
}

/** Maps one `region_boxes` element (the served per-box wire shape, keys
 *  pinned to the vendored `RegionTestCandidate` schema by
 *  `contract/wireKeys.test.ts`) to a `SlotBox`. Exported for direct unit
 *  testing of the mapping independent of a full `readSlot` call. */
export function mapRegionBoxWire(el: unknown): SlotBox | null {
  if (el == null || typeof el !== 'object') return null;
  const r = el as Record<string, unknown>;
  const rawXyxy = asXyxy(r.bbox_norm);
  const parentXyxy = asXyxy(r.bbox_in_parent);
  const parent = parentXyxy
    ? {
        cx: (parentXyxy[0] + parentXyxy[2]) / 2,
        cy: (parentXyxy[1] + parentXyxy[3]) / 2,
        w: parentXyxy[2] - parentXyxy[0],
        h: parentXyxy[3] - parentXyxy[1],
      }
    : null;
  return {
    boxId: asString(r.box_id),
    state: asString(r.state) ?? 'proposed',
    rawXyxy,
    parent,
    score: asNumber(r.score),
    detector: asString(r.detector),
    detectorVersion: asString(r.detector_version),
    source: asString(r.source),
    bboxCorrect: asBoolean(r.bbox_correct),
    confidence: asString(r.confidence),
    rejectionReason: asString(r.rejection_reason),
    locked: asBoolean(r.locked),
    text: asString(r.text),
    textRaw: asString(r.text_raw),
    textSource: asString(r.text_source),
    textConfidence: asNumber(r.text_confidence),
    textEngineVersion: asString(r.text_engine_version),
    textVlm: asString(r.text_vlm),
    textOcr: asString(r.text_ocr),
    textDisagreement: asBoolean(r.text_disagreement),
    textChoice: asString(r.text_choice),
    textVlmInvalid: asString(r.text_vlm_invalid),
    clusterId: asNumber(r.cluster_id),
    clusterSubid: asString(r.cluster_subid),
    clusterDistance: asNumber(r.cluster_distance),
    detectedAt: asString(r.detected_at),
    thumbnailUrl: asString(r.thumbnail_url),
  };
}

/** Maps the whole `region_boxes` (or equivalent `listField`) array off a
 *  raw crop. Never throws on a malformed element — an element that isn't
 *  a plain object is dropped, so one bad row can't blank the whole list. */
export function mapRegionBoxList(raw: unknown): SlotBox[] {
  if (!Array.isArray(raw)) return [];
  const out: SlotBox[] = [];
  for (const el of raw) {
    const box = mapRegionBoxWire(el);
    if (box) out.push(box);
  }
  return out;
}

export function readSlot(
  raw: Record<string, unknown>,
  spec: SlotSpec,
  parentXyxy: XYXY,
): SlotData {
  const out: SlotData = { key: spec.key };
  const cap = spec.capabilities;

  if (cap.subBox?.listField) {
    // W8 multi-box list — always an array, [] when none (owner decision:
    // no backward compatibility with the pre-W8 scalar shape, so this is
    // the only region box path; the legacy single-box block below never
    // runs for a capability declaring listField — see next `if`).
    out.subBoxes = mapRegionBoxList(pick(raw, cap.subBox.listField));
    out.boxSet = {
      count: asNumber(pick(raw, cap.subBox.countField)),
      rejectedCount: asNumber(pick(raw, cap.subBox.rejectedCountField)),
      maxScore: asNumber(pick(raw, cap.subBox.maxScoreField)),
      setComplete: asBoolean(pick(raw, cap.subBox.setCompleteField)),
      revision: asNumber(pick(raw, cap.subBox.revisionField)),
    };
  }

  if (cap.subBox && cap.subBox.bboxField != null && cap.subBox.listField == null) {
    const rawXyxy = asXyxy(pick(raw, cap.subBox.bboxField));
    const frameRaw = cap.subBox.frameField
      ? asString(pick(raw, cap.subBox.frameField))
      : null;
    const frame: SlotFrame =
      frameRaw === 'parent' || frameRaw === 'source'
        ? frameRaw
        : (cap.subBox.storedFrame ?? 'source');
    out.subBox = {
      parent: rawXyxy ? projectToParent(rawXyxy, parentXyxy, frame) : null,
      rawXyxy,
      frame,
      score: asNumber(pick(raw, cap.subBox.scoreField)),
      visible: asBoolean(pick(raw, cap.subBox.visibleField)),
    };
  }

  if (cap.text?.valueField) {
    out.text = {
      value: asString(pick(raw, cap.text.valueField)),
      raw: asString(pick(raw, cap.text.rawField)),
      source: asString(pick(raw, cap.text.sourceField)),
      confidence: asNumber(pick(raw, cap.text.confidenceField)),
      engineVersion: asString(pick(raw, cap.text.engineVersionField)),
    };
  }

  if (cap.provenance) {
    out.provenance = {
      detector: asString(pick(raw, cap.provenance.detectorField)),
      detectorVersion: asString(pick(raw, cap.provenance.detectorVersionField)),
      chain: asStringArray(pick(raw, cap.provenance.chainField)),
      verifier: asString(pick(raw, cap.provenance.verifierField)),
      verifierVersion: asString(pick(raw, cap.provenance.verifierVersionField)),
      verifiedAt: asString(pick(raw, cap.provenance.verifiedAtField)),
      detectedAt: asString(pick(raw, cap.provenance.detectedAtField)),
    };
  }

  if (cap.lifecycle) {
    const status = asString(pick(raw, cap.lifecycle.statusField));
    const state = status
      ? (cap.lifecycle.states.find(
          (s) => s.value === status || (s.aliases?.includes(status) ?? false),
        ) ?? null)
      : null;
    out.lifecycle = {
      status,
      state,
      verified: asBoolean(pick(raw, cap.lifecycle.verifiedField)),
      validated: asBoolean(pick(raw, cap.lifecycle.validatedField)),
      autoConfirmed: asBoolean(pick(raw, cap.lifecycle.autoConfirmedField)),
      rejectionReason: asString(pick(raw, cap.lifecycle.rejectionReasonField)),
    };
  }

  return out;
}
