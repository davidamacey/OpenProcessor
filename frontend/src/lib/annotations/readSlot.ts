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

import type { SlotSpec, SlotData, XYXY, SlotFrame, BBoxNormLike } from './types';

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
  // frame === 'source': project through the parent box, matching
  // sourceToCropFrame's convention (see bboxFrames.ts).
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

/**
 * Inverse of `projectToParent`: given a box already expressed in the
 * PARENT crop's frame (`{cx,cy,w,h}`), returns it as `[x1,y1,x2,y2]` in
 * the frame the slot actually stores (`'source'` or `'parent'`).
 *
 * Needed for saving an edited box back to the wire: the editor UI always
 * works in parent-crop-normalized coordinates, but a slot may store
 * `'source'`-frame boxes, so a straight write would silently corrupt the
 * geometry. `/review` and `SlotBboxEditor.svelte` both hand-rolled this
 * via `cropToSourceFrame`, which hardcodes `'source'` — this is the one
 * documented, slot-generic way to do it.
 */
export function projectFromParent(
  parentFrameBox: BBoxNormLike,
  parentSourceXyxy: XYXY,
  frame: SlotFrame,
): XYXY {
  const { cx, cy, w, h } = parentFrameBox;
  if (frame === 'parent') {
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2];
  }
  // frame === 'source': un-project through the parent box.
  const [vx1, vy1, vx2, vy2] = parentSourceXyxy;
  const vw = vx2 - vx1;
  const vh = vy2 - vy1;
  const px1 = vx1 + (cx - w / 2) * vw;
  const py1 = vy1 + (cy - h / 2) * vh;
  const px2 = vx1 + (cx + w / 2) * vw;
  const py2 = vy1 + (cy + h / 2) * vh;
  return [px1, py1, px2, py2];
}

export function readSlot(
  raw: Record<string, unknown>,
  spec: SlotSpec,
  parentXyxy: XYXY,
): SlotData {
  const out: SlotData = { key: spec.key };
  const cap = spec.capabilities;

  if (cap.subBox) {
    const rawXyxy = asXyxy(pick(raw, cap.subBox.bboxField));
    const frameRaw = cap.subBox.frameField
      ? asString(pick(raw, cap.subBox.frameField))
      : null;
    const frame: SlotFrame =
      frameRaw === 'parent' || frameRaw === 'source' ? frameRaw : cap.subBox.storedFrame;
    // Prefer the server's own parent-frame projection when it sent one
    // (regions: region_bbox_in_parent) over projecting rawXyxy ourselves —
    // one less place client and server geometry can disagree.
    const servedParentXyxy = cap.subBox.bboxInParentField
      ? asXyxy(pick(raw, cap.subBox.bboxInParentField))
      : null;
    const parent = servedParentXyxy
      ? {
          cx: (servedParentXyxy[0] + servedParentXyxy[2]) / 2,
          cy: (servedParentXyxy[1] + servedParentXyxy[3]) / 2,
          w: servedParentXyxy[2] - servedParentXyxy[0],
          h: servedParentXyxy[3] - servedParentXyxy[1],
        }
      : rawXyxy
        ? projectToParent(rawXyxy, parentXyxy, frame)
        : null;
    // Candidate: a verifier-rejected box, only meaningful when there's no
    // real box (mutually exclusive on the wire — see SubBoxCapability's
    // candidateBboxField doc comment). Same server-projection preference
    // as the main box above.
    const candidateRawXyxy = asXyxy(pick(raw, cap.subBox.candidateBboxField));
    const servedCandidateParentXyxy = cap.subBox.candidateBboxInParentField
      ? asXyxy(pick(raw, cap.subBox.candidateBboxInParentField))
      : null;
    const candidateParent = servedCandidateParentXyxy
      ? {
          cx: (servedCandidateParentXyxy[0] + servedCandidateParentXyxy[2]) / 2,
          cy: (servedCandidateParentXyxy[1] + servedCandidateParentXyxy[3]) / 2,
          w: servedCandidateParentXyxy[2] - servedCandidateParentXyxy[0],
          h: servedCandidateParentXyxy[3] - servedCandidateParentXyxy[1],
        }
      : candidateRawXyxy
        ? projectToParent(candidateRawXyxy, parentXyxy, frame)
        : null;
    out.subBox = {
      parent,
      rawXyxy,
      frame,
      score: asNumber(pick(raw, cap.subBox.scoreField)),
      visible: asBoolean(pick(raw, cap.subBox.visibleField)),
      candidate: candidateRawXyxy
        ? {
            parent: candidateParent,
            rawXyxy: candidateRawXyxy,
            score: asNumber(pick(raw, cap.subBox.candidateScoreField)),
            detector: asString(pick(raw, cap.subBox.candidateDetectorField)),
            detectorVersion: asString(
              pick(raw, cap.subBox.candidateDetectorVersionField),
            ),
            source: asString(pick(raw, cap.subBox.candidateSourceField)),
          }
        : null,
    };
  }

  if (cap.text) {
    out.text = {
      value: asString(pick(raw, cap.text.valueField)),
      raw: asString(pick(raw, cap.text.rawField)),
      source: asString(pick(raw, cap.text.sourceField)),
      confidence: asNumber(pick(raw, cap.text.confidenceField)),
      engineVersion: asString(pick(raw, cap.text.engineVersionField)),
      vlmValue: asString(pick(raw, cap.text.vlmValueField)),
      ocrValue: asString(pick(raw, cap.text.ocrValueField)),
      disagreement: asBoolean(pick(raw, cap.text.disagreementField)),
      choice: asString(pick(raw, cap.text.choiceField)),
      invalidReason: asString(pick(raw, cap.text.invalidReasonField)),
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
      boxCorrect: asBoolean(pick(raw, cap.lifecycle.boxCorrectField)),
    };
  }

  return out;
}
