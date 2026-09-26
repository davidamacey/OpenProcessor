import { describe, it, expect } from 'vitest';
import {
  readSlot,
  projectFromParent,
  mapRegionBoxWire,
  mapRegionBoxList,
} from './readSlot';
import { slotIsPresent } from './types';
import { widgetTagSlot } from '$lib/test/fixtures/regionSlot';
import type { XYXY, BBoxNormLike, SlotFrame } from './types';

describe('W8 multi-box mapping (region_boxes)', () => {
  it('maps a full RegionBoxWire element', () => {
    const box = mapRegionBoxWire({
      box_id: 'b1',
      state: 'accepted',
      bbox_norm: [0.41, 0.62, 0.47, 0.71],
      bbox_in_parent: [0.12, 0.7, 0.3, 0.98],
      score: 0.88,
      detector: 'sam3',
      detector_version: '1',
      source: 'segmenter',
      bbox_correct: true,
      confidence: 'high',
      rejection_reason: null,
      text: 'TAG-001',
      cluster_id: 12,
      thumbnail_url: '/curation/crops/c_123/region_thumbnail?box_id=b1',
    });
    expect(box).not.toBeNull();
    expect(box?.boxId).toBe('b1');
    expect(box?.state).toBe('accepted');
    expect(box?.rawXyxy).toEqual([0.41, 0.62, 0.47, 0.71]);
    expect(box?.parent).toEqual({ cx: 0.21, cy: 0.84, w: 0.18, h: 0.28 });
    expect(box?.text).toBe('TAG-001');
    expect(box?.clusterId).toBe(12);
  });

  it('drops a malformed element instead of throwing', () => {
    expect(
      mapRegionBoxList(['not-an-object', { box_id: 'b1', state: 'proposed' }]),
    ).toHaveLength(1);
  });

  it('returns [] for a missing/absent list (no boxes on this item)', () => {
    expect(mapRegionBoxList(undefined)).toEqual([]);
    expect(mapRegionBoxList(null)).toEqual([]);
  });

  it('readSlot populates subBoxes for an unbounded number of boxes', () => {
    const parent: XYXY = [0, 0, 1, 1];
    const region_boxes = Array.from({ length: 12 }, (_, i) => ({
      box_id: `b${i}`,
      state: i % 2 === 0 ? 'accepted' : 'rejected',
      bbox_norm: [0.1, 0.1, 0.2, 0.2],
    }));
    const d = readSlot({ region_boxes }, widgetTagSlot, parent);
    expect(d.subBoxes).toHaveLength(12);
    expect(d.subBoxes?.[1].state).toBe('rejected');
  });
});

describe('readSlot / widgetTagSlot', () => {
  const parent: XYXY = [0, 0, 0.4, 0.2]; // vw=0.4, vh=0.2

  it('reads a full region row into every capability', () => {
    const raw = {
      region_bbox_norm: [0.1, 0.08, 0.3, 0.12],
      region_bbox_frame: 'source',
      region_score: 0.91,
      region_visible: true,
      region_text: 'TAG-001',
      region_text_raw: 'tag-001',
      region_text_source: 'gemma',
      region_text_confidence: 0.8,
      region_detector: 'tag_detector_v1',
      region_detector_chain: ['tag_detector_v1:hit'],
      region_verifier: 'gemma-4-e4b',
      region_status: 'detected',
      region_verified: true,
      region_rejection_reason: null,
    };
    const d = readSlot(raw, widgetTagSlot, parent);
    expect(slotIsPresent(d)).toBe(true);
    expect(d.subBox?.rawXyxy).toEqual([0.1, 0.08, 0.3, 0.12]);
    expect(d.subBox?.parent).not.toBeNull();
    expect(d.text?.value).toBe('TAG-001');
    expect(d.provenance?.detector).toBe('tag_detector_v1');
    expect(d.lifecycle?.status).toBe('detected');
    expect(d.lifecycle?.state?.role).toBe('proposed');
    expect(d.lifecycle?.verified).toBe(true);
  });

  it('resolves an unknown lifecycle status to a null state without crashing (forward tolerance)', () => {
    const raw = { region_status: 'some_future_status' };
    const d = readSlot(raw, widgetTagSlot, parent);
    expect(d.lifecycle?.status).toBe('some_future_status');
    expect(d.lifecycle?.state).toBeNull();
  });

  it('yields no slot data at all when no region fields are present', () => {
    const d = readSlot({}, widgetTagSlot, parent);
    expect(slotIsPresent(d)).toBe(false);
    expect(d.subBox?.rawXyxy).toBeNull();
    expect(d.text?.value).toBeNull();
  });

  it('prefers the server-projected bboxInParentField over its own projection', () => {
    // Deliberately inconsistent with region_bbox_norm/parent so the
    // assertion only passes if bboxInParentField actually won.
    const raw = {
      region_bbox_norm: [0.1, 0.08, 0.3, 0.12],
      region_bbox_in_parent: [0.4, 0.4, 0.6, 0.6],
    };
    const d = readSlot(raw, widgetTagSlot, parent);
    expect(d.subBox?.parent?.cx).toBeCloseTo(0.5);
    expect(d.subBox?.parent?.cy).toBeCloseTo(0.5);
    expect(d.subBox?.parent?.w).toBeCloseTo(0.2);
    expect(d.subBox?.parent?.h).toBeCloseTo(0.2);
  });

  it('falls back to its own projection when bboxInParentField is absent', () => {
    const raw = { region_bbox_norm: [0.1, 0.08, 0.3, 0.12] };
    const d = readSlot(raw, widgetTagSlot, parent);
    // vw=0.4, vh=0.2 (parent): cx=(0.2/0.4)=0.5, cy=(0.1/0.2)=0.5
    expect(d.subBox?.parent?.cx).toBeCloseTo(0.5);
    expect(d.subBox?.parent?.cy).toBeCloseTo(0.5);
  });

  it('resolves a legacy status value via aliases to the state a rename declares', () => {
    const spec = {
      ...widgetTagSlot,
      capabilities: {
        ...widgetTagSlot.capabilities,
        lifecycle: {
          ...widgetTagSlot.capabilities.lifecycle!,
          states: widgetTagSlot.capabilities.lifecycle!.states.map((s) =>
            s.value === 'no_region_visible' ? { ...s, aliases: ['legacy_absent'] } : s,
          ),
        },
      },
    };
    const d = readSlot({ region_status: 'legacy_absent' }, spec, parent);
    expect(d.lifecycle?.status).toBe('legacy_absent');
    expect(d.lifecycle?.state?.value).toBe('no_region_visible');
    expect(d.lifecycle?.state?.role).toBe('absent');
  });

  it('reads the VLM/OCR text candidates and the disagreement flag (2026-09-24 logic-moves W8)', () => {
    const raw = {
      region_text: 'TAG-001',
      region_text_vlm: 'TAG-001',
      region_text_ocr: 'TAG-008',
      region_text_disagreement: true,
      region_text_engine_version: '1',
    };
    const d = readSlot(raw, widgetTagSlot, parent);
    expect(d.text?.vlmValue).toBe('TAG-001');
    expect(d.text?.ocrValue).toBe('TAG-008');
    expect(d.text?.disagreement).toBe(true);
    expect(d.text?.engineVersion).toBe('1');
  });

  it('leaves the OCR candidate fields null when the wire omits them', () => {
    const d = readSlot({ region_text: 'TAG-001' }, widgetTagSlot, parent);
    expect(d.text?.vlmValue).toBeNull();
    expect(d.text?.ocrValue).toBeNull();
    expect(d.text?.disagreement).toBeNull();
  });
});

describe('readSlot — dq-region candidate box / auto-confirm / text choice (2026-09-24)', () => {
  const parent: XYXY = [0, 0, 0.4, 0.2];

  it('reads a verify_rejected candidate box when there is no main box', () => {
    const raw = {
      region_bbox_norm: null,
      region_status: 'verify_rejected',
      region_rejection_reason: 'sanity_reject:aspect_ratio',
      region_candidate_bbox_norm: [0.1, 0.08, 0.3, 0.12],
      region_candidate_score: 0.42,
      region_candidate_detector: 'tag_detector_v1',
      region_candidate_detector_version: 'v3',
      region_candidate_source: 'detector',
    };
    const d = readSlot(raw, widgetTagSlot, parent);
    expect(d.subBox?.rawXyxy).toBeNull();
    expect(d.subBox?.candidate).not.toBeNull();
    expect(d.subBox?.candidate?.rawXyxy).toEqual([0.1, 0.08, 0.3, 0.12]);
    expect(d.subBox?.candidate?.score).toBeCloseTo(0.42);
    expect(d.subBox?.candidate?.detector).toBe('tag_detector_v1');
    expect(d.subBox?.candidate?.detectorVersion).toBe('v3');
    expect(d.subBox?.candidate?.source).toBe('detector');
    // Projected into the parent frame the same way the main box is.
    expect(d.subBox?.candidate?.parent?.cx).toBeCloseTo(0.5);
    expect(d.lifecycle?.rejectionReason).toBe('sanity_reject:aspect_ratio');
  });

  it('prefers the server-projected candidateBboxInParentField over its own projection', () => {
    const raw = {
      region_candidate_bbox_norm: [0.1, 0.08, 0.3, 0.12],
      region_candidate_bbox_in_parent: [0.4, 0.4, 0.6, 0.6],
    };
    const d = readSlot(raw, widgetTagSlot, parent);
    expect(d.subBox?.candidate?.parent?.cx).toBeCloseTo(0.5);
    expect(d.subBox?.candidate?.parent?.w).toBeCloseTo(0.2);
  });

  it('has no candidate when candidateBboxField is absent', () => {
    const raw = { region_bbox_norm: [0.1, 0.08, 0.3, 0.12] };
    const d = readSlot(raw, widgetTagSlot, parent);
    expect(d.subBox?.candidate).toBeNull();
  });

  it('reads region_validated as human-only validation, separate from region_verified', () => {
    const raw = {
      region_status: 'detected',
      region_verified: true,
      region_validated: false,
      region_auto_confirmed: true,
    };
    const d = readSlot(raw, widgetTagSlot, parent);
    expect(d.lifecycle?.verified).toBe(true);
    expect(d.lifecycle?.validated).toBe(false);
    expect(d.lifecycle?.autoConfirmed).toBe(true);
  });

  it('reads the text-choice and vlm-invalid-reason fields', () => {
    const raw = {
      region_text: 'TAG-001',
      region_text_choice: 'vlm_preferred',
      region_text_vlm_invalid: null,
    };
    const d = readSlot(raw, widgetTagSlot, parent);
    expect(d.text?.choice).toBe('vlm_preferred');
    expect(d.text?.invalidReason).toBeNull();
  });

  it('reads a vlm_invalid text choice with its reason', () => {
    const raw = {
      region_text: '123456',
      region_text_choice: 'vlm_invalid',
      region_text_vlm_invalid: 'sequence',
    };
    const d = readSlot(raw, widgetTagSlot, parent);
    expect(d.text?.choice).toBe('vlm_invalid');
    expect(d.text?.invalidReason).toBe('sequence');
  });
});

describe('projectFromParent — inverse of the private projectToParent', () => {
  const cases: Array<{ frame: SlotFrame; parentXyxy: XYXY; childSourceXyxy: XYXY }> = [
    {
      frame: 'source',
      parentXyxy: [0, 0, 0.4, 0.2],
      childSourceXyxy: [0.1, 0.08, 0.3, 0.12],
    },
    {
      frame: 'parent',
      parentXyxy: [0.2, 0.2, 0.6, 0.8],
      childSourceXyxy: [0.3, 0.3, 0.5, 0.5],
    },
  ];

  for (const { frame, parentXyxy, childSourceXyxy } of cases) {
    it(`round-trips through readSlot's forward projection (${frame} frame)`, () => {
      const spec = {
        ...widgetTagSlot,
        capabilities: {
          ...widgetTagSlot.capabilities,
          subBox: { ...widgetTagSlot.capabilities.subBox!, storedFrame: frame },
        },
      };
      const raw = { region_bbox_norm: childSourceXyxy };
      const d = readSlot(raw, spec, parentXyxy);
      const parentFrameBox = d.subBox!.parent as BBoxNormLike;
      const back = projectFromParent(parentFrameBox, parentXyxy, frame);
      for (let i = 0; i < 4; i++) {
        expect(back[i]).toBeCloseTo(childSourceXyxy[i], 9);
      }
    });
  }
});
