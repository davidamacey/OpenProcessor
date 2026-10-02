import { describe, it, expect } from 'vitest';
import { readSlot, mapRegionBoxWire, mapRegionBoxList } from './readSlot';
import { slotIsPresent } from './types';
import { widgetTagSlot } from '$lib/test/fixtures/regionSlot';
import type { XYXY, SlotFrame } from './types';

describe('multi-box mapping (region_boxes)', () => {
  it('maps a full served box element, including its per-box text and cluster keys', () => {
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
      locked: true,
      text: 'TAG-001',
      text_raw: 'tag-001',
      text_source: 'gemma',
      text_confidence: 0.8,
      text_engine_version: '2',
      text_vlm: 'TAG-001',
      text_ocr: 'TAG-008',
      text_disagreement: true,
      text_choice: 'vlm_preferred',
      text_vlm_invalid: 'sequence',
      cluster_id: 12,
      cluster_subid: '12a',
      cluster_distance: 0.2,
      detected_at: '2026-10-01T00:00:00Z',
      thumbnail_url: '/curation/crops/c_123/region_thumbnail?box_id=b1',
    });
    expect(box).not.toBeNull();
    expect(box?.boxId).toBe('b1');
    expect(box?.state).toBe('accepted');
    expect(box?.rawXyxy).toEqual([0.41, 0.62, 0.47, 0.71]);
    expect(box?.parent).toEqual({ cx: 0.21, cy: 0.84, w: 0.18, h: 0.28 });
    expect(box?.text).toBe('TAG-001');
    expect(box?.textRaw).toBe('tag-001');
    expect(box?.textSource).toBe('gemma');
    expect(box?.textConfidence).toBeCloseTo(0.8);
    expect(box?.textEngineVersion).toBe('2');
    expect(box?.textVlm).toBe('TAG-001');
    expect(box?.textOcr).toBe('TAG-008');
    expect(box?.textDisagreement).toBe(true);
    expect(box?.textChoice).toBe('vlm_preferred');
    expect(box?.textVlmInvalid).toBe('sequence');
    expect(box?.locked).toBe(true);
    expect(box?.bboxCorrect).toBe(true);
    expect(box?.clusterId).toBe(12);
    expect(box?.clusterSubid).toBe('12a');
    expect(box?.clusterDistance).toBeCloseTo(0.2);
    expect(box?.detectedAt).toBe('2026-10-01T00:00:00Z');
    expect(box?.thumbnailUrl).toBe('/curation/crops/c_123/region_thumbnail?box_id=b1');
  });

  it('leaves the text keys null when the profile serves none (a text-free box)', () => {
    const box = mapRegionBoxWire({ box_id: 'b1', state: 'accepted' });
    expect(box?.text).toBeNull();
    expect(box?.textVlm).toBeNull();
    expect(box?.textDisagreement).toBeNull();
    expect(box?.locked).toBeNull();
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

  function oneBox(overrides: Record<string, unknown> = {}) {
    return {
      box_id: 'b1',
      state: 'accepted',
      bbox_norm: [0.1, 0.08, 0.3, 0.12],
      bbox_in_parent: [0.1, 0.08, 0.3, 0.12],
      score: 0.91,
      detector: null,
      detector_version: null,
      source: null,
      bbox_correct: null,
      confidence: null,
      rejection_reason: null,
      text: null,
      cluster_id: null,
      thumbnail_url: null,
      ...overrides,
    };
  }

  it('reads a full region row (region_boxes list + item summary) into every capability', () => {
    const raw = {
      region_boxes: [oneBox({ text: 'TAG-001' })],
      region_count: 1,
      region_rejected_count: 0,
      region_max_score: 0.91,
      region_set_complete: false,
      region_revision: 7,
      region_detector_chain: ['tag_detector_v1:hit'],
      region_verifier: 'gemma-4-e4b',
      region_status: 'detected',
      region_verified: true,
      region_rejection_reason: null,
    };
    const d = readSlot(raw, widgetTagSlot, parent);
    expect(slotIsPresent(d)).toBe(true);
    expect(d.subBoxes).toHaveLength(1);
    expect(d.subBoxes?.[0].rawXyxy).toEqual([0.1, 0.08, 0.3, 0.12]);
    expect(d.subBoxes?.[0].parent).not.toBeNull();
    expect(d.subBoxes?.[0].text).toBe('TAG-001');
    expect(d.boxSet).toEqual({
      count: 1,
      rejectedCount: 0,
      maxScore: 0.91,
      setComplete: false,
      revision: 7,
    });
    // The region's text and detector are per box: no item-level readings.
    expect(d.text).toBeUndefined();
    expect(d.provenance?.detector).toBeNull();
    expect(d.provenance?.chain).toEqual(['tag_detector_v1:hit']);
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
    expect(d.subBoxes).toEqual([]);
    expect(d.boxSet).toEqual({
      count: null,
      rejectedCount: null,
      maxScore: null,
      setComplete: null,
      revision: null,
    });
  });

  it('a box with no bbox_in_parent has no drawable crop-local geometry', () => {
    const raw = { region_boxes: [oneBox({ bbox_in_parent: null })] };
    const d = readSlot(raw, widgetTagSlot, parent);
    expect(d.subBoxes?.[0].rawXyxy).toEqual([0.1, 0.08, 0.3, 0.12]);
    expect(d.subBoxes?.[0].parent).toBeNull();
  });

  it('reads the served bbox_in_parent directly — no client-side projection', () => {
    const raw = {
      region_boxes: [oneBox({ bbox_in_parent: [0.4, 0.4, 0.6, 0.6] })],
    };
    const d = readSlot(raw, widgetTagSlot, parent);
    expect(d.subBoxes?.[0].parent?.cx).toBeCloseTo(0.5);
    expect(d.subBoxes?.[0].parent?.cy).toBeCloseTo(0.5);
    expect(d.subBoxes?.[0].parent?.w).toBeCloseTo(0.2);
    expect(d.subBoxes?.[0].parent?.h).toBeCloseTo(0.2);
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
});

describe('readSlot — rejected box / auto-confirm', () => {
  const parent: XYXY = [0, 0, 0.4, 0.2];

  it('reads a rejected box as a SlotBox with state "rejected" and its own rejection reason (no separate candidate concept, W8)', () => {
    const raw = {
      region_status: 'verify_rejected',
      region_boxes: [
        {
          box_id: 'b1',
          state: 'rejected',
          bbox_norm: [0.1, 0.08, 0.3, 0.12],
          bbox_in_parent: [0.1, 0.08, 0.3, 0.12],
          score: 0.42,
          detector: 'tag_detector_v1',
          detector_version: 'v3',
          source: 'detector',
          bbox_correct: null,
          confidence: null,
          rejection_reason: 'sanity_reject:aspect_ratio',
          text: null,
          cluster_id: null,
          thumbnail_url: null,
        },
      ],
    };
    const d = readSlot(raw, widgetTagSlot, parent);
    expect(d.subBoxes).toHaveLength(1);
    const box = d.subBoxes![0];
    expect(box.state).toBe('rejected');
    expect(box.rawXyxy).toEqual([0.1, 0.08, 0.3, 0.12]);
    expect(box.score).toBeCloseTo(0.42);
    expect(box.detector).toBe('tag_detector_v1');
    expect(box.detectorVersion).toBe('v3');
    expect(box.source).toBe('detector');
    // bbox_in_parent = [0.1,0.08,0.3,0.12] served directly, no projection.
    expect(box.parent?.cx).toBeCloseTo(0.2);
    expect(box.parent?.cy).toBeCloseTo(0.1);
    expect(box.rejectionReason).toBe('sanity_reject:aspect_ratio');
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
});

describe('readSlot — read-only scalar single-box slot (tier 2, bboxField)', () => {
  const cases: Array<{
    frame: SlotFrame;
    parentXyxy: XYXY;
    childXyxy: XYXY;
    expected: { cx: number; cy: number; w: number; h: number };
  }> = [
    {
      frame: 'source',
      parentXyxy: [0, 0, 0.4, 0.2],
      childXyxy: [0.1, 0.08, 0.3, 0.12],
      expected: { cx: 0.5, cy: 0.5, w: 0.5, h: 0.2 },
    },
    {
      frame: 'parent',
      parentXyxy: [0.2, 0.2, 0.6, 0.8],
      childXyxy: [0.3, 0.3, 0.5, 0.5],
      expected: { cx: 0.4, cy: 0.4, w: 0.2, h: 0.2 },
    },
  ];

  for (const { frame, parentXyxy, childXyxy, expected } of cases) {
    it(`projects the stored ${frame}-frame box into the crop frame`, () => {
      const spec = {
        ...widgetTagSlot,
        capabilities: {
          ...widgetTagSlot.capabilities,
          subBox: {
            bboxField: 'tail_bbox',
            storedFrame: frame,
            ring: { confirmed: '', proposed: '', rejected: '' },
            editor: { thumbSize: 512, viewPadding: 2.5, nudgeStep: 1 / 512 },
          },
        },
      };
      const d = readSlot({ tail_bbox: childXyxy }, spec, parentXyxy);
      expect(d.subBoxes).toBeUndefined();
      expect(d.boxSet).toBeUndefined();
      expect(d.subBox?.rawXyxy).toEqual(childXyxy);
      expect(d.subBox?.parent?.cx).toBeCloseTo(expected.cx, 9);
      expect(d.subBox?.parent?.cy).toBeCloseTo(expected.cy, 9);
      expect(d.subBox?.parent?.w).toBeCloseTo(expected.w, 9);
      expect(d.subBox?.parent?.h).toBeCloseTo(expected.h, 9);
    });
  }
});
