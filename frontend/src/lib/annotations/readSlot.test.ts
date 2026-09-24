import { describe, it, expect } from 'vitest';
import { readSlot, projectFromParent } from './readSlot';
import { slotIsPresent } from './types';
import { licensePlateSlot } from './profiles/licensePlate';
import type { XYXY, BBoxNormLike, SlotFrame } from './types';

describe('readSlot / licensePlateSlot', () => {
  const parent: XYXY = [0, 0, 0.4, 0.2]; // vw=0.4, vh=0.2

  it('reads a full plate row into every capability', () => {
    const raw = {
      region_bbox_norm: [0.1, 0.08, 0.3, 0.12],
      region_bbox_frame: 'source',
      region_score: 0.91,
      region_visible: true,
      region_text: 'ABC123',
      region_text_raw: 'abc123',
      region_text_source: 'gemma',
      region_text_confidence: 0.8,
      region_detector: 'lpr_nanov11_640',
      region_detector_chain: ['lpr_nanov11_640:hit'],
      region_verifier: 'gemma-4-e4b',
      region_status: 'detected',
      region_verified: true,
      region_rejection_reason: null,
    };
    const d = readSlot(raw, licensePlateSlot, parent);
    expect(slotIsPresent(d)).toBe(true);
    expect(d.subBox?.rawXyxy).toEqual([0.1, 0.08, 0.3, 0.12]);
    expect(d.subBox?.parent).not.toBeNull();
    expect(d.subBox?.shapeWarning).toBe(false);
    expect(d.text?.value).toBe('ABC123');
    expect(d.provenance?.detector).toBe('lpr_nanov11_640');
    expect(d.lifecycle?.status).toBe('detected');
    expect(d.lifecycle?.state?.role).toBe('proposed');
    expect(d.lifecycle?.verified).toBe(true);
  });

  it('resolves an unknown lifecycle status to a null state without crashing (forward tolerance)', () => {
    const raw = { region_status: 'some_future_status' };
    const d = readSlot(raw, licensePlateSlot, parent);
    expect(d.lifecycle?.status).toBe('some_future_status');
    expect(d.lifecycle?.state).toBeNull();
  });

  it('yields no slot data at all when no plate fields are present', () => {
    const d = readSlot({}, licensePlateSlot, parent);
    expect(slotIsPresent(d)).toBe(false);
    expect(d.subBox?.rawXyxy).toBeNull();
    expect(d.text?.value).toBeNull();
  });

  it('flags an implausible plate shape via the shared shape gate', () => {
    // aspect too narrow: w/vw small relative to h/vh
    const raw = { region_bbox_norm: [0.15, 0.02, 0.2, 0.18] };
    const d = readSlot(raw, licensePlateSlot, parent);
    expect(d.subBox?.shapeWarning).toBe(true);
  });

  it('resolves a legacy status value via aliases to the state a rename declares', () => {
    const spec = {
      ...licensePlateSlot,
      capabilities: {
        ...licensePlateSlot.capabilities,
        lifecycle: {
          ...licensePlateSlot.capabilities.lifecycle!,
          states: licensePlateSlot.capabilities.lifecycle!.states.map((s) =>
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
        ...licensePlateSlot,
        capabilities: {
          ...licensePlateSlot.capabilities,
          subBox: { ...licensePlateSlot.capabilities.subBox!, storedFrame: frame },
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
