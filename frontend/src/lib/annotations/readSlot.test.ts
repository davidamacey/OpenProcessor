import { describe, it, expect } from 'vitest';
import { readSlot } from './readSlot';
import { slotIsPresent } from './types';
import { licensePlateSlot } from './profiles/licensePlate';
import type { XYXY } from './types';

describe('readSlot / licensePlateSlot', () => {
  const parent: XYXY = [0, 0, 0.4, 0.2]; // vw=0.4, vh=0.2

  it('reads a full plate row into every capability', () => {
    const raw = {
      plate_bbox_norm: [0.1, 0.08, 0.3, 0.12],
      plate_bbox_frame: 'source',
      plate_score: 0.91,
      plate_visible: true,
      plate_text: 'ABC123',
      plate_text_raw: 'abc123',
      plate_text_source: 'gemma',
      plate_text_confidence: 0.8,
      plate_detector: 'lpr_nanov11_640',
      plate_detector_chain: ['lpr_nanov11_640:hit'],
      plate_verifier: 'gemma-4-e4b',
      plate_status: 'detected',
      plate_verified: true,
      plate_rejection_reason: null,
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
    const raw = { plate_status: 'some_future_status' };
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
    const raw = { plate_bbox_norm: [0.15, 0.02, 0.2, 0.18] };
    const d = readSlot(raw, licensePlateSlot, parent);
    expect(d.subBox?.shapeWarning).toBe(true);
  });
});
