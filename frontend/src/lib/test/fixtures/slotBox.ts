import type { SlotBox } from '$lib/annotations/types';

/** A served region box with every field present (nulls where the wire
 *  would serve null); override what a test cares about. */
export function makeSlotBox(overrides: Partial<SlotBox> = {}): SlotBox {
  return {
    boxId: 'b1',
    state: 'proposed',
    rawXyxy: [0.1, 0.1, 0.2, 0.2],
    parent: { cx: 0.5, cy: 0.5, w: 0.2, h: 0.2 },
    score: 0.9,
    detector: 'tag_detector_v1',
    detectorVersion: '1',
    source: 'detector',
    bboxCorrect: null,
    confidence: null,
    rejectionReason: null,
    locked: null,
    text: null,
    textRaw: null,
    textSource: null,
    textConfidence: null,
    textEngineVersion: null,
    textVlm: null,
    textOcr: null,
    textDisagreement: null,
    textChoice: null,
    textVlmInvalid: null,
    clusterId: null,
    clusterSubid: null,
    clusterDistance: null,
    detectedAt: null,
    thumbnailUrl: null,
    ...overrides,
  };
}
