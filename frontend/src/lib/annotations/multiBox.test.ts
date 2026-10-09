import { describe, it, expect } from 'vitest';
import {
  toEditableBoxes,
  nextBoxIndex,
  removeBoxAt,
  addBox,
  buildRegionsPutBoxes,
  confirmProposedBoxes,
  hasAcceptedBox,
  type EditableBox,
} from './multiBox';
import { makeSlotBox } from '$lib/test/fixtures/slotBox';

const box = makeSlotBox;

describe('toEditableBoxes', () => {
  it('drops boxes with no parent-frame projection', () => {
    const boxes = [box({ boxId: 'b1' }), box({ boxId: 'b2', parent: null })];
    const editable = toEditableBoxes(boxes);
    expect(editable).toHaveLength(1);
    expect(editable[0].boxId).toBe('b1');
    expect(editable[0].dirty).toBe(false);
  });

  it('handles an unbounded number of boxes (no client cap)', () => {
    const many = Array.from({ length: 200 }, (_, i) => box({ boxId: `b${i}` }));
    expect(toEditableBoxes(many)).toHaveLength(200);
  });
});

describe('nextBoxIndex (Tab / box_edit.next_box)', () => {
  it('selects the first box when nothing is selected', () => {
    expect(nextBoxIndex(3, null)).toBe(0);
  });
  it('wraps around past the last box', () => {
    expect(nextBoxIndex(3, 2)).toBe(0);
  });
  it('returns null when there are no boxes', () => {
    expect(nextBoxIndex(0, null)).toBeNull();
  });
});

describe('removeBoxAt (Backspace/Delete)', () => {
  const set: EditableBox[] = [
    {
      boxId: 'b1',
      state: 'accepted',
      box: { cx: 0.1, cy: 0.1, w: 0.1, h: 0.1 },
      dirty: false,
    },
    {
      boxId: 'b2',
      state: 'proposed',
      box: { cx: 0.5, cy: 0.5, w: 0.1, h: 0.1 },
      dirty: false,
    },
    {
      boxId: 'b3',
      state: 'rejected',
      box: { cx: 0.9, cy: 0.9, w: 0.1, h: 0.1 },
      dirty: false,
    },
  ];

  it('removes the middle box and keeps selection at the same index', () => {
    const { boxes, selected } = removeBoxAt(set, 1);
    expect(boxes.map((b) => b.boxId)).toEqual(['b1', 'b3']);
    expect(selected).toBe(1);
  });

  it('clamps selection when the last box is removed', () => {
    const { boxes, selected } = removeBoxAt(set, 2);
    expect(boxes).toHaveLength(2);
    expect(selected).toBe(1);
  });

  it('returns null selection when the last remaining box is removed', () => {
    const one: EditableBox[] = [set[0]];
    const { boxes, selected } = removeBoxAt(one, 0);
    expect(boxes).toHaveLength(0);
    expect(selected).toBeNull();
  });
});

describe('addBox', () => {
  it('appends unbounded — no client cap', () => {
    let boxes: EditableBox[] = [];
    for (let i = 0; i < 500; i++) {
      const result = addBox(boxes, { cx: 0.1, cy: 0.1, w: 0.05, h: 0.05 });
      boxes = result.boxes;
    }
    expect(boxes).toHaveLength(500);
    expect(boxes[499].boxId).toBeNull();
    expect(boxes[499].dirty).toBe(true);
  });

  it('defaults a new box to accepted state, matching new_box_default', () => {
    const { boxes } = addBox([], { cx: 0.5, cy: 0.5, w: 0.1, h: 0.1 });
    expect(boxes[0].state).toBe('accepted');
  });
});

describe('buildRegionsPutBoxes (W8.8 write shape)', () => {
  const original: EditableBox[] = [
    {
      boxId: 'b1',
      state: 'proposed',
      box: { cx: 0.1, cy: 0.1, w: 0.1, h: 0.1 },
      dirty: false,
    },
    {
      boxId: 'b2',
      state: 'rejected',
      box: { cx: 0.5, cy: 0.5, w: 0.1, h: 0.1 },
      dirty: false,
    },
  ];

  it('sends an untouched box as {box_id} alone', () => {
    const result = buildRegionsPutBoxes(original, original);
    expect(result).toEqual([{ box_id: 'b1' }, { box_id: 'b2' }]);
  });

  it('sends a moved box as {box_id, bbox_norm}, keeping its state out of the body', () => {
    const edited = original.map((b) =>
      b.boxId === 'b1'
        ? { ...b, box: { cx: 0.2, cy: 0.2, w: 0.1, h: 0.1 }, dirty: true }
        : b,
    );
    const result = buildRegionsPutBoxes(original, edited);
    expect(result[0].box_id).toBe('b1');
    expect((result[0] as { bbox_norm: number[] }).bbox_norm).toEqual([
      0.15000000000000002, 0.15000000000000002, 0.25, 0.25,
    ]);
    expect(result[1]).toEqual({ box_id: 'b2' });
  });

  it('sends a state-only change as {box_id, state}', () => {
    const edited = original.map((b) =>
      b.boxId === 'b2' ? { ...b, state: 'accepted' } : b,
    );
    const result = buildRegionsPutBoxes(original, edited);
    expect(result[1]).toEqual({ box_id: 'b2', state: 'accepted' });
  });

  it('sends both bbox_norm and state when a box is moved AND its state changed', () => {
    const edited = original.map((b) =>
      b.boxId === 'b1'
        ? {
            ...b,
            box: { cx: 0.3, cy: 0.3, w: 0.1, h: 0.1 },
            state: 'accepted',
            dirty: true,
          }
        : b,
    );
    const result = buildRegionsPutBoxes(original, edited);
    expect(result[0]).toEqual({
      box_id: 'b1',
      bbox_norm: [0.25, 0.25, 0.35, 0.35],
      state: 'accepted',
    });
  });

  it('sends a new box as {box_id: null, bbox_norm} with no state (server default)', () => {
    const edited = original.concat([
      {
        boxId: null,
        state: 'accepted',
        box: { cx: 0.7, cy: 0.7, w: 0.1, h: 0.1 },
        dirty: true,
      },
    ]);
    const result = buildRegionsPutBoxes(original, edited);
    expect(result[2].box_id).toBeNull();
    expect((result[2] as { bbox_norm: number[] }).bbox_norm).toEqual([
      0.6499999999999999, 0.6499999999999999, 0.75, 0.75,
    ]);
  });

  it('omits a stored box that is absent from the edited set (delete)', () => {
    const edited = [original[0]];
    const result = buildRegionsPutBoxes(original, edited);
    expect(result).toEqual([{ box_id: 'b1' }]);
  });
});

describe('confirmProposedBoxes (Enter — owner decision)', () => {
  it('flips only proposed boxes to accepted, leaving rejected/false_positive untouched', () => {
    const boxes: EditableBox[] = [
      { boxId: 'b1', state: 'proposed', box: null, dirty: false },
      { boxId: 'b2', state: 'rejected', box: null, dirty: false },
      { boxId: 'b3', state: 'false_positive', box: null, dirty: false },
      { boxId: 'b4', state: 'accepted', box: null, dirty: false },
    ];
    const result = confirmProposedBoxes(boxes);
    expect(result.map((b) => b.state)).toEqual([
      'accepted',
      'rejected',
      'false_positive',
      'accepted',
    ]);
  });
});

describe('hasAcceptedBox', () => {
  it('is false when every box is rejected (server would 422 no_accepted_box)', () => {
    const boxes: EditableBox[] = [
      { boxId: 'b1', state: 'rejected', box: null, dirty: false },
    ];
    expect(hasAcceptedBox(boxes)).toBe(false);
  });
  it('is true with at least one accepted box', () => {
    const boxes: EditableBox[] = [
      { boxId: 'b1', state: 'accepted', box: null, dirty: false },
    ];
    expect(hasAcceptedBox(boxes)).toBe(true);
  });
});
