import { describe, expect, it } from 'vitest';
import { labelOpSubtitle, labelOpTitle } from './labelOpText';
import type { LabelOp } from '$lib/types_labelOps';

const op = (over: Partial<LabelOp>): LabelOp => ({
  op_id: 'op-1',
  kind: 'label',
  at: '2026-11-12T23:06:00+00:00',
  finished_at: null,
  class_name: 'bottle',
  target_cluster_uid: 'cl-3',
  label_source: 'human',
  target: 'selection',
  selected: 24,
  count: 24,
  conflicts: 0,
  current: 24,
  status: 'written',
  undone: false,
  undoable: true,
  last_undo: null,
  ...over,
});

describe('labelOpTitle', () => {
  it('says what was done to how many items', () => {
    expect(labelOpTitle(op({}))).toBe('Labelled 24 items as bottle');
    expect(labelOpTitle(op({ kind: 'move', count: 3 }))).toBe('Moved 3 items to bottle');
  });
  it('uses the singular for one item, never "1 items"', () => {
    expect(labelOpTitle(op({ count: 1 }))).toBe('Labelled 1 item as bottle');
  });
  it('falls back to the selected count while the count is unknown', () => {
    expect(labelOpTitle(op({ count: null, selected: 5 }))).toBe(
      'Labelled 5 items as bottle',
    );
  });
  it('names the assist loop and the detector adoption', () => {
    expect(labelOpTitle(op({ kind: 'adopt_detector', class_name: null }))).toBe(
      "Adopted the detector's classes for 24 items",
    );
  });

  it('names a bulk accept of confident machine labels', () => {
    expect(labelOpTitle(op({ kind: 'accept_machine', class_name: null }))).toBe(
      'Accepted the confident machine labels of 24 items',
    );
  });
});

describe('labelOpSubtitle', () => {
  it('names the cluster, who did it and when', () => {
    expect(labelOpSubtitle(op({}))).toBe(
      'from cluster cl-3 · by a person · 2026-11-12 23:06:00 UTC',
    );
  });
  it('leaves out what was not served', () => {
    expect(labelOpSubtitle(op({ target_cluster_uid: null, label_source: 'vlm' }))).toBe(
      'by VLM · 2026-11-12 23:06:00 UTC',
    );
  });
});
