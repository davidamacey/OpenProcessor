import { describe, expect, it } from 'vitest';
import { dropReasonText, hitShapes, optionLabel } from './hitShapes';
import { testResponseFixture, vocabularyFixture } from './fixtures';

const VOCAB = vocabularyFixture();

describe('hitShapes', () => {
  const shapes = hitShapes(testResponseFixture().hits, VOCAB.drop_reasons);

  it('draws one box per hit and a polygon only for a hit with a mask', () => {
    expect(shapes.map((s) => `${s.kind}:${s.key}`)).toEqual([
      'box:hit:0:box',
      'polygon:hit:0:poly',
      'box:hit:1:box',
    ]);
  });

  it('keeps the served geometry untouched', () => {
    const box = shapes[0]!;
    expect(box.kind === 'box' && box.box).toEqual([0.1, 0.1, 0.4, 0.5]);
  });

  it('dims a dropped hit and names its score and its served reason label', () => {
    expect(shapes[0]!.dimmed).toBe(false);
    const dropped = shapes[2]!;
    expect(dropped.dimmed).toBe(true);
    expect(dropped.label).toBe('hit #1');
    expect(dropped.title).toContain('0.62');
    expect(dropped.title).toContain('Matches an item already there');
  });

  it('skips a box that is not four numbers', () => {
    const out = hitShapes([{ bbox_norm: [0.1, 0.2], score: 0.5, selected: true }], []);
    expect(out).toEqual([]);
  });
});

describe('optionLabel / dropReasonText', () => {
  it('reads the label from the served options', () => {
    expect(optionLabel(VOCAB.gate_reasons, 'vlm_no')).toBe('The VLM pre-check said no');
    expect(dropReasonText('nms', VOCAB.drop_reasons)).toBe('Overlapped a better hit');
  });

  it('prints an id the options do not list verbatim, never a made-up label', () => {
    expect(dropReasonText('brand_new_reason' as never, VOCAB.drop_reasons)).toBe(
      'brand_new_reason',
    );
    expect(dropReasonText('nms', [])).toBe('nms');
  });

  it('is empty for no reason', () => {
    expect(dropReasonText(null, VOCAB.drop_reasons)).toBe('');
    expect(dropReasonText(undefined, VOCAB.drop_reasons)).toBe('');
  });
});
