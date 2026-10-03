import { describe, expect, it } from 'vitest';
import { dropReasonText, hitShapes } from './hitShapes';
import { testResponseFixture } from './fixtures';

describe('hitShapes', () => {
  const shapes = hitShapes(testResponseFixture().hits);

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

  it('dims a dropped hit and names its reason and score', () => {
    expect(shapes[0]!.dimmed).toBe(false);
    const dropped = shapes[2]!;
    expect(dropped.dimmed).toBe(true);
    expect(dropped.label).toBe('hit #1');
    expect(dropped.title).toContain('0.62');
    expect(dropped.title).toContain('Agrees with an existing item');
  });

  it('skips a box that is not four numbers', () => {
    const out = hitShapes([{ bbox_norm: [0.1, 0.2], score: 0.5, selected: true }]);
    expect(out).toEqual([]);
  });
});

describe('dropReasonText', () => {
  it('words agree_existing and humanises the rest', () => {
    expect(dropReasonText('agree_existing')).toBe('Agrees with an existing item');
    expect(dropReasonText('below_min_score')).toBe('Below min score');
    expect(dropReasonText(null)).toBe('');
  });
});
