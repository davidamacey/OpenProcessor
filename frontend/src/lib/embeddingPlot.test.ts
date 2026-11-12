import { describe, expect, it } from 'vitest';
import { embeddingCoverageSuffix } from './embeddingPlot';
import {
  colourForIndex,
  computeScale,
  dragToBox,
  nearestPointId,
  PALETTE_SIZE,
  pointAlpha,
  pointInPolygon,
  selectIdsInLasso,
  type ScreenPoint,
  classifyRebuildPoll,
} from './embeddingPlot';

describe('computeScale', () => {
  it('maps points into [padding, dim-padding] on both axes', () => {
    const scale = computeScale(
      [
        { x: 0, y: 0 },
        { x: 10, y: 10 },
      ],
      100,
      100,
      10,
    );
    const a = scale.toScreen(0, 0);
    const b = scale.toScreen(10, 10);
    for (const p of [a, b]) {
      expect(p.x).toBeGreaterThanOrEqual(10);
      expect(p.x).toBeLessThanOrEqual(90);
      expect(p.y).toBeGreaterThanOrEqual(10);
      expect(p.y).toBeLessThanOrEqual(90);
    }
  });

  it('flips y so a larger data-space y renders higher on screen (smaller pixel y)', () => {
    const scale = computeScale(
      [
        { x: 0, y: 0 },
        { x: 0, y: 10 },
      ],
      100,
      100,
      0,
    );
    const low = scale.toScreen(0, 0);
    const high = scale.toScreen(0, 10);
    expect(high.y).toBeLessThan(low.y);
  });

  it('degrades to a fixed center point for an empty point list instead of throwing', () => {
    const scale = computeScale([], 200, 100, 10);
    expect(() => scale.toScreen(1, 1)).not.toThrow();
    expect(scale.toScreen(1, 1)).toEqual({ x: 100, y: 50 });
  });

  it('degrades to a fixed center point for a zero-size canvas instead of throwing', () => {
    const scale = computeScale([{ x: 1, y: 1 }], 0, 0, 10);
    expect(() => scale.toScreen(1, 1)).not.toThrow();
  });

  it('does not divide by zero when every point is coincident', () => {
    const scale = computeScale(
      [
        { x: 5, y: 5 },
        { x: 5, y: 5 },
      ],
      100,
      100,
      10,
    );
    const p = scale.toScreen(5, 5);
    expect(Number.isFinite(p.x)).toBe(true);
    expect(Number.isFinite(p.y)).toBe(true);
  });
});

describe('pointInPolygon', () => {
  const square: ScreenPoint[] = [
    { x: 0, y: 0 },
    { x: 10, y: 0 },
    { x: 10, y: 10 },
    { x: 0, y: 10 },
  ];

  it('reports a point inside the polygon as inside', () => {
    expect(pointInPolygon({ x: 5, y: 5 }, square)).toBe(true);
  });

  it('reports a point outside the polygon as outside', () => {
    expect(pointInPolygon({ x: 50, y: 50 }, square)).toBe(false);
  });

  it('returns false for fewer than 3 vertices (degenerate lasso path)', () => {
    expect(pointInPolygon({ x: 5, y: 5 }, [])).toBe(false);
    expect(pointInPolygon({ x: 5, y: 5 }, [{ x: 0, y: 0 }])).toBe(false);
    expect(
      pointInPolygon({ x: 5, y: 5 }, [
        { x: 0, y: 0 },
        { x: 10, y: 10 },
      ]),
    ).toBe(false);
  });
});

describe('selectIdsInLasso', () => {
  const square: ScreenPoint[] = [
    { x: 0, y: 0 },
    { x: 10, y: 0 },
    { x: 10, y: 10 },
    { x: 0, y: 10 },
  ];

  it('returns only the ids of points inside the lasso path', () => {
    const points = [
      { id: 'a', x: 5, y: 5 },
      { id: 'b', x: 50, y: 50 },
      { id: 'c', x: 2, y: 2 },
    ];
    expect(selectIdsInLasso(points, square).sort()).toEqual(['a', 'c']);
  });

  it('returns an empty array for a degenerate (< 3 vertex) lasso path', () => {
    const points = [{ id: 'a', x: 5, y: 5 }];
    expect(selectIdsInLasso(points, [])).toEqual([]);
    expect(
      selectIdsInLasso(points, [
        { x: 0, y: 0 },
        { x: 1, y: 1 },
      ]),
    ).toEqual([]);
  });

  it('returns an empty array when no points fall inside', () => {
    const points = [{ id: 'a', x: 500, y: 500 }];
    expect(selectIdsInLasso(points, square)).toEqual([]);
  });
});

describe('colourForIndex', () => {
  it('maps a served slot to a palette colour and wraps past the palette', () => {
    expect(colourForIndex(0)).toBe(colourForIndex(PALETTE_SIZE));
    expect(colourForIndex(1)).not.toBe(colourForIndex(0));
    expect(colourForIndex(-1)).toBe(colourForIndex(PALETTE_SIZE - 1));
  });
});

describe('pointAlpha', () => {
  it('dims a point the filter leaves out below every matched point', () => {
    expect(pointAlpha(false, false, false)).toBeLessThan(pointAlpha(true, false, true));
    expect(pointAlpha(false, true, true)).toBeLessThan(pointAlpha(true, false, true));
  });

  it('keeps the selection bright and the rest of the matched points muted', () => {
    expect(pointAlpha(true, true, true)).toBeGreaterThan(pointAlpha(true, false, true));
    expect(pointAlpha(true, false, false)).toBe(pointAlpha(true, true, true));
  });
});

describe('dragToBox', () => {
  const scale = computeScale(
    [
      { x: 0, y: 0 },
      { x: 10, y: 20 },
    ],
    120,
    120,
    10,
  );

  it('turns a drag into a data-space box whatever the drag direction', () => {
    const from = scale.toScreen(2, 4);
    const to = scale.toScreen(6, 12);
    const forward = dragToBox(scale, from, to);
    const backward = dragToBox(scale, to, from);
    expect(forward[0]).toBeCloseTo(2);
    expect(forward[1]).toBeCloseTo(4);
    expect(forward[2]).toBeCloseTo(6);
    expect(forward[3]).toBeCloseTo(12);
    expect(backward).toEqual(forward);
  });

  it('round-trips toScreen and toData', () => {
    const at = scale.toScreen(3, 7);
    const back = scale.toData(at.x, at.y);
    expect(back.x).toBeCloseTo(3);
    expect(back.y).toBeCloseTo(7);
  });
});

describe('nearestPointId', () => {
  const pts = [
    { id: 'a', x: 10, y: 10 },
    { id: 'b', x: 14, y: 10 },
  ];

  it('picks the nearest point inside the radius', () => {
    expect(nearestPointId(pts, { x: 13, y: 10 }, 8)).toBe('b');
  });

  it('is null when nothing is near enough', () => {
    expect(nearestPointId(pts, { x: 100, y: 100 }, 8)).toBeNull();
  });
});

describe('classifyRebuildPoll', () => {
  it('keeps polling while the job runs, tracked or not', () => {
    expect(classifyRebuildPoll('running', true)).toBe('running');
    expect(classifyRebuildPoll('running', false)).toBe('running');
  });

  it('keeps polling a queued, paused or cancelling job', () => {
    for (const s of ['queued', 'paused', 'cancelling']) {
      expect(classifyRebuildPoll(s, true)).toBe('running');
      expect(classifyRebuildPoll(s, false)).toBe('running');
    }
    expect(classifyRebuildPoll('interrupted', true)).toBe('failed');
  });

  it('reports the terminal status of a job this plot was watching', () => {
    expect(classifyRebuildPoll('completed', true)).toBe('completed');
    expect(classifyRebuildPoll('failed', true)).toBe('failed');
    expect(classifyRebuildPoll('cancelled', true)).toBe('idle');
  });

  it("ignores a terminal status for a job it never watched (someone else's history)", () => {
    expect(classifyRebuildPoll('completed', false)).toBe('idle');
    expect(classifyRebuildPoll('failed', false)).toBe('idle');
  });
});

describe('embeddingCoverageSuffix (m20, 2026-09-24 interactive pass)', () => {
  it('reads "points" plainly when there is no served pool total', () => {
    expect(embeddingCoverageSuffix(116, null)).toBe('points');
  });

  it('reads "points" when the pool total equals what is shown (full coverage)', () => {
    expect(embeddingCoverageSuffix(422, 422)).toBe('points');
  });

  it('notes the gap and points at Rebuild when the pool total exceeds what is shown', () => {
    expect(embeddingCoverageSuffix(116, 422)).toBe(
      'of 422 projected — Rebuild to include the rest',
    );
  });

  it('never claims a gap when the pool total is smaller than shown (stale/inconsistent data)', () => {
    expect(embeddingCoverageSuffix(500, 422)).toBe('points');
  });
});
