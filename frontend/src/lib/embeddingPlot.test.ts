import { describe, expect, it } from 'vitest';
import { embeddingCoverageSuffix } from './embeddingPlot';
import {
  colorForCluster,
  computeScale,
  pointInPolygon,
  selectIdsInLasso,
  type ScreenPoint,
  classifyRebuildPoll,
  legendEntries,
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

describe('colorForCluster', () => {
  it('is deterministic — the same cluster_id always returns the same color', () => {
    expect(colorForCluster(17)).toBe(colorForCluster(17));
  });

  it('returns a distinct neutral color for null (unassigned)', () => {
    const unassigned = colorForCluster(null);
    // Distinct from at least a spread sample of real cluster ids — not
    // asserting against the whole palette, just that null doesn't
    // silently collide with a "real" cluster color.
    for (const id of [0, 1, 2, 3, 10173, 99999]) {
      expect(colorForCluster(id)).not.toBe(unassigned);
    }
  });

  it('never throws on a non-finite or negative cluster_id', () => {
    expect(() => colorForCluster(Number.NaN)).not.toThrow();
    expect(() => colorForCluster(Number.POSITIVE_INFINITY)).not.toThrow();
    expect(() => colorForCluster(-5)).not.toThrow();
    expect(colorForCluster(Number.NaN)).toBe(colorForCluster(null));
  });

  it('produces more than one distinct color across a spread of cluster ids', () => {
    const colors = new Set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11].map(colorForCluster));
    expect(colors.size).toBeGreaterThan(1);
  });
});

describe('classifyRebuildPoll', () => {
  it('keeps polling while the job runs, tracked or not', () => {
    expect(classifyRebuildPoll('running', true)).toBe('running');
    expect(classifyRebuildPoll('running', false)).toBe('running');
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

describe('legendEntries (F-68)', () => {
  it('lists the biggest clusters with their color and most common class name', () => {
    const pts = [
      { cluster_id: 3, class_name: 'widget_a' },
      { cluster_id: 3, class_name: 'widget_a' },
      { cluster_id: 3, class_name: 'widget_b' },
      { cluster_id: 7, class_name: null },
      { cluster_id: null, class_name: null },
      { cluster_id: null, class_name: null },
    ];
    const e = legendEntries(pts, 2);
    expect(e).toEqual([
      { clusterId: 3, color: colorForCluster(3), count: 3, className: 'widget_a' },
      { clusterId: null, color: colorForCluster(null), count: 2, className: null },
    ]);
  });
});
