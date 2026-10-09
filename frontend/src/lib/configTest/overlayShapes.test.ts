/**
 * `candidateShapes`: one box (and one polygon when a mask came back) per
 * candidate, from the served geometry of the requested frame; a dropped
 * candidate is dimmed; nothing is projected or recomputed.
 */
import { describe, expect, it } from 'vitest';
import { candidateFixture, legsFixture } from '$lib/test/fixtures/configTest';
import { candidateLabel, candidateShapes } from './overlayShapes';
import { refText } from './refText';

describe('candidateShapes', () => {
  it('emits the served source-frame boxes and polygons, labelled leg #index', () => {
    const shapes = candidateShapes(legsFixture());
    expect(shapes.map((s) => s.key)).toEqual([
      'detector:0:box',
      'detector:1:box',
      'segmenter:0:box',
      'segmenter:0:poly',
    ]);
    const first = shapes[0]!;
    expect(first.kind).toBe('box');
    if (first.kind === 'box') expect(first.box).toEqual([0.2, 0.2, 0.6, 0.5]);
    const poly = shapes[3]!;
    if (poly.kind === 'polygon') {
      expect(poly.points).toEqual([
        [0.2, 0.2],
        [0.6, 0.2],
        [0.4, 0.5],
      ]);
    } else {
      throw new Error('expected a polygon');
    }
    expect(first.label).toBe('detector #0');
    expect(candidateLabel('segmenter', 3)).toBe('segmenter #3');
  });

  it('dims exactly the dropped candidates and names the reason in the title', () => {
    const shapes = candidateShapes(legsFixture());
    const byKey = Object.fromEntries(shapes.map((s) => [s.key, s]));
    expect(byKey['detector:0:box']!.dimmed).toBe(false);
    expect(byKey['detector:1:box']!.dimmed).toBe(true);
    expect(byKey['detector:1:box']!.title).toContain('Below min score');
    expect(byKey['detector:0:box']!.title).not.toContain('dropped');
  });

  it('reads the server-projected parent-frame geometry for the crop frame', () => {
    const shapes = candidateShapes(legsFixture(), 'parent');
    const box = shapes.find((s) => s.key === 'detector:1:box')!;
    if (box.kind === 'box') expect(box.box).toEqual([0.6, 0.6, 0.8, 0.8]);
    const poly = shapes.find((s) => s.key === 'segmenter:0:poly')!;
    if (poly.kind === 'polygon') expect(poly.points[2]).toEqual([0.5, 0.9]);
  });

  it('skips a candidate with no geometry in the requested frame, and a polygon under 3 points', () => {
    const legs = [
      {
        leg: 'detector' as const,
        status: 'ok' as const,
        candidates: [
          candidateFixture({ bbox_in_parent: null, mask_polygon_in_parent: null }),
          candidateFixture({
            candidate_index: 1,
            bbox_in_parent: null,
            mask_polygon: [
              [0, 0],
              [1, 1],
            ],
          }),
        ],
      },
    ];
    expect(candidateShapes(legs, 'parent')).toEqual([]);
    expect(candidateShapes(legs).map((s) => s.kind)).toEqual(['box', 'box']);
  });

  it('a leg with no candidates (skipped or errored) contributes nothing', () => {
    expect(
      candidateShapes([{ leg: 'segmenter', status: 'skipped', reason: 'off' }]),
    ).toEqual([]);
  });
});

describe('refText', () => {
  it('prints name@revision, the bare name, or draft', () => {
    expect(refText({ draft: false, name: 'widget_tag', revision: 2 })).toBe(
      'widget_tag@2',
    );
    expect(refText({ draft: false, name: 'env_tags', revision: null })).toBe('env_tags');
    expect(refText({ draft: true, name: null, revision: null })).toBe('draft');
    expect(refText({ draft: true, name: 'widget_tag', revision: 2 })).toBe('draft');
  });
});
