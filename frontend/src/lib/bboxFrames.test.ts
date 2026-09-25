/**
 * Unit tests for bboxFrames helpers.
 */

import { describe, expect, it } from 'vitest';
import { bboxNormToXYXY, cropToSourceFrame, sourceToCropFrame } from './bboxFrames';
import type { BBoxNorm } from './types';

const EPS = 1e-9;

function expectBoxApprox(actual: BBoxNorm, expected: BBoxNorm): void {
  for (const k of ['cx', 'cy', 'w', 'h'] as const) {
    expect(actual[k], `mismatch on ${k}`).toBeCloseTo(expected[k], 9);
  }
}

describe('bboxFrames', () => {
  it('round-trip: source -> crop -> source returns the input', () => {
    // Item occupies the lower-right quadrant of the source image.
    const item: BBoxNorm = { cx: 0.6, cy: 0.7, w: 0.4, h: 0.3 };
    // Region sits inside that item in source-frame coords.
    const regionSrc: BBoxNorm = { cx: 0.62, cy: 0.78, w: 0.06, h: 0.02 };
    const boxInCrop = sourceToCropFrame(regionSrc, item);
    expect(boxInCrop).not.toBeNull();
    const back = cropToSourceFrame(boxInCrop!, item);
    expectBoxApprox(back, regionSrc);
  });

  it('off-center item: region in crop frame uses parent box origin', () => {
    // Item: x1=0.2, y1=0.4, x2=0.6, y2=0.8 (center 0.4/0.6, w=0.4, h=0.4).
    const item: BBoxNorm = { cx: 0.4, cy: 0.6, w: 0.4, h: 0.4 };
    // Region in source frame: covers the bottom-center of the item.
    // x1=0.36, y1=0.74, x2=0.44, y2=0.78 → cx=0.40, cy=0.76, w=0.08, h=0.04.
    const regionSrc: BBoxNorm = { cx: 0.4, cy: 0.76, w: 0.08, h: 0.04 };
    const boxInCrop = sourceToCropFrame(regionSrc, item);
    expect(boxInCrop).not.toBeNull();
    // In crop frame:
    //   x1' = (0.36 - 0.2) / 0.4 = 0.40
    //   y1' = (0.74 - 0.4) / 0.4 = 0.85
    //   x2' = (0.44 - 0.2) / 0.4 = 0.60
    //   y2' = (0.78 - 0.4) / 0.4 = 0.95
    // → cx=0.50, cy=0.90, w=0.20, h=0.10
    expectBoxApprox(boxInCrop!, { cx: 0.5, cy: 0.9, w: 0.2, h: 0.1 });
  });

  it('degenerate item (zero width) returns null', () => {
    const item: BBoxNorm = { cx: 0.5, cy: 0.5, w: 0, h: 0.4 };
    const region: BBoxNorm = { cx: 0.5, cy: 0.5, w: 0.05, h: 0.02 };
    expect(sourceToCropFrame(region, item)).toBeNull();
  });

  it('degenerate item (zero height) returns null', () => {
    const item: BBoxNorm = { cx: 0.5, cy: 0.5, w: 0.4, h: 0 };
    const region: BBoxNorm = { cx: 0.5, cy: 0.5, w: 0.05, h: 0.02 };
    expect(sourceToCropFrame(region, item)).toBeNull();
  });

  it('clamping at edges: region extending past item is clipped to [0,1] in crop frame', () => {
    // Item covers the whole image so source==crop frame.
    const item: BBoxNorm = { cx: 0.5, cy: 0.5, w: 1.0, h: 1.0 };
    // Region that extends well past the right edge.
    const region: BBoxNorm = { cx: 0.95, cy: 0.5, w: 0.4, h: 0.1 };
    const out = sourceToCropFrame(region, item);
    expect(out).not.toBeNull();
    // Corners should clamp to x1=0.75, x2=1.0 → cx=0.875, w=0.25.
    expectBoxApprox(out!, { cx: 0.875, cy: 0.5, w: 0.25, h: 0.1 });
    // Every component must be inside [0, 1].
    for (const k of ['cx', 'cy', 'w', 'h'] as const) {
      expect(out![k], `${k} out of [0,1]`).toBeGreaterThanOrEqual(0);
      expect(out![k], `${k} out of [0,1]`).toBeLessThanOrEqual(1);
    }
  });

  it('cropToSourceFrame clamps when the converted box would exceed the image', () => {
    // Tiny item near the right edge; "region" in crop frame extends
    // beyond the right side, so the source-frame result must clamp.
    const item: BBoxNorm = { cx: 0.95, cy: 0.5, w: 0.1, h: 0.1 };
    const boxInCrop: BBoxNorm = { cx: 0.9, cy: 0.5, w: 0.6, h: 0.2 };
    const out = cropToSourceFrame(boxInCrop, item);
    // Item corners (source frame): x1=0.90, y1=0.45, x2=1.00, y2=0.55.
    // Region-in-crop corners: x1=0.6, y1=0.4, x2=1.2, y2=0.6.
    // Source-frame raw corners:
    //   x1' = 0.90 + 0.6*0.10 = 0.96
    //   x2' = 0.90 + 1.2*0.10 = 1.02   → clamp to 1.00
    //   y1' = 0.45 + 0.4*0.10 = 0.49
    //   y2' = 0.45 + 0.6*0.10 = 0.51
    // → cx=0.98, cy=0.50, w=0.04, h=0.02
    expect(out.cx + out.w / 2).toBeLessThanOrEqual(1 + EPS);
    expect(out.cx - out.w / 2).toBeGreaterThanOrEqual(-EPS);
    expectBoxApprox(out, { cx: 0.98, cy: 0.5, w: 0.04, h: 0.02 });
  });

  it('bboxNormToXYXY produces clamped [x1,y1,x2,y2] tuple', () => {
    const b: BBoxNorm = { cx: 0.5, cy: 0.5, w: 0.4, h: 0.2 };
    const t = bboxNormToXYXY(b);
    expect(t).toHaveLength(4);
    expect(t[0]).toBeCloseTo(0.3, 9);
    expect(t[1]).toBeCloseTo(0.4, 9);
    expect(t[2]).toBeCloseTo(0.7, 9);
    expect(t[3]).toBeCloseTo(0.6, 9);
  });
});
