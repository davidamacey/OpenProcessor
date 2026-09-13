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
    // Vehicle occupies the lower-right quadrant of the source image.
    const vehicle: BBoxNorm = { cx: 0.6, cy: 0.7, w: 0.4, h: 0.3 };
    // Plate sits inside that vehicle in source-frame coords.
    const plateSrc: BBoxNorm = { cx: 0.62, cy: 0.78, w: 0.06, h: 0.02 };
    const plateInCrop = sourceToCropFrame(plateSrc, vehicle);
    expect(plateInCrop).not.toBeNull();
    const back = cropToSourceFrame(plateInCrop!, vehicle);
    expectBoxApprox(back, plateSrc);
  });

  it('off-center vehicle: plate in crop frame uses parent box origin', () => {
    // Vehicle: x1=0.2, y1=0.4, x2=0.6, y2=0.8 (center 0.4/0.6, w=0.4, h=0.4).
    const vehicle: BBoxNorm = { cx: 0.4, cy: 0.6, w: 0.4, h: 0.4 };
    // Plate in source frame: covers the bottom-center of the vehicle.
    // x1=0.36, y1=0.74, x2=0.44, y2=0.78 → cx=0.40, cy=0.76, w=0.08, h=0.04.
    const plateSrc: BBoxNorm = { cx: 0.4, cy: 0.76, w: 0.08, h: 0.04 };
    const plateInCrop = sourceToCropFrame(plateSrc, vehicle);
    expect(plateInCrop).not.toBeNull();
    // In crop frame:
    //   x1' = (0.36 - 0.2) / 0.4 = 0.40
    //   y1' = (0.74 - 0.4) / 0.4 = 0.85
    //   x2' = (0.44 - 0.2) / 0.4 = 0.60
    //   y2' = (0.78 - 0.4) / 0.4 = 0.95
    // → cx=0.50, cy=0.90, w=0.20, h=0.10
    expectBoxApprox(plateInCrop!, { cx: 0.5, cy: 0.9, w: 0.2, h: 0.1 });
  });

  it('degenerate vehicle (zero width) returns null', () => {
    const vehicle: BBoxNorm = { cx: 0.5, cy: 0.5, w: 0, h: 0.4 };
    const plate: BBoxNorm = { cx: 0.5, cy: 0.5, w: 0.05, h: 0.02 };
    expect(sourceToCropFrame(plate, vehicle)).toBeNull();
  });

  it('degenerate vehicle (zero height) returns null', () => {
    const vehicle: BBoxNorm = { cx: 0.5, cy: 0.5, w: 0.4, h: 0 };
    const plate: BBoxNorm = { cx: 0.5, cy: 0.5, w: 0.05, h: 0.02 };
    expect(sourceToCropFrame(plate, vehicle)).toBeNull();
  });

  it('clamping at edges: plate extending past vehicle is clipped to [0,1] in crop frame', () => {
    // Vehicle covers the whole image so source==crop frame.
    const vehicle: BBoxNorm = { cx: 0.5, cy: 0.5, w: 1.0, h: 1.0 };
    // Plate that extends well past the right edge.
    const plate: BBoxNorm = { cx: 0.95, cy: 0.5, w: 0.4, h: 0.1 };
    const out = sourceToCropFrame(plate, vehicle);
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
    // Tiny vehicle near the right edge; "plate" in crop frame extends
    // beyond the right side, so the source-frame result must clamp.
    const vehicle: BBoxNorm = { cx: 0.95, cy: 0.5, w: 0.1, h: 0.1 };
    const plateInCrop: BBoxNorm = { cx: 0.9, cy: 0.5, w: 0.6, h: 0.2 };
    const out = cropToSourceFrame(plateInCrop, vehicle);
    // Vehicle corners (source frame): x1=0.90, y1=0.45, x2=1.00, y2=0.55.
    // Plate-in-crop corners: x1=0.6, y1=0.4, x2=1.2, y2=0.6.
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
