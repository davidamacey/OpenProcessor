/**
 * Unit tests for plate_geometry helpers.
 *
 * The labeler doesn't (yet) ship a configured test runner — `package.json`
 * has `check` (svelte-check) and `lint` but no `test` script — so this
 * file exposes a self-contained harness instead of importing vitest. The
 * tests are still callable from any future runner via `runAll()`.
 *
 *   import { runAll } from './plate_geometry.test';
 *   runAll(); // throws on first failure
 */

import {
  bboxNormToXYXY,
  cropToSourceFrame,
  sourceToCropFrame,
} from './plate_geometry';
import type { BBoxNorm } from './types';

// -- tiny assertion harness ---------------------------------------------

class AssertionError extends Error {}

function assert(cond: unknown, msg: string): asserts cond {
  if (!cond) throw new AssertionError(msg);
}

function approxEqual(a: number, b: number, eps = 1e-9): boolean {
  return Math.abs(a - b) <= eps;
}

function assertBoxApprox(
  actual: BBoxNorm,
  expected: BBoxNorm,
  eps = 1e-9,
  label = '',
): void {
  for (const k of ['cx', 'cy', 'w', 'h'] as const) {
    if (!approxEqual(actual[k], expected[k], eps)) {
      throw new AssertionError(
        `${label} mismatch on ${k}: got ${actual[k]}, expected ${expected[k]}`,
      );
    }
  }
}

// -- test cases ---------------------------------------------------------

interface NamedTest {
  name: string;
  run: () => void;
}

export const tests: NamedTest[] = [
  {
    name: 'round-trip: source -> crop -> source returns the input',
    run: () => {
      // Vehicle occupies the lower-right quadrant of the source image.
      const vehicle: BBoxNorm = { cx: 0.6, cy: 0.7, w: 0.4, h: 0.3 };
      // Plate sits inside that vehicle in source-frame coords.
      const plateSrc: BBoxNorm = { cx: 0.62, cy: 0.78, w: 0.06, h: 0.02 };
      const plateInCrop = sourceToCropFrame(plateSrc, vehicle);
      assert(plateInCrop != null, 'expected non-null plate in crop frame');
      const back = cropToSourceFrame(plateInCrop, vehicle);
      assertBoxApprox(back, plateSrc, 1e-9, 'round-trip');
    },
  },
  {
    name: 'off-center vehicle: plate in crop frame uses parent box origin',
    run: () => {
      // Vehicle: x1=0.2, y1=0.4, x2=0.6, y2=0.8 (center 0.4/0.6, w=0.4, h=0.4).
      const vehicle: BBoxNorm = { cx: 0.4, cy: 0.6, w: 0.4, h: 0.4 };
      // Plate in source frame: covers the bottom-center of the vehicle.
      // x1=0.36, y1=0.74, x2=0.44, y2=0.78 → cx=0.40, cy=0.76, w=0.08, h=0.04.
      const plateSrc: BBoxNorm = { cx: 0.4, cy: 0.76, w: 0.08, h: 0.04 };
      const plateInCrop = sourceToCropFrame(plateSrc, vehicle);
      assert(plateInCrop != null, 'expected non-null plate in crop frame');
      // In crop frame:
      //   x1' = (0.36 - 0.2) / 0.4 = 0.40
      //   y1' = (0.74 - 0.4) / 0.4 = 0.85
      //   x2' = (0.44 - 0.2) / 0.4 = 0.60
      //   y2' = (0.78 - 0.4) / 0.4 = 0.95
      // → cx=0.50, cy=0.90, w=0.20, h=0.10
      assertBoxApprox(
        plateInCrop,
        { cx: 0.5, cy: 0.9, w: 0.2, h: 0.1 },
        1e-9,
        'off-center vehicle source->crop',
      );
    },
  },
  {
    name: 'degenerate vehicle (zero width) returns null',
    run: () => {
      const vehicle: BBoxNorm = { cx: 0.5, cy: 0.5, w: 0, h: 0.4 };
      const plate: BBoxNorm = { cx: 0.5, cy: 0.5, w: 0.05, h: 0.02 };
      const out = sourceToCropFrame(plate, vehicle);
      assert(out === null, 'expected null for zero-width vehicle');
    },
  },
  {
    name: 'degenerate vehicle (zero height) returns null',
    run: () => {
      const vehicle: BBoxNorm = { cx: 0.5, cy: 0.5, w: 0.4, h: 0 };
      const plate: BBoxNorm = { cx: 0.5, cy: 0.5, w: 0.05, h: 0.02 };
      const out = sourceToCropFrame(plate, vehicle);
      assert(out === null, 'expected null for zero-height vehicle');
    },
  },
  {
    name: 'clamping at edges: plate extending past vehicle is clipped to [0,1] in crop frame',
    run: () => {
      // Vehicle covers the whole image so source==crop frame.
      const vehicle: BBoxNorm = { cx: 0.5, cy: 0.5, w: 1.0, h: 1.0 };
      // Plate that extends well past the right edge.
      const plate: BBoxNorm = { cx: 0.95, cy: 0.5, w: 0.4, h: 0.1 };
      const out = sourceToCropFrame(plate, vehicle);
      assert(out != null, 'expected non-null result');
      // Corners should clamp to x1=0.75, x2=1.0 → cx=0.875, w=0.25.
      assertBoxApprox(
        out,
        { cx: 0.875, cy: 0.5, w: 0.25, h: 0.1 },
        1e-9,
        'clamped plate',
      );
      // Every component must be inside [0, 1].
      for (const k of ['cx', 'cy', 'w', 'h'] as const) {
        assert(out[k] >= 0 && out[k] <= 1, `${k} out of [0,1]`);
      }
    },
  },
  {
    name: 'cropToSourceFrame clamps when the converted box would exceed the image',
    run: () => {
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
      assert(out.cx + out.w / 2 <= 1 + 1e-9, 'right edge must clamp to <=1');
      assert(out.cx - out.w / 2 >= -1e-9, 'left edge must clamp to >=0');
      assertBoxApprox(
        out,
        { cx: 0.98, cy: 0.5, w: 0.04, h: 0.02 },
        1e-9,
        'cropToSourceFrame edge clamp',
      );
    },
  },
  {
    name: 'bboxNormToXYXY produces clamped [x1,y1,x2,y2] tuple',
    run: () => {
      const b: BBoxNorm = { cx: 0.5, cy: 0.5, w: 0.4, h: 0.2 };
      const t = bboxNormToXYXY(b);
      assert(t.length === 4, 'tuple must be length 4');
      assert(approxEqual(t[0], 0.3), `x1=${t[0]}`);
      assert(approxEqual(t[1], 0.4), `y1=${t[1]}`);
      assert(approxEqual(t[2], 0.7), `x2=${t[2]}`);
      assert(approxEqual(t[3], 0.6), `y2=${t[3]}`);
    },
  },
];

export function runAll(): { passed: number; failed: number } {
  let passed = 0;
  let failed = 0;
  for (const t of tests) {
    try {
      t.run();
      passed += 1;
    } catch (e) {
      failed += 1;
      // eslint-disable-next-line no-console
      console.error(`FAIL: ${t.name}\n  ${(e as Error).message}`);
    }
  }
  return { passed, failed };
}
