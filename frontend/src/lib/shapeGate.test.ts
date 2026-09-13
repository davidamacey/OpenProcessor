import { describe, it, expect } from 'vitest';
import { evaluateShapeGate, describeEnvelope, PLATE_SHAPE_ENVELOPE } from './shapeGate';

// Case table shared by both former call sites (api.ts's `_platesShapeWarning`
// and PlateCard.svelte's inline `shapeWarning()`), which used to disagree on
// non-finite input — see docs/genericization-plan-2026-09-13.md Finding C.1.
// Parent (vehicle) box is a unit square at various sizes; child (plate) box
// is expressed in the same absolute frame.
describe('evaluateShapeGate', () => {
  const parent: [number, number, number, number] = [0, 0, 1, 0.5]; // vw=1, vh=0.5

  it('passes a plausible wide plate', () => {
    // child w/vw = 0.3, h/vh = 0.1 -> aspect 3, area 0.03
    const child: [number, number, number, number] = [0.1, 0.2, 0.4, 0.25];
    expect(evaluateShapeGate(child, parent, PLATE_SHAPE_ENVELOPE)).toBe(false);
  });

  it('warns when aspect is too narrow (< 1.2)', () => {
    // w/vw = 0.1, h/vh = 0.2 -> aspect 0.5
    const child: [number, number, number, number] = [0.1, 0.1, 0.2, 0.2];
    expect(evaluateShapeGate(child, parent, PLATE_SHAPE_ENVELOPE)).toBe(true);
  });

  it('warns when aspect is too wide (> 8.0)', () => {
    // w/vw = 0.9, h/vh = 0.02 -> aspect 45
    const child: [number, number, number, number] = [0.05, 0.1, 0.95, 0.11];
    expect(evaluateShapeGate(child, parent, PLATE_SHAPE_ENVELOPE)).toBe(true);
  });

  it('warns when width fraction exceeds maxWidthFrac (0.5)', () => {
    // w/vw = 0.6, h/vh = 0.3 -> aspect 2 (fine), but width too wide
    const child: [number, number, number, number] = [0.1, 0.1, 0.7, 0.25];
    expect(evaluateShapeGate(child, parent, PLATE_SHAPE_ENVELOPE)).toBe(true);
  });

  it('warns when area fraction exceeds maxAreaFrac (0.15)', () => {
    // w/vw = 0.45, h/vh = 0.4 -> aspect 1.125 fails too, but exercise area path
    const child: [number, number, number, number] = [0.1, 0.1, 0.55, 0.3];
    expect(evaluateShapeGate(child, parent, PLATE_SHAPE_ENVELOPE)).toBe(true);
  });

  it('does not warn on a degenerate parent (zero area)', () => {
    const degenerateParent: [number, number, number, number] = [0, 0, 1, 0];
    const child: [number, number, number, number] = [0.1, 0.1, 0.2, 0.2];
    expect(evaluateShapeGate(child, degenerateParent, PLATE_SHAPE_ENVELOPE)).toBe(false);
  });

  it('warns on a missing child box', () => {
    expect(evaluateShapeGate(null, parent, PLATE_SHAPE_ENVELOPE)).toBe(false);
    expect(evaluateShapeGate(undefined, parent, PLATE_SHAPE_ENVELOPE)).toBe(false);
  });

  it('warns on non-finite child coordinates (Finding C.1 fix)', () => {
    // Previously: api.ts's _platePlausibleEnvelope guarded with
    // Number.isFinite and returned warning=true; PlateCard's inline copy
    // had no such guard, so NaN compared false against every bound and
    // returned warning=false. Both call sites now share this function, so
    // they agree: a corrupt row warns everywhere.
    const child = [Number.NaN, 0.1, 0.2, 0.2] as [number, number, number, number];
    expect(evaluateShapeGate(child, parent, PLATE_SHAPE_ENVELOPE)).toBe(true);
  });

  it('warns on an infinite child coordinate', () => {
    const child = [Number.POSITIVE_INFINITY, 0.1, 0.2, 0.2] as [
      number,
      number,
      number,
      number,
    ];
    expect(evaluateShapeGate(child, parent, PLATE_SHAPE_ENVELOPE)).toBe(true);
  });
});

describe('describeEnvelope', () => {
  it('renders the plate envelope as the tooltip sentence review/+page.svelte used to hardcode', () => {
    const text = describeEnvelope(PLATE_SHAPE_ENVELOPE);
    expect(text).toContain('aspect ∉ [1.2, 8]');
    expect(text).toContain('50% of parent width');
  });
});
