import { describe, expect, it } from 'vitest';
import {
  actualSize,
  clampViewport,
  FIT,
  MAX_SCALE,
  panBy,
  viewportRect,
  zoomAt,
} from './viewport';

const FRAME = { width: 400, height: 300 };

describe('viewport', () => {
  it('zooms in keeping the point under the cursor fixed', () => {
    const at = { x: 100, y: 60 };
    const vp = zoomAt(FIT, 2, at, FRAME);
    // The picture point under the cursor before: (100, 60). After: (100 - x) / scale.
    expect((at.x - vp.x) / vp.scale).toBeCloseTo(100);
    expect((at.y - vp.y) / vp.scale).toBeCloseTo(60);
    expect(vp.scale).toBe(2);
  });

  it('zooming out never goes below the fitted size', () => {
    expect(zoomAt(FIT, 0.2, { x: 10, y: 10 }, FRAME)).toEqual(FIT);
  });

  it('caps the zoom', () => {
    expect(zoomAt(FIT, 1000, { x: 0, y: 0 }, FRAME).scale).toBe(MAX_SCALE);
  });

  it('a fitted picture does not pan', () => {
    expect(panBy(FIT, 50, 50, FRAME)).toEqual(FIT);
  });

  it('a zoomed picture pans up to its edges and no further', () => {
    const zoomed = { scale: 2, x: 0, y: 0 };
    expect(panBy(zoomed, -1000, -1000, FRAME)).toEqual({ scale: 2, x: -400, y: -300 });
    expect(panBy(zoomed, 1000, 1000, FRAME)).toEqual(zoomed);
  });

  it('clampViewport pulls an offset back inside', () => {
    expect(clampViewport({ scale: 2, x: 30, y: -999 }, FRAME)).toEqual({
      scale: 2,
      x: 0,
      y: -300,
    });
  });

  it('actualSize shows one picture pixel per screen pixel', () => {
    const vp = actualSize(FIT, 1600, { x: 200, y: 150 }, FRAME);
    expect(vp.scale).toBeCloseTo(4);
  });

  it('places the frame centred when fitted and grown from its corner when zoomed', () => {
    const container = { width: 500, height: 300 };
    expect(viewportRect(FIT, FRAME, container)).toEqual({
      left: 50,
      top: 0,
      width: 400,
      height: 300,
    });
    expect(viewportRect({ scale: 2, x: -100, y: -40 }, FRAME, container)).toEqual({
      left: -50,
      top: -40,
      width: 800,
      height: 600,
    });
  });
});
