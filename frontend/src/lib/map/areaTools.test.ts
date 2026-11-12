import { describe, expect, it, vi, type Mock } from 'vitest';
import { AreaTools, type AreaHost } from './areaTools.svelte';

function host(): AreaHost & { applyFilter: Mock<() => void> } {
  return { filter: { geoBbox: null, geoPolygon: null }, applyFilter: vi.fn() };
}

describe('AreaTools', () => {
  it('a rectangle sets geo_bbox from any two opposite corners', () => {
    const h = host();
    const tools = new AreaTools(h);
    tools.begin('rectangle', [9, 48]);
    tools.extend([7, 46]);
    tools.finish();
    expect(h.filter.geoBbox).toEqual({ west: 7, south: 46, east: 9, north: 48 });
    expect(h.filter.geoPolygon).toBeNull();
    expect(h.applyFilter).toHaveBeenCalledTimes(1);
    expect(tools.drawing).toBe(false);
  });

  it('a lasso sets geo_polygon and replaces a rectangle', () => {
    const h = host();
    h.filter.geoBbox = { west: 0, south: 0, east: 1, north: 1 };
    const tools = new AreaTools(h);
    tools.begin('lasso', [7, 46]);
    tools.extend([9, 46]);
    tools.extend([8, 48]);
    tools.finish();
    expect(h.filter.geoPolygon).toEqual([
      [7, 46],
      [9, 46],
      [8, 48],
    ]);
    expect(h.filter.geoBbox).toBeNull();
    expect(h.applyFilter).toHaveBeenCalledTimes(1);
  });

  it('a click or a two-point lasso changes nothing', () => {
    const h = host();
    const tools = new AreaTools(h);
    tools.begin('rectangle', [7, 46]);
    tools.finish();
    tools.begin('lasso', [7, 46]);
    tools.extend([8, 47]);
    tools.finish();
    expect(h.filter.geoBbox).toBeNull();
    expect(h.filter.geoPolygon).toBeNull();
    expect(h.applyFilter).not.toHaveBeenCalled();
  });

  it('panning never changes the filter: extend without a gesture is ignored', () => {
    const h = host();
    const tools = new AreaTools(h);
    tools.extend([8, 47]);
    tools.finish();
    expect(tools.points).toEqual([]);
    expect(h.filter.geoBbox).toBeNull();
    expect(h.applyFilter).not.toHaveBeenCalled();
  });

  it('cancel drops the gesture without touching the filter', () => {
    const h = host();
    const tools = new AreaTools(h);
    tools.begin('rectangle', [7, 46]);
    tools.extend([9, 48]);
    tools.cancel();
    tools.finish();
    expect(h.filter.geoBbox).toBeNull();
    expect(h.applyFilter).not.toHaveBeenCalled();
  });
});
