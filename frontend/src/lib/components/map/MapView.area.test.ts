/**
 * The map wires the area tools to MapLibre: Shift+drag must not zoom (`boxZoom` off), a
 * Shift+drag draws a rectangle that reaches the filter once, and a plain drag never does.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';

vi.mock('maplibre-gl/dist/maplibre-gl-worker.mjs?url', () => ({ default: '/w.mjs' }));
vi.mock('maplibre-gl/dist/maplibre-gl.css', () => ({}));

const h = vi.hoisted(() => {
  type Handler = (e: unknown) => void;
  const handlers: Record<string, Handler> = {};
  return {
    handlers,
    boxZoomDisable: vi.fn(),
    dragPanDisable: vi.fn(),
    dragPanEnable: vi.fn(),
  };
});

vi.mock('maplibre-gl', () => ({
  GPUInitializationError: class extends Error {},
  Map: class {
    boxZoom = { disable: h.boxZoomDisable };
    dragPan = { disable: h.dragPanDisable, enable: h.dragPanEnable };
    on(name: string, fn: (e: unknown) => void) {
      h.handlers[name] = fn;
    }
    addControl() {}
    remove() {}
    getSource() {
      return undefined;
    }
  },
  NavigationControl: class {},
  Popup: class {},
  setWorkerUrl: () => {},
}));

import MapView from './MapView.svelte';
import { AreaTools } from '$lib/map/areaTools.svelte';
import { mapMeta } from '$lib/test/fixtures/mapMeta';

let instance: ReturnType<typeof mount> | null = null;
afterEach(() => {
  if (instance) unmount(instance);
  instance = null;
  document.body.innerHTML = '';
});

function setup() {
  const filter = { geoBbox: null, geoPolygon: null };
  const applyFilter = vi.fn();
  const area = new AreaTools({ filter, applyFilter });
  const target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(MapView, {
    target,
    props: { meta: mapMeta(), cells: null, onview: () => {}, oncell: () => false, area },
  });
  flushSync();
  return { filter, applyFilter };
}

const at = (lng: number, lat: number, shiftKey: boolean) => ({
  lngLat: { lng, lat },
  originalEvent: { shiftKey },
  preventDefault: () => {},
});

describe('MapView area tools', () => {
  it('disables box zoom so Shift+drag is free for the rectangle', () => {
    setup();
    expect(h.boxZoomDisable).toHaveBeenCalledTimes(1);
  });

  it('a Shift+drag writes the exact box and applies the filter once', () => {
    const { filter, applyFilter } = setup();
    h.handlers.mousedown(at(9.123456, 48, true));
    h.handlers.mousemove(at(8, 47, true));
    h.handlers.mousemove(at(7.5, 46.25, true));
    h.handlers.mouseup({});
    expect(filter.geoBbox).toEqual({
      west: 7.5,
      south: 46.25,
      east: 9.123456,
      north: 48,
    });
    expect(applyFilter).toHaveBeenCalledTimes(1);
    expect(h.dragPanDisable).toHaveBeenCalledTimes(1);
    expect(h.dragPanEnable).toHaveBeenCalledTimes(1);
  });

  it('a plain drag (a pan) never changes the filter', () => {
    const { filter, applyFilter } = setup();
    h.handlers.mousedown(at(9, 48, false));
    h.handlers.mousemove(at(8, 47, false));
    h.handlers.mouseup({});
    expect(filter.geoBbox).toBeNull();
    expect(filter.geoPolygon).toBeNull();
    expect(applyFilter).not.toHaveBeenCalled();
  });
});
