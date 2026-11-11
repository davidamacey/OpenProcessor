/**
 * A MapLibre `error` that is not a single failed tile (a worker that cannot load, a bad style)
 * leaves the map blank, so the card says why in place instead of drawing nothing.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';

vi.mock('maplibre-gl/dist/maplibre-gl-worker.mjs?url', () => ({
  default: '/worker.mjs',
}));
vi.mock('maplibre-gl/dist/maplibre-gl.css', () => ({}));

const h = vi.hoisted(() => ({
  handlers: {} as Record<string, (e: unknown) => void>,
}));

vi.mock('maplibre-gl', () => ({
  GPUInitializationError: class extends Error {},
  Map: class {
    on(name: string, fn: (e: unknown) => void) {
      h.handlers[name] = fn;
    }
    addControl() {}
    getSource() {
      return undefined;
    }
    remove() {}
  },
  NavigationControl: class {},
  Popup: class {},
  setWorkerUrl: () => {},
}));

import MapView from './MapView.svelte';
import type { MapMeta } from '$lib/types_map';

const META: MapMeta = {
  available: true,
  location_policy: 'store',
  n_located: 3,
  bounds: null,
  tile_url: '/t/{z}/{x}/{y}.png',
  tile_attribution: '',
  max_zoom: 12,
};

let instance: ReturnType<typeof mount> | null = null;
let target: HTMLDivElement;

function render(): void {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(MapView, {
    target,
    props: { meta: META, cells: null, onview: () => {}, oncell: () => false },
  });
  flushSync();
}

const note = () => target.querySelector('[data-testid="map-load-error"]');

afterEach(() => {
  if (instance) unmount(instance);
  instance = null;
  document.body.innerHTML = '';
});

describe('MapView load failure', () => {
  it('names a worker failure in the card', () => {
    render();
    h.handlers.error!({ error: new Error('Worker failed to load.') });
    flushSync();
    expect(note()?.textContent).toContain('Map could not load: Worker failed to load.');
  });

  it('ignores a single failed tile', () => {
    render();
    h.handlers.error!({ error: new Error('404'), tile: {}, sourceId: 'basemap' });
    flushSync();
    expect(note()).toBeNull();
  });
});
