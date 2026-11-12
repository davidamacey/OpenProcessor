/**
 * Without WebGL2 the MapLibre constructor throws `GPUInitializationError`
 * (https://maplibre.org/maplibre-gl-js/docs/API/classes/Map/). The map card says so in
 * place, with the located-photo count, instead of letting the page fall to the error boundary.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';

vi.mock('maplibre-gl/dist/maplibre-gl-worker.mjs?url', () => ({
  default: '/worker.mjs',
}));
vi.mock('maplibre-gl/dist/maplibre-gl.css', () => ({}));

const { gpuError } = vi.hoisted(() => {
  class GPUInitializationError extends Error {
    constructor() {
      super('WebGL2 is required');
      this.name = 'GPUInitializationError';
    }
  }
  return { gpuError: GPUInitializationError };
});

vi.mock('maplibre-gl', () => ({
  GPUInitializationError: gpuError,
  Map: class {
    constructor() {
      throw new gpuError();
    }
  },
  NavigationControl: class {},
  Popup: class {},
  setWorkerUrl: () => {},
}));

import MapView from './MapView.svelte';
import { mapMeta } from '$lib/test/fixtures/mapMeta';
import type { MapMeta } from '$lib/types_map';

const META: MapMeta = mapMeta({ location_policy: 'store_coarse' });

let instance: ReturnType<typeof mount> | null = null;

afterEach(() => {
  if (instance) unmount(instance);
  instance = null;
  document.body.innerHTML = '';
});

describe('MapView without WebGL2', () => {
  it('shows an in-page message naming the located photo count', () => {
    const target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(MapView, {
      target,
      props: { meta: META, cells: null, onview: () => {}, oncell: () => false },
    });
    flushSync();
    const note = target.querySelector('[data-testid="map-webgl-unsupported"]');
    expect(note?.textContent).toContain('WebGL2');
    expect(note?.textContent).toContain('3 photos have GPS');
  });
});
