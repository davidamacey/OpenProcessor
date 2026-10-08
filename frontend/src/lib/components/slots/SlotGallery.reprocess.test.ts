/**
 * W10 image Reprocess from the region gallery: a card's "Reprocess
 * image..." hands the served items to the gallery, which re-fetches its
 * page (re-detection can add, move or drop the image's regions) instead
 * of leaving the old cards on screen. Mounts the real component.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import SlotGallery from './SlotGallery.svelte';
import { createSlotGalleryController } from '../../../routes/p/[project]/clusters/slotGalleryController.svelte';
import { widgetTagSlot } from '$lib/test/fixtures/regionSlot';
import { API_PREFIX } from '$lib/api';
import { datasetsAvailability } from '$lib/datasets/datasetsAvailability.svelte';
import { formatsFixture, reprocessFixture } from '$lib/test/fixtures/datasetImport';

vi.mock('$lib/api', async () => {
  const actual = await vi.importActual<typeof import('$lib/api')>('$lib/api');
  return { ...actual, getRegionClusters: vi.fn(), getRegions: vi.fn() };
});
import { getRegionClusters, getRegions } from '$lib/api';

let target: HTMLDivElement;
let instance: unknown;

const row = (id: string) => ({
  crop_id: id,
  id,
  image_id: 'img_1',
  image_path: '/img.jpg',
  bbox_norm: [0.3, 0.3, 0.7, 0.7],
  region_status: 'detected',
  region_verified: false,
});

function json(b: unknown, s = 200): Response {
  return new Response(JSON.stringify(b), {
    status: s,
    headers: { 'content-type': 'application/json' },
  });
}

beforeEach(() => {
  vi.stubGlobal(
    'IntersectionObserver',
    class {
      observe() {}
      unobserve() {}
      disconnect() {}
    },
  );
  datasetsAvailability.reset();
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      const u = String(url);
      if (u === `${API_PREFIX}/datasets/formats`) return json(formatsFixture());
      if (u.endsWith('/images/img_1/reprocess')) {
        return json(reprocessFixture({ dry_run: false, items: [{ crop_id: 'c1' }] }));
      }
      return json({ detail: 'Not Found' }, 404);
    }),
  );
});

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
  document.querySelectorAll('[role="dialog"]').forEach((d) => d.remove());
  vi.unstubAllGlobals();
  vi.mocked(getRegions).mockReset();
  vi.mocked(getRegionClusters).mockReset();
  datasetsAvailability.reset();
});

describe('SlotGallery image Reprocess', () => {
  it('re-fetches the gallery page after a card reprocesses its image', async () => {
    vi.mocked(getRegionClusters).mockResolvedValue({ clusters: [] } as never);
    vi.mocked(getRegions).mockResolvedValue({ items: [row('c1')], total: 1 } as never);
    const gallery = createSlotGalleryController(widgetTagSlot);
    await gallery.loadClusters();
    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(SlotGallery, { target, props: { gallery } } as never);
    gallery.openAll();
    await vi.waitFor(() => expect(gallery.pager.items).toHaveLength(1));
    await datasetsAvailability.init();
    flushSync();
    const before = vi.mocked(getRegions).mock.calls.length;

    const open = target.querySelector(
      '[data-testid="card-reprocess-image"] [data-testid="reprocess-open"]',
    ) as HTMLButtonElement;
    expect(open).not.toBeNull();
    open.click();
    flushSync();
    const box = [...document.querySelectorAll('fieldset label')]
      .find((l) => l.textContent?.includes('detect'))
      ?.querySelector('input') as HTMLInputElement;
    box.checked = true;
    box.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    (
      [...document.querySelectorAll('button')].find(
        (b) => b.textContent?.trim() === 'Reprocess',
      ) as HTMLButtonElement
    ).click();

    await vi.waitFor(() =>
      expect(vi.mocked(getRegions).mock.calls.length).toBeGreaterThan(before),
    );
  });
});
