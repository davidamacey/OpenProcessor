/**
 * The region gallery on `/clusters` lists the serving region partition's
 * cluster cards (`GET {API_PREFIX}/regions/clusters`) by `cluster_uid`.
 * False positives are a box state, not a cluster, and sub-clusters are gone.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import SlotGallery from './SlotGallery.svelte';
import { createSlotGalleryController } from '$lib/slots/slotGalleryController.svelte';
import { widgetTagSlot } from '$lib/test/fixtures/regionSlot';
import type { Cluster } from '$lib/types';

vi.mock('$lib/api_regions', async () => {
  const actual =
    await vi.importActual<typeof import('$lib/api_regions')>('$lib/api_regions');
  return { ...actual, getRegionClusters: vi.fn(), getRegions: vi.fn() };
});
import { getRegionClusters, getRegions } from '$lib/api_regions';

function baseCluster(overrides: Partial<Cluster>): Cluster {
  return {
    uid: '0b9f9a38-0000-4000-8000-000000000001',
    size: 12,
    validated_count: 0,
    dominant_class_name: null,
    dominant_pct: null,
    is_unlabeled: true,
    representative_crop_ids: [],
    updated_at: null,
    ...overrides,
  };
}

let target: HTMLDivElement;
let instance: unknown;

function renderGallery(gallery: ReturnType<typeof createSlotGalleryController>) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(SlotGallery, { target, props: { gallery } } as never);
  flushSync();
  return target;
}

afterEach(() => {
  if (instance) {
    unmount(instance);
    instance = undefined;
  }
  target?.remove();
  vi.mocked(getRegions).mockReset();
});

describe('SlotGallery — region cluster cards are keyed by cluster_uid', () => {
  const UID = '0b9f9a38-0000-4000-8000-0000000000aa';

  it('shows a card per served cluster and no false-positive bucket or sub-cluster chrome', async () => {
    vi.mocked(getRegionClusters).mockResolvedValue({
      clusters: [baseCluster({ uid: UID, display_no: 12 })],
    } as never);
    vi.mocked(getRegions).mockResolvedValue({ items: [], total: 0 } as never);
    const gallery = createSlotGalleryController(widgetTagSlot);
    await gallery.loadClusters();
    const el = renderGallery(gallery);

    expect(el.textContent).toContain('#12');
    expect(el.textContent).not.toContain(UID.slice(0, 8));
    expect(el.textContent).not.toContain('False positives');
    expect(el.textContent).not.toContain('Refine');
    expect(el.textContent).not.toContain('sub-cluster');
  });

  it('opening a card browses the regions of that cluster_uid and names it', async () => {
    vi.mocked(getRegionClusters).mockResolvedValue({
      clusters: [baseCluster({ uid: UID })],
    } as never);
    vi.mocked(getRegions).mockResolvedValue({ items: [], total: 0 } as never);
    const gallery = createSlotGalleryController(widgetTagSlot);
    await gallery.loadClusters();
    const el = renderGallery(gallery);

    gallery.openCluster(UID);
    await gallery.loadFirst();
    flushSync();

    expect(gallery.selectedCluster).toBe(UID);
    const query = vi.mocked(getRegions).mock.calls.at(-1)![1] as Record<string, unknown>;
    expect(query.cluster_uid).toBe(UID);
    expect(query).not.toHaveProperty('region_cluster_id');
    expect(query).not.toHaveProperty('sort_by_subid');
    expect(el.textContent).toContain(`cluster ${UID.slice(0, 8)}`);
  });
});
