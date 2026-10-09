/**
 * #90: the FP-bucket styling on `/clusters`' region gallery must key off
 * the served `cluster_kind === 'false_positive'` on the SELECTED cluster
 * card (`GET {API_PREFIX}/regions/clusters`), never a client-side id
 * constant (the deleted `FALSE_POSITIVE_REGION_CLUSTER_ID = -100`). These
 * two cases are exactly what an id-based check gets backwards:
 *
 * - a false-positive cluster whose served id is NOT -100 must still get
 *   the FP treatment.
 * - a normal (non-FP) cluster whose served id happens to BE -100 must not.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import SlotGallery from './SlotGallery.svelte';
import { createSlotGalleryController } from '../../../routes/p/[project]/clusters/slotGalleryController.svelte';
import { widgetTagSlot } from '$lib/test/fixtures/regionSlot';
import type { Cluster } from '$lib/types';

vi.mock('$lib/api', async () => {
  const actual = await vi.importActual<typeof import('$lib/api')>('$lib/api');
  return { ...actual, getRegionClusters: vi.fn(), getRegions: vi.fn() };
});
import { getRegionClusters, getRegions } from '$lib/api';

function baseCluster(overrides: Partial<Cluster>): Cluster {
  return {
    id: 1,
    cluster_kind: 'candidate',
    size: 12,
    validated_count: 0,
    dominant_class_id: null,
    dominant_class_name: null,
    dominant_pct: null,
    purity: null,
    purity_tier: null,
    promotable: false,
    core_similarity_min: null,
    is_unlabeled: false,
    representative_crop_ids: [],
    has_subclusters: false,
    n_subclusters: 0,
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

describe('SlotGallery — FP bucket keyed by served cluster_kind, not id', () => {
  it('a false-positive cluster whose id is NOT -100 still gets the FP treatment', async () => {
    const fp = baseCluster({ id: 42, cluster_kind: 'false_positive' });
    vi.mocked(getRegionClusters).mockResolvedValue({ clusters: [fp] } as never);
    vi.mocked(getRegions).mockResolvedValue({ items: [], total: 0 } as never);
    const gallery = createSlotGalleryController(widgetTagSlot);
    await gallery.loadClusters();
    const el = renderGallery(gallery);

    // Grid card itself is styled/labelled as FP purely off cluster_kind.
    expect(el.textContent).toContain('False positives');

    gallery.openCluster(42);
    await Promise.resolve();
    flushSync();

    expect(gallery.selectedClusterIsFalsePositive).toBe(true);
    expect(el.textContent).toContain('False-positive cluster');
    expect(el.textContent).not.toContain('bucket #42');
  });

  it('a normal cluster whose id happens to be -100 does NOT get FP treatment', async () => {
    const notFp = baseCluster({ id: -100, cluster_kind: 'candidate' });
    vi.mocked(getRegionClusters).mockResolvedValue({ clusters: [notFp] } as never);
    vi.mocked(getRegions).mockResolvedValue({ items: [], total: 0 } as never);
    const gallery = createSlotGalleryController(widgetTagSlot);
    await gallery.loadClusters();
    const el = renderGallery(gallery);

    expect(el.textContent).not.toContain('False positives');

    gallery.openCluster(-100);
    await Promise.resolve();
    flushSync();

    expect(gallery.selectedClusterIsFalsePositive).toBe(false);
    expect(el.textContent).not.toContain('False-positive cluster');
    expect(el.textContent).toContain('bucket #-100');
  });
});
