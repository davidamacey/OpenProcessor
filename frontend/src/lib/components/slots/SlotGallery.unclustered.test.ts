/**
 * M3 (docs/design/interactive-pass-2026-09-24.md): before this fix, the
 * slot gallery on /clusters only ever rendered region-cluster cards —
 * when the only bucket that existed was the permanent false-positive
 * one, every other region was unreachable, and the counter strip leaked
 * a literal `gallery.pager.items` template fragment. Mounts the real component (see CropCard.test.ts's header
 * comment for the convention) rather than scanning source text, since
 * both bugs are directly observable in the rendered DOM.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import SlotGallery from './SlotGallery.svelte';
import { createSlotGalleryController } from '../../../routes/clusters/slotGalleryController.svelte';
import { widgetTagSlot } from '$lib/test/fixtures/regionSlot';
import type { Cluster } from '$lib/types';

vi.mock('$lib/api', async () => {
  const actual = await vi.importActual<typeof import('$lib/api')>('$lib/api');
  return { ...actual, getRegionClusters: vi.fn(), getRegions: vi.fn() };
});
import { getRegionClusters, getRegions } from '$lib/api';

function fpCluster(): Cluster {
  return {
    id: -100,
    cluster_kind: 'false_positive',
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
});

describe('SlotGallery — regions reachable when only the FP bucket is clustered', () => {
  it('offers a "Browse all <plural>" entry point alongside the FP-only cluster grid', async () => {
    vi.mocked(getRegionClusters).mockResolvedValue({ clusters: [fpCluster()] } as never);
    const gallery = createSlotGalleryController(widgetTagSlot);
    await gallery.loadClusters();
    const el = renderGallery(gallery);

    const btn = [...el.querySelectorAll('button')].find(
      (b) => b.textContent?.trim() === 'Browse all tags',
    );
    expect(btn).toBeDefined();
  });

  it('clicking it opens the flat gallery view (viewingAll) with no region_cluster_id filter', async () => {
    vi.mocked(getRegionClusters).mockResolvedValue({ clusters: [fpCluster()] } as never);
    vi.mocked(getRegions).mockResolvedValue({ items: [], total: 0 } as never);
    const gallery = createSlotGalleryController(widgetTagSlot);
    await gallery.loadClusters();
    const el = renderGallery(gallery);

    const btn = [...el.querySelectorAll('button')].find(
      (b) => b.textContent?.trim() === 'Browse all tags',
    )!;
    btn.click();
    flushSync();
    await Promise.resolve();

    expect(gallery.viewingAll).toBe(true);
    expect(gallery.selectedCluster).toBeNull();
    expect(getRegions).toHaveBeenCalled();
    const lastCallParams = vi.mocked(getRegions).mock.calls.at(-1)?.[1] as
      | { region_cluster_id?: number }
      | undefined;
    expect(lastCallParams?.region_cluster_id).toBeUndefined();
    // The bucket grid ("← FALSE POSITIVES" card etc.) must no longer be
    // the only thing rendered — the header now reads "All tags".
    expect(el.textContent).toContain('All tags');
  });

  it('never leaks the stray `gallery.pager.items` template literal into the counter strip', async () => {
    vi.mocked(getRegionClusters).mockResolvedValue({ clusters: [fpCluster()] } as never);
    const gallery = createSlotGalleryController(widgetTagSlot);
    await gallery.loadClusters();
    const el = renderGallery(gallery);

    expect(el.textContent).not.toMatch(/gallery\.pager\.items/);
  });

  it("names the region by the gallery slot's own label, not a fixed noun", async () => {
    vi.mocked(getRegionClusters).mockResolvedValue({ clusters: [fpCluster()] } as never);
    const gallery = createSlotGalleryController({
      ...widgetTagSlot,
      label: { singular: 'sprocket', plural: 'sprockets', title: 'Sprocket' },
    });
    await gallery.loadClusters();
    const el = renderGallery(gallery);

    const text = el.textContent ?? '';
    expect(text).toContain('Cluster sprockets');
    expect(text).toContain('Browse all sprockets');
    expect(text).not.toMatch(/region/i);
  });
});
