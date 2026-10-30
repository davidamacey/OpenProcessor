/**
 * The cluster page's facet chips: the counts are the served facets, and a
 * selected chip goes out as the matching `GET /crops` parameter, unchanged.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync } from 'svelte';

vi.mock('$app/state', () => ({
  page: { params: { id: '7' }, url: new URL('http://x/p/default/clusters/7') },
}));
vi.mock('$lib/api_clusters', async () => {
  const actual =
    await vi.importActual<typeof import('$lib/api_clusters')>('$lib/api_clusters');
  return { ...actual, getCluster: vi.fn(), getClusterFacets: vi.fn() };
});
import { getCluster, getClusterFacets } from '$lib/api_clusters';
import { createClusterDetailController } from './clusterDetailController.svelte';
import type { ClusterFacets } from '$lib/types_clusterList';

const FACETS: ClusterFacets = {
  cluster_id: 7,
  total: 30,
  detector_classes: [
    { value: 'widget_x', count: 20 },
    { value: 'widget_y', count: 0 },
  ],
  class_sources: [],
  validation: { validated: 4, unvalidated: 26 },
  review_tabs: { outliers: 3, detector_disagreements: 0 },
  detector_confidence: [
    { min: 0.0, max: 0.1, count: 0 },
    { min: 0.9, max: null, count: 12 },
  ],
  detector_confidence_missing: 2,
};

let cleanup: (() => void) | undefined;

function wire() {
  let view!: ReturnType<typeof createClusterDetailController>;
  cleanup = $effect.root(() => {
    view = createClusterDetailController();
  });
  flushSync();
  return view;
}
async function settle() {
  for (let i = 0; i < 5; i++) {
    await Promise.resolve();
    flushSync();
  }
}
const lastOpts = () => vi.mocked(getCluster).mock.calls.at(-1)![4]!;

beforeEach(() => {
  vi.mocked(getCluster).mockResolvedValue({
    cluster: { id: 7 } as never,
    cardError: null,
    crops: { items: [], total: 0, page: 1, page_size: 60 },
  });
  vi.mocked(getClusterFacets).mockResolvedValue(FACETS);
});
afterEach(() => {
  cleanup?.();
  cleanup = undefined;
  vi.mocked(getCluster).mockReset();
  vi.mocked(getClusterFacets).mockReset();
});

describe('cluster facets', () => {
  it('reads the served facets for the viewed cluster', async () => {
    const view = wire();
    await settle();
    expect(getClusterFacets).toHaveBeenCalledWith(7);
    expect(view.facets).toEqual(FACETS);
  });

  it('shows a failed facets read as its error, with no fabricated counts', async () => {
    vi.mocked(getClusterFacets).mockRejectedValue(new Error('boom'));
    const view = wire();
    await settle();
    expect(view.facets).toBeNull();
    expect(view.facetsError).toBeTruthy();
  });

  it('sends a detector class chip as detector_class_name, and a second click clears it', async () => {
    const view = wire();
    await settle();
    expect(lastOpts().detectorClassName).toEqual([]);
    view.toggleDetectorClass('widget_x');
    flushSync();
    await settle();
    expect(lastOpts().detectorClassName).toEqual(['widget_x']);
    view.toggleDetectorClass('widget_x');
    flushSync();
    await settle();
    expect(lastOpts().detectorClassName).toEqual([]);
  });

  it('sends a confidence bin with the served min and max (max null sends none)', async () => {
    const view = wire();
    await settle();
    view.toggleBand(FACETS.detector_confidence[0]!);
    flushSync();
    await settle();
    expect(lastOpts()).toMatchObject({ detectorConfMin: 0, detectorConfMax: 0.1 });
    view.toggleBand(FACETS.detector_confidence[1]!);
    flushSync();
    await settle();
    expect(lastOpts()).toMatchObject({ detectorConfMin: 0.9, detectorConfMax: null });
  });

  it('sends the review tab and the validated flag', async () => {
    const view = wire();
    await settle();
    view.toggleReviewTab('outliers');
    view.toggleValidated(false);
    flushSync();
    await settle();
    expect(lastOpts()).toMatchObject({ reviewTab: 'outliers', labelValidated: false });
  });

  it('sorts by detector confidence in place of the core-first order', async () => {
    const view = wire();
    await settle();
    expect(lastOpts().order).toBe('core_first');
    view.detectorSort = 'desc';
    flushSync();
    await settle();
    expect(lastOpts()).toMatchObject({
      sort: 'detector_confidence:desc',
      order: 'default',
    });
    view.resetFacetFilters();
    flushSync();
    await settle();
    expect(lastOpts().sort).toBeNull();
    expect(lastOpts().order).toBe('core_first');
    expect(view.facetFilterActive).toBe(false);
  });
});
