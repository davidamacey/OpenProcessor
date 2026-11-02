/** Query by picture on /clusters, and the Crops / Regions target shared by every search. */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api_errors';
import type { Crop } from '$lib/types';

const searchImageCrops = vi.fn();
const searchSimilarCrops = vi.fn();
vi.mock('$lib/api_cropRead', () => ({
  getCrops: vi.fn(),
  searchSimilarCrops: (...a: unknown[]) => searchSimilarCrops(...a),
  searchImageCrops: (...a: unknown[]) => searchImageCrops(...a),
}));
vi.mock('$lib/api_clusters', () => ({
  getClusters: vi.fn().mockResolvedValue({ items: [] }),
}));
vi.mock('$app/navigation', () => ({ goto: vi.fn() }));
vi.mock('$app/paths', () => ({ resolve: (p: string) => p }));
vi.mock('$app/state', () => ({ page: { url: new URL('http://x/p/default/clusters') } }));
vi.mock('$lib/projectPaths', () => ({ projectHref: (p: string) => `/p/default${p}` }));

import { createClusterSearchModes } from './clusterSearchModes.svelte';

let stop: (() => void) | undefined;

function modes() {
  let m!: ReturnType<typeof createClusterSearchModes>;
  stop = $effect.root(() => {
    m = createClusterSearchModes({
      clusterPager: { items: [] } as never,
      showEmbeddingViz: false,
      gridItemFilter: () => ({ class_name: ['widget'] }),
    });
  });
  return m;
}

afterEach(() => {
  stop?.();
  searchImageCrops.mockReset();
  searchSimilarCrops.mockReset();
});

const file = new File([new Uint8Array([1])], 'q.png', { type: 'image/png' });
const hit = {
  id: 'n1',
  cluster_id: null,
  similarity_score: 0.7,
  matched_box_id: 'b2',
  matched_box_stale: true,
} as unknown as Crop;

describe('runImage', () => {
  it('searches by the picture and box, scoped by the grid filter', async () => {
    searchImageCrops.mockResolvedValue({
      items: [hit],
      total: 1,
      unembedded_in_scope: 3,
    });
    const m = modes();
    await m.runImage({ file, box: [0, 0, 0.5, 0.5] });
    expect(searchImageCrops).toHaveBeenCalledWith(
      { file, box: [0, 0, 0.5, 0.5] },
      1,
      200,
      { class_name: ['widget'] },
    );
    expect(m.searchModeActive).toBe(true);
    expect(m.imageQueryName).toBe('q.png');
    expect(m.searchResults).toEqual([hit]);
    expect(m.searchScores.get('n1')).toBe(0.7);
    expect(m.searchUnembedded).toBe(3);
  });

  it('keeps the served refusal as inline text', async () => {
    searchImageCrops.mockRejectedValue(
      new ApiError(413, '/x', {
        detail: { error: 'upload_too_large', message: 'too big' },
      }),
    );
    const m = modes();
    await m.runImage({ file });
    expect(m.similarError).toContain('too big');
    expect(m.searchResults).toEqual([]);
  });

  it('exitSearchMode forgets the picture', async () => {
    searchImageCrops.mockResolvedValue({ items: [], total: 0 });
    const m = modes();
    await m.runImage({ file });
    m.exitSearchMode();
    expect(m.imageQueryName).toBeNull();
  });
});

describe('search target', () => {
  it('sends target=box and exposes the matched region per crop', async () => {
    searchImageCrops.mockResolvedValue({ items: [hit], total: 1 });
    const m = modes();
    m.searchTarget = 'box';
    await m.runImage({ file });
    expect(searchImageCrops.mock.calls[0][3]).toEqual({
      class_name: ['widget'],
      target: 'box',
    });
    expect(m.matchOf('n1')).toEqual({ boxId: 'b2', stale: true });
    expect(m.matchOf('other')).toBeNull();
  });

  it('switching the target re-runs the picture search from page one', async () => {
    searchImageCrops.mockResolvedValue({ items: [], total: 0 });
    const m = modes();
    await m.runImage({ file });
    await m.setSearchTarget('box');
    expect(searchImageCrops).toHaveBeenCalledTimes(2);
    expect(searchImageCrops.mock.calls[1][3]).toEqual({
      class_name: ['widget'],
      target: 'box',
    });
  });

  it('switching the target re-runs More like this', async () => {
    searchSimilarCrops.mockResolvedValue({ items: [], total: 0 });
    const m = modes();
    await m.runSimilar('crop-0');
    await m.setSearchTarget('box');
    expect(searchSimilarCrops.mock.calls[1]).toEqual([
      'crop-0',
      1,
      200,
      { class_name: ['widget'], target: 'box' },
    ]);
  });
});
