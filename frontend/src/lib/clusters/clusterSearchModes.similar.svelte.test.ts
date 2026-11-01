/** "More like this" on /clusters: results, the served refusal and leaving the mode. */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api_errors';
import type { Crop } from '$lib/types';

const searchSimilarCrops = vi.fn();
const goto = vi.fn();
vi.mock('$lib/api_cropRead', () => ({
  getCrops: vi.fn(),
  searchSimilarCrops: (...a: unknown[]) => searchSimilarCrops(...a),
}));
vi.mock('$lib/api_clusters', () => ({
  getClusters: vi.fn().mockResolvedValue({ items: [] }),
}));
vi.mock('$app/navigation', () => ({ goto: (...a: unknown[]) => goto(...a) }));
vi.mock('$app/paths', () => ({ resolve: (p: string) => p }));
vi.mock('$app/state', () => ({
  page: { url: new URL('http://x/p/default/clusters?similar_to=crop-0') },
}));
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
  searchSimilarCrops.mockReset();
  goto.mockReset();
});

describe('runSimilar', () => {
  it('shows the neighbours with their scores, scoped by the grid filter', async () => {
    const hit = { id: 'n1', cluster_id: null, similarity_score: 0.8 } as unknown as Crop;
    searchSimilarCrops.mockResolvedValue({
      items: [hit],
      total: 1,
      unembedded_in_scope: 2,
    });
    const m = modes();
    await m.runSimilar('crop-0');
    expect(searchSimilarCrops).toHaveBeenCalledWith('crop-0', 1, 200, {
      class_name: ['widget'],
    });
    expect(m.searchModeActive).toBe(true);
    expect(m.similarTo).toBe('crop-0');
    expect(m.searchResults).toEqual([hit]);
    expect(m.searchScores.get('n1')).toBe(0.8);
    expect(m.searchUnembedded).toBe(2);
  });

  it('keeps the served refusal as inline text', async () => {
    searchSimilarCrops.mockRejectedValue(
      new ApiError(409, '/x', {
        detail: {
          error: 'crop_not_embedded',
          message: 'crop crop-0 has no vector (deferred)',
        },
      }),
    );
    const m = modes();
    await m.runSimilar('crop-0');
    expect(m.similarError).toContain('has no vector (deferred)');
    expect(m.searchResults).toEqual([]);
  });

  it('exitSearchMode clears the source crop and the error', async () => {
    searchSimilarCrops.mockRejectedValue(
      new ApiError(404, '/x', { detail: { error: 'not_found', message: 'no crop' } }),
    );
    const m = modes();
    await m.runSimilar('crop-0');
    m.exitSearchMode();
    expect(m.similarTo).toBeNull();
    expect(m.similarError).toBeNull();
    expect(m.searchModeActive).toBe(false);
  });
});
