import { describe, expect, it, vi } from 'vitest';

vi.mock('$lib/api', async (orig) => ({
  ...(await orig<object>()),
  apiFetch: vi.fn().mockResolvedValue({
    count: 1,
    clusters: [
      {
        cluster_uid: 'u1',
        size: 3,
        box_count: 4,
        validated_count: 0,
        display_name: 'bus',
        display_name_kind: 'detector',
        representative_crop_ids: [],
        representative_thumb_urls: [],
      },
    ],
  }),
}));
import { getRegionClusters } from './api_regions';

describe('getRegionClusters', () => {
  it('carries the served display name and its kind onto the card', async () => {
    const { clusters } = await getRegionClusters();
    expect(clusters[0]).toMatchObject({
      display_name: 'bus',
      display_name_kind: 'detector',
    });
  });
});
