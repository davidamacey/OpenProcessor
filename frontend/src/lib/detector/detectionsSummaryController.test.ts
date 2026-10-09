import { describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api';
import type { DetectionsSummary } from '$lib/types_detector';
import { DetectionsSummaryState } from './detectionsSummaryController.svelte';

const SUMMARY: DetectionsSummary = {
  total: 120,
  embedding: {
    embedded: 100,
    not_embedded: 20,
    by_state: { embedded: 100, not_selected: 15, failed: 5 },
  },
  by_label: [
    {
      name: 'widget',
      count: 80,
      embedding: { embedded: 70, not_embedded: 10, by_state: {} },
    },
  ],
  labels_truncated: true,
  suggested_reprocess: {
    targets: { filter: { embedding_state: ['not_selected', 'failed'] } },
    scopes: ['embed'],
    dry_run: true,
  },
};

describe('DetectionsSummaryState', () => {
  it('loads the served summary and keeps suggested_reprocess as served', async () => {
    const get = vi.fn(async () => structuredClone(SUMMARY));
    const s = new DetectionsSummaryState(get);
    await s.load();
    expect(s.summary?.total).toBe(120);
    expect(s.summary?.suggested_reprocess).toEqual(SUMMARY.suggested_reprocess);
    expect(s.error).toBeNull();
  });

  it('shows the served error and keeps no summary on a failed read', async () => {
    const s = new DetectionsSummaryState(
      vi.fn(async () => {
        throw new ApiError(500, 'u', { detail: 'index down' });
      }),
    );
    await s.load();
    expect(s.summary).toBeNull();
    expect(s.error).toBe('index down');
  });

  it('refresh re-reads', async () => {
    const get = vi.fn(async () => structuredClone(SUMMARY));
    const s = new DetectionsSummaryState(get);
    await s.load();
    await s.load();
    expect(get).toHaveBeenCalledTimes(2);
  });
});
