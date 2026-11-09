/**
 * Shared fixtures for the clusterController.*.test.ts files: a minimal
 * crop and a controller wired to a real pager, selection and exclusion
 * guard with spy callbacks. Each test file mocks `$lib/api` itself.
 */
import { vi, type Mock } from 'vitest';
import { getJob } from '$lib/api_jobs';
import {
  createClusterActionController,
  createExclusionGuard,
} from './clusterController.svelte';
import { createPager } from '$lib/pager.svelte';
import { createSelection } from '$lib/selection.svelte';
import type { Crop } from '$lib/types';
import type { JobView } from '$lib/types_jobs';

export function crop(id: string, extra: Partial<Crop> = {}): Crop {
  return {
    id,
    source_image_path: '/img.jpg',
    bbox_norm: { cx: 0.5, cy: 0.5, w: 0.4, h: 0.4 },
    class_id: null,
    class_name: null,
    class_source: null,
    label_source: 'model',
    label_validated: false,
    class_validated: false,
    label_confidence: null,
    cluster: {
      uid: 'cluster-uid-42',
      partition_id: 'partition-1',
      state: 'assigned',
      score: null,
      margin: null,
      assigned_by: 'run',
    },
    ...extra,
  } as Crop;
}

export function setup(items: Crop[]) {
  const cropPager = createPager<Crop>({
    fetchPage: async () => ({ items: [], total: 0 }),
    keyOf: (c) => c.id,
  });
  cropPager.items = items;
  cropPager.total = items.length;
  const sel = createSelection({ plainClick: 'replace' });
  const exclusionGuard = createExclusionGuard();
  let dragIds: string[] = [];
  const resetGrid = vi.fn();
  const loadFirst = vi.fn(async () => {});
  const rememberTarget = vi.fn();
  let visible = items;
  const controller = createClusterActionController({
    cropPager,
    sel,
    exclusionGuard,
    resetGrid,
    getDragIds: () => dragIds,
    setDragIds: (ids) => {
      dragIds = ids;
    },
    getVisibleCrops: () => visible,
    getClusterUid: () => 'cluster-uid-42',
    rememberTarget,
    loadFirst,
  });
  return {
    cropPager,
    sel,
    exclusionGuard,
    controller,
    resetGrid,
    loadFirst,
    rememberTarget,
    getDragIds: () => dragIds,
    setDragIds: (ids: string[]) => {
      dragIds = ids;
    },
    setVisible: (v: Crop[]) => {
      visible = v;
    },
  };
}

/** The 202 body of a queued relabel or move. */
export const ACCEPTED = { op_id: 'op1', job_id: 'job1', status: 'queued' } as const;

/** A finished relabel job as `GET /jobs/{id}` serves it. */
export function doneJob(
  o: { updated: number; conflicts?: number },
  id = 'job1',
): JobView {
  return {
    id,
    kind: 'label_relabel',
    status: 'completed',
    result: {
      op_id: 'op1',
      updated: o.updated,
      conflicts: o.conflicts ?? 0,
      complete: true,
    },
  } as unknown as JobView;
}

/** Make `write` (bulkLabel or moveCropsToCluster) queue a job that finishes with these counts.
 *  The test file must `vi.mock('$lib/api_jobs')` so `getJob` is a mock. */
export function mockLabelOp(
  write: unknown,
  o: { updated: number; conflicts?: number },
): void {
  (write as Mock).mockResolvedValue(ACCEPTED);
  (getJob as unknown as Mock).mockResolvedValue(doneJob(o));
}
