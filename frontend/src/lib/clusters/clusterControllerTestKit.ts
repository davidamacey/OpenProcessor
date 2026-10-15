/**
 * Shared fixtures for the clusterController.*.test.ts files: a minimal
 * crop and a controller wired to a real pager, selection and exclusion
 * guard with spy callbacks. Each test file mocks `$lib/api` itself.
 */
import { vi } from 'vitest';
import {
  createClusterActionController,
  createExclusionGuard,
} from './clusterController.svelte';
import { createPager } from '$lib/pager.svelte';
import { createSelection } from '$lib/selection.svelte';
import type { Crop } from '$lib/types';

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
    cluster_id: 42,
    similarity_to_centroid: null,
    cluster_subid: null,
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
    getClusterId: () => 42,
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
