/**
 * The Reprocess scope and region-mode ids. The backend serves no
 * vocabulary for these (`/datasets/formats` has no `reprocess` block), so
 * the ids are pinned to the vendored `ReprocessOneRequest` enums by
 * `contract/datasetsContract.test.ts` and labelled by `humanizeId`
 * (question C-1: replace with served labels when they exist).
 */
import { humanizeId } from '$lib/humanizeId';
import type { ReprocessRegionMode, ReprocessScope } from '$lib/types_import';

export const REPROCESS_SCOPES: readonly ReprocessScope[] = [
  'detect',
  'region',
  'vlm',
  'embed',
];

export const REGION_MODES: readonly ReprocessRegionMode[] = ['redetect', 'reverify'];

export const reprocessLabel = (id: string): string => humanizeId(id);
