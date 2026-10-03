/**
 * The Reprocess scope and region-mode ids. The backend serves no
 * vocabulary for these (`/datasets/formats` has no `reprocess` block), so
 * the ids are pinned to the vendored `ReprocessOneRequest` enums by
 * `contract/datasetsContract.test.ts` and labelled by `humanizeId`
 * (question C-1: replace with served labels when they exist).
 */
import { humanizeId } from '$lib/humanizeId';
import type {
  EmbedOptions,
  ReprocessRegionMode,
  ReprocessScope,
} from '$lib/types_import';

export const REPROCESS_SCOPES: readonly ReprocessScope[] = [
  'detect',
  'open_vocab',
  'region',
  'vlm',
  'embed',
];

export const REGION_MODES: readonly ReprocessRegionMode[] = ['redetect', 'reverify'];

/** The `embed.parts` ids, pinned to the vendored `EmbedOptions` by
 *  `contract/detectorContract.test.ts`. */
export const EMBED_PARTS: readonly NonNullable<EmbedOptions['parts']>[number][] = [
  'crop',
  'frame',
  'region',
];

export const reprocessLabel = (id: string): string => humanizeId(id);
