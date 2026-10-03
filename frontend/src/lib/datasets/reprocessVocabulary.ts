/**
 * The Reprocess scope, region-mode and embed-part ids the client may SEND,
 * pinned to the vendored request enums by `contract/datasetsContract.test.ts`.
 * Scope labels come from the served `reprocess` vocabulary
 * (`reprocessVocabularyStore`); region modes and embed parts have no served
 * vocabulary, so those two still read through `humanizeId`.
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
