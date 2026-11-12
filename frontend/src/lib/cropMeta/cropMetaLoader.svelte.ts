/**
 * The crop detail panel's two lazy reads: the write history and the source
 * image (its facts, siblings and the full response for SourceImageOverlay).
 * Each refetches when the crop (or its image) changes, aborts the superseded
 * request, and never blocks the rest of the panel.
 */

import { getCropHistory, getImage } from '$lib/api_cropRead';
import { apiErrorText } from '$lib/api_errors';
import type { Crop, CropHistoryEntry } from '$lib/types';
import type { ImageDetailResponse, ImageInfo } from '$lib/types_images';

export function createCropMetaLoader(crop: () => Pick<Crop, 'id' | 'image_id'>) {
  // -- History (G7) — lazy, fetched once per crop.id, never blocks the
  // rest of the panel from rendering. ------------------------------------
  let historyEntries = $state<CropHistoryEntry[] | null>(null);
  let historyError = $state<string | null>(null);
  let historyLoading = $state(false);

  // -- Source image + siblings (G7/G8) — same lazy-on-open pattern. ------
  let imageMeta = $state<ImageInfo | null>(null);
  let siblings = $state<Crop[] | null>(null);
  let imageError = $state<string | null>(null);
  let imageLoading = $state(false);
  // The full response, handed straight to SourceImageOverlay so it
  // doesn't re-fetch what this panel already loaded.
  let contextRes = $state<ImageDetailResponse | null>(null);

  $effect(() => {
    const id = crop().id;
    historyEntries = null;
    historyError = null;
    historyLoading = true;
    const controller = new AbortController();
    getCropHistory(id, controller.signal)
      .then((res) => {
        historyEntries = res.entries;
      })
      .catch((e: unknown) => {
        if ((e as Error)?.name === 'AbortError') return;
        historyError = apiErrorText(e);
      })
      .finally(() => {
        historyLoading = false;
      });
    return () => controller.abort();
  });

  $effect(() => {
    const { id, image_id: imageId } = crop();
    imageMeta = null;
    siblings = null;
    contextRes = null;
    imageError = null;
    if (!imageId) {
      imageLoading = false;
      imageError = 'No source image is recorded for this item.';
      return;
    }
    imageLoading = true;
    const controller = new AbortController();
    getImage(imageId, controller.signal)
      .then((res) => {
        imageMeta = res.image;
        siblings = res.items.filter((it) => it.id !== id);
        contextRes = res;
      })
      .catch((e: unknown) => {
        if ((e as Error)?.name === 'AbortError') return;
        imageError = apiErrorText(e);
      })
      .finally(() => {
        imageLoading = false;
      });
    return () => controller.abort();
  });

  return {
    get historyEntries() {
      return historyEntries;
    },
    get historyError() {
      return historyError;
    },
    get historyLoading() {
      return historyLoading;
    },
    get imageMeta() {
      return imageMeta;
    },
    get siblings() {
      return siblings;
    },
    get imageError() {
      return imageError;
    },
    get imageLoading() {
      return imageLoading;
    },
    get contextRes() {
      return contextRes;
    },
  };
}

export type CropMetaLoader = ReturnType<typeof createCropMetaLoader>;
