/**
 * The crop detail panel's two lazy reads: the write history and the source
 * image context (image meta, siblings and the full context response for
 * SourceImageOverlay). Each refetches when the crop id changes, aborts the
 * superseded request, and never blocks the rest of the panel.
 */

import { apiErrorText, getCropContext, getCropHistory } from '$lib/api';
import type {
  Crop,
  CropContextResponse,
  CropHistoryEntry,
  CropImageMeta,
} from '$lib/types';

export function createCropMetaLoader(cropId: () => string) {
  // -- History (G7) — lazy, fetched once per crop.id, never blocks the
  // rest of the panel from rendering. ------------------------------------
  let historyEntries = $state<CropHistoryEntry[] | null>(null);
  let historyError = $state<string | null>(null);
  let historyLoading = $state(false);

  // -- Source image + siblings (G7/G8) — same lazy-on-open pattern. ------
  let imageMeta = $state<CropImageMeta | null>(null);
  let siblings = $state<Crop[] | null>(null);
  let imageError = $state<string | null>(null);
  let imageLoading = $state(false);
  // K6: the full context response, handed straight to SourceImageOverlay
  // so it doesn't re-fetch what this panel already loaded.
  let contextRes = $state<CropContextResponse | null>(null);

  $effect(() => {
    const id = cropId();
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
    const id = cropId();
    imageMeta = null;
    siblings = null;
    contextRes = null;
    imageError = null;
    imageLoading = true;
    const controller = new AbortController();
    getCropContext(id, controller.signal)
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
