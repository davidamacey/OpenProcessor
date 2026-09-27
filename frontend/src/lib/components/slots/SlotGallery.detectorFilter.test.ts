/**
 * W0 naming-sweep finding m9: SlotGallery's region-gallery Detector <select>
 * used to hardcode `tag_detector_v1`/`sam3`/`paddleocr_det_trt`/`human`
 * options. Its options are now the served vocabulary's filterable
 * detectors (`GET {API_PREFIX}/regions/vocabulary`, `regionVocabularyStore`).
 * Mounts the real component (see SlotGallery.unclustered.test.ts's
 * header comment for the convention).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import SlotGallery from './SlotGallery.svelte';
import { createSlotGalleryController } from '../../../routes/p/[project]/clusters/slotGalleryController.svelte';
import { widgetTagSlot } from '$lib/test/fixtures/regionSlot';
import { regionVocabularyStore } from '$stores/regionVocabulary.svelte';

vi.mock('$lib/api', async () => {
  const actual = await vi.importActual<typeof import('$lib/api')>('$lib/api');
  return { ...actual, getRegionClusters: vi.fn(), getRegions: vi.fn() };
});
import { getRegionClusters } from '$lib/api';

let target: HTMLDivElement;
let instance: unknown;

function resetStore(): void {
  regionVocabularyStore.detectors = [];
  regionVocabularyStore.regionSources = [];
  regionVocabularyStore.chainActors = [];
  regionVocabularyStore.loaded = false;
}

function renderGallery(gallery: ReturnType<typeof createSlotGalleryController>) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(SlotGallery, { target, props: { gallery } } as never);
  flushSync();
  return target;
}

beforeEach(() => {
  resetStore();
});

afterEach(() => {
  if (instance) {
    unmount(instance);
    instance = undefined;
  }
  target?.remove();
});

describe('SlotGallery detector filter — served vocabulary (m9)', () => {
  it("renders only the vocabulary's filterable detectors as options, using their served labels", async () => {
    regionVocabularyStore.detectors = [
      {
        id: 'tag_detector_v1',
        label: 'Tag detector',
        role: 'detector',
        filterable: true,
      },
      { id: 'sam3', label: 'SAM3', role: 'segmenter', filterable: true },
      // Not filterable — must not appear as an option.
      { id: 'human', label: 'Human', role: 'human', filterable: false },
    ];
    vi.mocked(getRegionClusters).mockResolvedValue({ clusters: [] } as never);
    const gallery = createSlotGalleryController(widgetTagSlot);
    await gallery.loadClusters();
    const el = renderGallery(gallery);

    const select = el.querySelector('select') as HTMLSelectElement;
    const optionLabels = [...select.options].map((o) => o.textContent?.trim());
    const optionValues = [...select.options].map((o) => o.value);

    expect(optionLabels).toEqual(['any', 'Tag detector', 'SAM3']);
    expect(optionValues).toEqual(['', 'tag_detector_v1', 'sam3']);
    expect(el.textContent).not.toContain('Paddle det');
    expect(el.textContent).not.toContain('Human');
  });

  it('renders only the "any" option when the vocabulary has no filterable detectors', async () => {
    regionVocabularyStore.detectors = [];
    vi.mocked(getRegionClusters).mockResolvedValue({ clusters: [] } as never);
    const gallery = createSlotGalleryController(widgetTagSlot);
    await gallery.loadClusters();
    const el = renderGallery(gallery);

    const select = el.querySelector('select') as HTMLSelectElement;
    expect([...select.options].map((o) => o.value)).toEqual(['']);
  });
});
