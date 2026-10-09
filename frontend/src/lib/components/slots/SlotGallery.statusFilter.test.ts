/**
 * dq-region (2026-09-24): SlotGallery's region-gallery gained a Status
 * filter backed by `GET {API_PREFIX}/regions?status=` — options are the
 * served region-status vocabulary (`GET {API_PREFIX}/regions/statuses`,
 * `regionStatusesStore`), not a hardcoded list, mirroring the Detector
 * filter's served-vocabulary pattern (SlotGallery.detectorFilter.test.ts).
 * Mounts the real component.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import SlotGallery from './SlotGallery.svelte';
import { createSlotGalleryController } from '../../../routes/p/[project]/clusters/slotGalleryController.svelte';
import { widgetTagSlot } from '$lib/test/fixtures/regionSlot';
import { regionVocabularyStore } from '$stores/regionVocabulary.svelte';
import { regionStatusesStore } from '$stores/regionStatuses.svelte';

vi.mock('$lib/api', async () => {
  const actual = await vi.importActual<typeof import('$lib/api')>('$lib/api');
  return { ...actual, getRegionClusters: vi.fn(), getRegions: vi.fn() };
});
import { getRegionClusters } from '$lib/api';

let target: HTMLDivElement;
let instance: unknown;

function resetStores(): void {
  regionVocabularyStore.detectors = [];
  regionVocabularyStore.regionSources = [];
  regionVocabularyStore.chainActors = [];
  regionVocabularyStore.loaded = false;
  regionStatusesStore.list = [];
  regionStatusesStore.confirmStatus = null;
  regionStatusesStore.rejectStatus = null;
  regionStatusesStore.falsePositiveStatus = null;
  regionStatusesStore.loaded = false;
}

function renderGallery(gallery: ReturnType<typeof createSlotGalleryController>) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(SlotGallery, { target, props: { gallery } } as never);
  flushSync();
  return target;
}

function statusSelect(el: HTMLElement): HTMLSelectElement {
  // The label whose text includes "Status" and contains a <select> — not
  // positional indexing, so a template reorder (or the "Verified only"
  // checkbox label sitting between them) can't silently pick the wrong
  // control.
  const label = [...el.querySelectorAll('label')].find(
    (l) => l.textContent?.includes('Status') && l.querySelector('select'),
  );
  return label!.querySelector('select') as HTMLSelectElement;
}

beforeEach(() => {
  resetStores();
});

afterEach(() => {
  if (instance) {
    unmount(instance);
    instance = undefined;
  }
  target?.remove();
});

describe('SlotGallery status filter — served vocabulary (dq-region)', () => {
  it('renders the served region-status vocabulary as options', async () => {
    regionStatusesStore.list = [
      {
        value: 'detected',
        label: 'Detected',
        role: 'proposed',
        terminal: false,
        human_writable: true,
        clears_box: false,
        wants_reason: false,
      },
      {
        value: 'verify_rejected',
        label: 'Rejected (candidate kept)',
        role: 'rejected',
        terminal: false,
        human_writable: true,
        clears_box: false,
        wants_reason: false,
      },
    ];
    vi.mocked(getRegionClusters).mockResolvedValue({ clusters: [] } as never);
    const gallery = createSlotGalleryController(widgetTagSlot);
    await gallery.loadClusters();
    const el = renderGallery(gallery);

    const select = statusSelect(el);
    const optionLabels = [...select.options].map((o) => o.textContent?.trim());
    const optionValues = [...select.options].map((o) => o.value);
    expect(optionLabels).toEqual(['any', 'Detected', 'Rejected (candidate kept)']);
    expect(optionValues).toEqual(['', 'detected', 'verify_rejected']);
  });

  it('renders only the "any" option when the status vocabulary is empty', async () => {
    vi.mocked(getRegionClusters).mockResolvedValue({ clusters: [] } as never);
    const gallery = createSlotGalleryController(widgetTagSlot);
    await gallery.loadClusters();
    const el = renderGallery(gallery);

    const select = statusSelect(el);
    expect([...select.options].map((o) => o.value)).toEqual(['']);
  });

  it('picking a status sets gallery.statusFilter (the value browseQuery forwards to ?status=)', async () => {
    regionStatusesStore.list = [
      {
        value: 'verify_rejected',
        label: 'Rejected',
        role: 'rejected',
        terminal: false,
        human_writable: true,
        clears_box: false,
        wants_reason: false,
      },
    ];
    vi.mocked(getRegionClusters).mockResolvedValue({ clusters: [] } as never);
    const gallery = createSlotGalleryController(widgetTagSlot);
    await gallery.loadClusters();
    const el = renderGallery(gallery);

    const select = statusSelect(el);
    select.value = 'verify_rejected';
    select.dispatchEvent(new Event('change'));
    flushSync();

    expect(gallery.statusFilter).toBe('verify_rejected');
  });
});
