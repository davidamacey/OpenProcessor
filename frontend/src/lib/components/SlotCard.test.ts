/**
 * Mount-based behavior test for SlotCard (docs/design/test-audit-2026-09-24.md
 * P1-4): a region browse row is the full item plus `region_box_id`; the card
 * describes that row's own box (score, reading, detector, state, thumbnail),
 * asserted on the rendered DOM, not a source scan.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { reprocessVocabularyStore } from '$lib/stores/reprocessVocabulary.svelte';
import { reprocessVocabularyFixture } from '$lib/test/fixtures/regionProfiles';
import { mount, unmount, flushSync } from 'svelte';
import SlotCard from './SlotCard.svelte';
import type { RegionBrowseItem } from '$lib/api';
import { regionSlotFromServedProfile } from '$lib/annotations/servedRegionSlot';
import { widgetTagSlot, WIDGET_TAG_PROFILE_NO_TEXT } from '$lib/test/fixtures/regionSlot';
import { API_PREFIX } from '$lib/api';
import { datasetsAvailability } from '$lib/datasets/datasetsAvailability.svelte';
import { formatsFixture } from '$lib/test/fixtures/datasetImport';

const wireBox = (over: Record<string, unknown> = {}) => ({
  box_id: 'b1',
  state: 'accepted',
  bbox_norm: [0.4, 0.4, 0.6, 0.6],
  bbox_in_parent: [0.4, 0.4, 0.6, 0.6],
  score: 0.9,
  detector: 'tag_detector_v1',
  text: 'TAG-001',
  text_source: 'ocr',
  text_confidence: 0.8,
  thumbnail_url: '/curation/crops/c1/region_thumbnail?box_id=b1',
  ...over,
});

function fakeRegionItem(
  overrides: Record<string, unknown> = {},
  boxes: Array<Record<string, unknown>> = [wireBox()],
): RegionBrowseItem {
  return {
    crop_id: 'c1',
    id: 'c1',
    image_path: '/img.jpg',
    bbox_norm: [0.3, 0.3, 0.7, 0.7],
    region_status: 'detected',
    region_verified: true,
    region_validated: true,
    region_detector_chain: null,
    region_detected_at: null,
    region_verifier: null,
    region_verifier_version: null,
    region_verified_at: null,
    region_rejection_reason: null,
    region_visible: true,
    region_boxes: boxes,
    class_id: 3,
    class_name: 'widget_a',
    cluster_id: null,
    updated_at: '',
    ...overrides,
  } as unknown as RegionBrowseItem;
}

let target: HTMLDivElement;
let instance: unknown;

function renderCard(crop: RegionBrowseItem, slot = widgetTagSlot) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(SlotCard, { target, props: { crop, slot } });
  flushSync();
  return target;
}

afterEach(() => {
  if (instance) {
    unmount(instance);
    instance = undefined;
  }
  target?.remove();
});

describe('SlotCard — reader disagreement flag', () => {
  it('renders the readers-disagree chip when the box says its readers disagree', () => {
    const el = renderCard(
      fakeRegionItem({}, [
        wireBox({ text_disagreement: true, text_vlm: 'A', text_ocr: 'B' }),
      ]),
    );
    expect(el.textContent).toContain('readers disagree');
    // C6 (visual audit 2026-09-24): plain text, no emoji warning glyph.
    expect(el.textContent).not.toContain('⚠');
  });

  it('does not render the chip when text_disagreement is false or absent', () => {
    expect(
      renderCard(fakeRegionItem({}, [wireBox({ text_disagreement: false })])).textContent,
    ).not.toContain('readers disagree');
    unmount(instance as never);
    instance = undefined;
    target.remove();
    expect(renderCard(fakeRegionItem()).textContent).not.toContain('readers disagree');
  });
});

describe('SlotCard — the row describes its own box', () => {
  const two = [
    wireBox({ box_id: 'b1', score: 0.9, text: 'TAG-001' }),
    wireBox({
      box_id: 'b2',
      state: 'false_positive',
      score: 0.31,
      text: 'TAG-002',
      thumbnail_url: '/curation/crops/c1/region_thumbnail?box_id=b2',
    }),
  ];

  it('a row naming box b2 shows b2 score, reading, false-positive badge and thumbnail', () => {
    const el = renderCard(fakeRegionItem({ region_box_id: 'b2' }, two));
    expect(el.querySelector('[data-testid="slot-text-value"]')?.textContent).toBe(
      'TAG-002',
    );
    expect(el.textContent).toContain('31%');
    expect(el.textContent).toContain('false pos');
    expect(el.querySelector('img')?.getAttribute('src')).toContain('box_id=b2');
    // A false-positive box is dimmed.
    expect(el.querySelector('button')?.className).toContain('opacity-50');
  });

  it('a row naming box b1 shows the box count and its own state', () => {
    const el = renderCard(fakeRegionItem({ region_box_id: 'b1' }, two));
    expect(el.textContent).toMatch(/2\sboxes/);
    expect(el.textContent).toContain('accepted');
  });

  it('an item-level row (no region_box_id) shows the first box', () => {
    const el = renderCard(fakeRegionItem({}, two));
    expect(el.querySelector('[data-testid="slot-text-value"]')?.textContent).toBe(
      'TAG-001',
    );
    expect(el.querySelector('img')?.getAttribute('src')).toContain('box_id=b1');
  });

  it('the served region_revision rides on the thumbnail URL so an edited box is re-cropped', () => {
    const el = renderCard(fakeRegionItem({ region_revision: 7 }, [wireBox()]));
    expect(el.querySelector('img')?.getAttribute('src')).toMatch(/box_id=b1&v=7$/);
  });

  it("builds the slot's own thumbnail path for a box with no served thumbnail_url", () => {
    const el = renderCard(fakeRegionItem({}, [wireBox({ thumbnail_url: null })]));
    expect(el.querySelector('img')?.getAttribute('src')).toBe(
      '/curation/crops/c1/region_thumbnail?box_id=b1&size=160',
    );
  });

  it('an item with no boxes shows the item thumbnail, not a region crop', () => {
    const el = renderCard(fakeRegionItem({}, []));
    expect(el.querySelector('img')?.getAttribute('src')).toContain('/crops/c1/thumbnail');
    expect(el.querySelector('img')?.getAttribute('src')).not.toContain(
      'region_thumbnail',
    );
  });

  it('badges a rejected box as a candidate', () => {
    const el = renderCard(fakeRegionItem({}, [wireBox({ state: 'rejected' })]));
    expect(el.textContent).toContain('candidate');
  });
});

describe('SlotCard — text-free region profile (OpenProcessor W1)', () => {
  it('renders no text value row at all for a slot with no text capability', () => {
    const noTextSlot = regionSlotFromServedProfile(WIDGET_TAG_PROFILE_NO_TEXT);
    expect(noTextSlot.capabilities.text).toBeUndefined();
    const el = renderCard(fakeRegionItem({}, [wireBox({ text: null })]), noTextSlot);
    expect(el.querySelector('[data-testid="slot-text-value"]')).toBeNull();
  });

  it('still renders the text value row (even with no reading yet) for a text-reading slot', () => {
    const el = renderCard(fakeRegionItem({}, [wireBox({ text: null })]));
    expect(el.querySelector('[data-testid="slot-text-value"]')).not.toBeNull();
  });
});

describe('SlotCard — W10 box lock glyph', () => {
  afterEach(() => reprocessVocabularyStore.resetForProjectChange());
  it('shows the lock only when the served box locked is true', () => {
    reprocessVocabularyStore.vocabulary = reprocessVocabularyFixture();
    reprocessVocabularyStore.loaded = true;
    const locked = renderCard(fakeRegionItem({}, [wireBox({ locked: true })]));
    expect(locked.querySelector('[data-testid="box-locked"]')).not.toBeNull();
    expect(
      locked.querySelector('[data-testid="box-locked"]')?.getAttribute('title'),
    ).toBe(
      'Locked. Locked when: Human label: Human label: served description; Validated: Validated: served description; Imported: Imported: served description; Test holdout: Test holdout: served description',
    );
  });

  it('shows no lock for false or absent', () => {
    const el = renderCard(fakeRegionItem({}, [wireBox({ locked: false })]));
    expect(el.querySelector('[data-testid="box-locked"]')).toBeNull();
    unmount(instance as never);
    instance = undefined;
    target.remove();
    const el2 = renderCard(fakeRegionItem({}, [wireBox()]));
    expect(el2.querySelector('[data-testid="box-locked"]')).toBeNull();
  });
});

describe('SlotCard — W10 image Reprocess', () => {
  beforeEach(() => datasetsAvailability.reset());
  afterEach(() => {
    vi.unstubAllGlobals();
    datasetsAvailability.reset();
  });

  function serve(status: number) {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string) =>
        String(url) === `${API_PREFIX}/datasets/formats`
          ? new Response(
              JSON.stringify(status === 200 ? formatsFixture() : { detail: 'x' }),
              {
                status,
                headers: { 'content-type': 'application/json' },
              },
            )
          : new Response('{}', { status: 404 }),
      ),
    );
  }

  const entry = (el: HTMLElement) =>
    el.querySelector(
      '[data-testid="card-reprocess-image"] [data-testid="reprocess-open"]',
    );

  it('offers "Reprocess image…" outside the card button when W10 is served', async () => {
    serve(200);
    const el = renderCard(fakeRegionItem({ image_id: 'img_1' }));
    await datasetsAvailability.init();
    flushSync();
    const open = entry(el);
    expect(open?.textContent?.trim()).toBe('Reprocess image…');
    // A button cannot nest inside the card's own <button>.
    expect(open?.closest('button[title]')).toBeNull();
  });

  it('is absent without an image id, or when W10 is not served', async () => {
    serve(200);
    const el = renderCard(fakeRegionItem());
    await datasetsAvailability.init();
    flushSync();
    expect(entry(el)).toBeNull();
    unmount(instance as never);
    instance = undefined;
    target.remove();
    datasetsAvailability.reset();
    serve(404);
    const el2 = renderCard(fakeRegionItem({ image_id: 'img_1' }));
    await datasetsAvailability.init();
    flushSync();
    expect(entry(el2)).toBeNull();
  });

  describe('SlotCard — provenance chain', () => {
    it('renders every step of a chain that repeats an entry (a keyed list must not throw)', () => {
      const el = renderCard(
        fakeRegionItem({
          region_detector_chain: ['tag_detector_v1:miss', 'tag_detector_v1:miss'],
        }),
      );
      expect(el.textContent?.match(/miss/g)?.length ?? 0).toBeGreaterThanOrEqual(2);
    });
  });
});
