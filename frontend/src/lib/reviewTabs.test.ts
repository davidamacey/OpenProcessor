import { afterEach, describe, expect, it, vi } from 'vitest';
import { getReviewQueue } from './api';
import { licensePlateSlot } from './annotations/profiles/licensePlate';
import {
  buildReviewTabs,
  CORE_REVIEW_TABS,
  endpointForTab,
  isSlotTab,
  REVIEW_PRESETS,
  REVIEW_TABS,
  resolveEffectiveTab,
  type ReviewPresetId,
} from './reviewTabs';

describe('REVIEW_TABS (2026-09 tab consolidation)', () => {
  it('has exactly 5 top-level tabs, down from 9', () => {
    expect(REVIEW_TABS).toHaveLength(5);
  });

  it('is exactly all / uncertainty / model_disagreements / coco_blind_spots / plates', () => {
    expect(REVIEW_TABS.map((t) => t.id)).toEqual([
      'all',
      'uncertainty',
      'model_disagreements',
      'coco_blind_spots',
      'plates',
    ]);
  });

  it('never renders Outliers as a tab', () => {
    expect(REVIEW_TABS.some((t) => t.id === ('outliers' as string))).toBe(false);
    expect(REVIEW_TABS.some((t) => /outlier/i.test(t.label))).toBe(false);
  });

  it('never renders the three collapsed queues as tabs (they are preset chips now)', () => {
    const collapsed = ['mismatches', 'gemma_low_conf', 'primary_low_conf'];
    for (const id of collapsed) {
      expect(REVIEW_TABS.some((t) => t.id === id)).toBe(false);
    }
  });
});

describe('buildReviewTabs (P2.8 data-driving)', () => {
  it('derives a tab from a queue-capable slot, matching its urlId/endpointId/label', () => {
    const tabs = buildReviewTabs([licensePlateSlot]);
    expect(tabs).toHaveLength(1);
    expect(tabs[0].id).toBe('plates');
    expect(tabs[0].urlId).toBe('plates');
    expect(tabs[0].endpointId).toBe('plates');
    expect(tabs[0].label).toBe('Plates');
    expect(tabs[0].slot).toBe(licensePlateSlot);
  });

  it('skips a slot with no queue capability', () => {
    const noQueueSlot = {
      ...licensePlateSlot,
      capabilities: { text: licensePlateSlot.capabilities.text },
    };
    expect(buildReviewTabs([noQueueSlot])).toEqual([]);
  });

  it('REVIEW_TABS is CORE_REVIEW_TABS plus the derived slot tabs, in that order', () => {
    expect(REVIEW_TABS).toEqual([
      ...CORE_REVIEW_TABS,
      ...buildReviewTabs([licensePlateSlot]),
    ]);
  });
});

describe('isSlotTab', () => {
  it('is true for the plates tab (backed by a slot queue)', () => {
    expect(isSlotTab('plates')).toBe(true);
  });

  it('is false for every core tab', () => {
    for (const t of CORE_REVIEW_TABS) expect(isSlotTab(t.id)).toBe(false);
  });

  it('is false for a preset id (not a real tab)', () => {
    expect(isSlotTab('mismatches')).toBe(false);
  });
});

describe('endpointForTab', () => {
  it('resolves a core tab to its own id', () => {
    expect(endpointForTab('uncertainty')).toBe('uncertainty');
  });

  it('resolves the plates tab to its endpointId (identical today, but a distinct lookup)', () => {
    expect(endpointForTab('plates')).toBe('plates');
  });

  it('falls through to the raw id for anything not in REVIEW_TABS (e.g. a preset id)', () => {
    expect(endpointForTab('mismatches')).toBe('mismatches');
  });
});

describe('REVIEW_PRESETS (All-tab quick-filter chips)', () => {
  it('offers exactly the three collapsed queues', () => {
    expect(REVIEW_PRESETS.map((p) => p.id).sort()).toEqual(
      ['gemma_low_conf', 'mismatches', 'primary_low_conf'].sort(),
    );
  });

  it('gives each preset a distinct, non-empty label', () => {
    const labels = REVIEW_PRESETS.map((p) => p.label);
    expect(new Set(labels).size).toBe(labels.length);
    for (const label of labels) expect(label.trim().length).toBeGreaterThan(0);
  });
});

describe('resolveEffectiveTab', () => {
  it('passes non-all tabs straight through, ignoring any stale preset', () => {
    expect(resolveEffectiveTab('uncertainty', null)).toBe('uncertainty');
    expect(resolveEffectiveTab('plates', 'mismatches')).toBe('plates');
    expect(resolveEffectiveTab('coco_blind_spots', 'gemma_low_conf')).toBe(
      'coco_blind_spots',
    );
  });

  it('resolves the all tab with no active preset to plain all', () => {
    expect(resolveEffectiveTab('all', null)).toBe('all');
  });

  it.each(REVIEW_PRESETS.map((p) => p.id))(
    'resolves the all tab with the %s preset active to that preset id',
    (presetId: ReviewPresetId) => {
      expect(resolveEffectiveTab('all', presetId)).toBe(presetId);
    },
  );
});

/**
 * Prove the preset mechanism actually reaches real data, not just that a
 * button exists: resolveEffectiveTab's output is exactly what +page.svelte
 * passes to getReviewQueue, which hits `GET /curation/review/{tab}` — the same
 * endpoint the old top-level tab used. This locks the full chip -> fetch
 * -> URL chain together so a future refactor can't silently point a chip
 * at the wrong cohort (or at `all` itself, which would make the preset a
 * no-op returning the full unfiltered queue instead of the curated one).
 */
describe('preset chip -> real queue fetch (regression: chip must not become a no-op)', () => {
  const jsonResponse = (body: unknown) =>
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it.each(REVIEW_PRESETS.map((p) => p.id))(
    'the %s chip on the All tab fetches /curation/review/%s, not /curation/review/all',
    async (presetId: ReviewPresetId) => {
      const fetchMock = vi
        .fn()
        .mockResolvedValue(
          jsonResponse({ total: 42, page: 1, page_size: 30, items: [] }),
        );
      vi.stubGlobal('fetch', fetchMock);

      const effectiveTab = resolveEffectiveTab('all', presetId);
      const res = await getReviewQueue(effectiveTab, 1, 30, {});

      const url = fetchMock.mock.calls[0]?.[0] as string;
      expect(url).toContain(`/curation/review/${presetId}`);
      expect(url).not.toContain('/curation/review/all');
      expect(res.total).toBe(42);
    },
  );

  it('plain All (no preset) still fetches /curation/review/all', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        jsonResponse({ total: 56797, page: 1, page_size: 30, items: [] }),
      );
    vi.stubGlobal('fetch', fetchMock);

    const effectiveTab = resolveEffectiveTab('all', null);
    const res = await getReviewQueue(effectiveTab, 1, 30, {});

    const url = fetchMock.mock.calls[0]?.[0] as string;
    expect(url).toContain('/curation/review/all');
    expect(res.total).toBe(56797);
  });
});
