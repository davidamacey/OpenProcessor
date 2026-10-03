import { afterEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX, getReviewQueue } from './api';
import { WIDGET_TAG_PROFILE, widgetTagSlot } from '$lib/test/fixtures/regionSlot';
import { registeredSlots, resetDeploymentSlots } from './annotations/registeredSlots';
import { installedQueueSlots } from '$lib/test/fixtures/installedQueueSlots';
import {
  buildReviewTabs,
  CORE_REVIEW_TABS,
  endpointForTab,
  isSlotTab,
  REVIEW_PRESETS,
  REVIEW_TABS,
  resolveEffectiveTab,
  slotTabId,
  reviewDeepLink,
  unavailableTabMessage,
  tabFromUrlId,
  tabHonorsPinnedSortDefault,
  visibleReviewTabs,
  type ReviewPresetId,
} from './reviewTabs';

// The registered queue slots (whatever this build registers); each gets
// exactly one tab after the core tabs.
const queueSlots = registeredSlots.filter((s) => s.capabilities.queue);

// `installedQueueSlots()` installs a served profile for the loops below.
afterEach(() => resetDeploymentSlots());

describe('REVIEW_TABS (2026-09 tab consolidation)', () => {
  it('has the core tabs from the 2026-09 consolidation plus new_class_proposals and imported, then one tab per queue slot', () => {
    expect(CORE_REVIEW_TABS).toHaveLength(6);
    expect(REVIEW_TABS).toHaveLength(6 + queueSlots.length);
  });

  it('is exactly all / uncertainty / model_disagreements / classifier_blind_spots / new_class_proposals / imported / slot:<key>...', () => {
    expect(REVIEW_TABS.map((t) => t.id)).toEqual([
      'all',
      'uncertainty',
      'model_disagreements',
      'classifier_blind_spots',
      'new_class_proposals',
      'imported',
      ...queueSlots.map((s) => `slot:${s.key}`),
    ]);
    // urlId is the bookmark contract — each slot tab keeps its own.
    for (const s of queueSlots) {
      expect(REVIEW_TABS.find((t) => t.id === `slot:${s.key}`)?.urlId).toBe(
        s.capabilities.queue!.urlId,
      );
    }
    // new_class_proposals (2026-09-24 logic-moves W5) is a real core tab —
    // its urlId/endpointId are the raw backend tab id, same as every
    // other core tab.
    expect(REVIEW_TABS.find((t) => t.id === 'new_class_proposals')).toMatchObject({
      urlId: 'new_class_proposals',
      endpointId: 'new_class_proposals',
    });
  });

  it('no region profile: exactly the core tabs (audit §5.4)', async () => {
    const reg = await import('./annotations/registeredSlots');
    const tabs = await import('./reviewTabs');
    reg.installServedRegionProfile(null);
    try {
      expect(tabs.REVIEW_TABS).toEqual(CORE_REVIEW_TABS);
      expect(tabs.tabFromUrlId('regions')).toBeUndefined();
    } finally {
      reg.resetDeploymentSlots();
    }
  });

  it('a served region profile adds one region tab labelled by its display_name', async () => {
    const reg = await import('./annotations/registeredSlots');
    const tabs = await import('./reviewTabs');
    reg.installServedRegionProfile(WIDGET_TAG_PROFILE);
    try {
      expect(tabs.REVIEW_TABS).toHaveLength(CORE_REVIEW_TABS.length + 1);
      expect(tabs.REVIEW_TABS.at(-1)).toMatchObject({
        id: `slot:${WIDGET_TAG_PROFILE.name}`,
        label: WIDGET_TAG_PROFILE.display_name,
        urlId: 'regions',
        endpointId: 'regions',
      });
      expect(tabs.tabFromUrlId('regions')).toBe(`slot:${WIDGET_TAG_PROFILE.name}`);
    } finally {
      reg.resetDeploymentSlots();
    }
  });

  it('every core tab uses the served tab id for its id, urlId and endpointId (naming-w2 F7: classifier_blind_spots)', () => {
    for (const t of CORE_REVIEW_TABS) {
      expect(t.urlId, t.id).toBe(t.id);
      expect(t.endpointId, t.id).toBe(t.id);
    }
    expect(tabFromUrlId('classifier_blind_spots')).toBe('classifier_blind_spots');
    expect(tabFromUrlId('coco_blind_spots')).not.toBe('classifier_blind_spots');
  });

  it('never renders Outliers as a tab', () => {
    expect(REVIEW_TABS.some((t) => t.id === ('outliers' as string))).toBe(false);
    expect(REVIEW_TABS.some((t) => /outlier/i.test(t.label))).toBe(false);
  });

  it('never renders the three collapsed queues as tabs (they are preset chips now)', () => {
    const collapsed = ['mismatches', 'vlm_low_conf', 'primary_low_conf'];
    for (const id of collapsed) {
      expect(REVIEW_TABS.some((t) => t.id === id)).toBe(false);
    }
  });
});

describe('buildReviewTabs (P2.8 data-driving)', () => {
  it('derives a tab from a queue-capable slot, matching its urlId/endpointId/label', () => {
    const tabs = buildReviewTabs([widgetTagSlot]);
    expect(tabs).toHaveLength(1);
    expect(tabs[0].id).toBe('slot:widget_tag');
    expect(tabs[0].urlId).toBe('regions');
    expect(tabs[0].endpointId).toBe('regions');
    expect(tabs[0].label).toBe('Widget tags');
    expect(tabs[0].slot).toBe(widgetTagSlot);
  });

  it('skips a slot with no queue capability', () => {
    const noQueueSlot = {
      ...widgetTagSlot,
      capabilities: { text: widgetTagSlot.capabilities.text },
    };
    expect(buildReviewTabs([noQueueSlot])).toEqual([]);
  });

  it('REVIEW_TABS is CORE_REVIEW_TABS plus the derived slot tabs, in that order', () => {
    expect(REVIEW_TABS).toEqual([
      ...CORE_REVIEW_TABS,
      ...buildReviewTabs(registeredSlots),
    ]);
  });
});

describe('isSlotTab', () => {
  it('is true for a slot tab (backed by a slot queue)', () => {
    expect(isSlotTab(slotTabId(widgetTagSlot.key))).toBe(true);
  });

  it('is structurally true for any slot: id, even an unregistered one', () => {
    expect(isSlotTab(slotTabId('some_future_slot'))).toBe(true);
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

  it("resolves each registered slot tab to its slot's endpointId", () => {
    for (const s of installedQueueSlots()) {
      expect(endpointForTab(slotTabId(s.key))).toBe(s.capabilities.queue!.endpointId);
    }
  });

  it('falls through to the raw id for anything not in REVIEW_TABS (e.g. a preset id)', () => {
    expect(endpointForTab('mismatches')).toBe('mismatches');
  });
});

describe('tabFromUrlId (bookmark contract)', () => {
  it("resolves each registered slot's urlId to its slot: tab", () => {
    for (const s of installedQueueSlots()) {
      expect(tabFromUrlId(s.capabilities.queue!.urlId)).toBe(slotTabId(s.key));
    }
  });

  it('resolves a core tab urlId to itself', () => {
    expect(tabFromUrlId('uncertainty')).toBe('uncertainty');
  });

  it('returns undefined for an unknown urlId', () => {
    expect(tabFromUrlId('nope')).toBeUndefined();
  });
});

describe('REVIEW_PRESETS (All-tab quick-filter chips)', () => {
  it('offers exactly the three collapsed queues', () => {
    expect(REVIEW_PRESETS.map((p) => p.id).sort()).toEqual(
      ['vlm_low_conf', 'mismatches', 'primary_low_conf'].sort(),
    );
  });

  it('gives each preset a distinct, non-empty label', () => {
    const labels = REVIEW_PRESETS.map((p) => p.label);
    expect(new Set(labels).size).toBe(labels.length);
    for (const label of labels) expect(label.trim().length).toBeGreaterThan(0);
  });
});

// S1 (visual audit 2026-09-24): only these two tabs have no tuned default
// sort of their own — CurationSettings.ts's `sort` axis blurb names them
// explicitly (curationSettings.test.ts pins that prose). Every other core
// tab, every slot tab, and every preset id must return false.
describe('tabHonorsPinnedSortDefault', () => {
  it('is true for all and new_class_proposals', () => {
    expect(tabHonorsPinnedSortDefault('all')).toBe(true);
    expect(tabHonorsPinnedSortDefault('new_class_proposals')).toBe(true);
  });

  it('is false for every tab with its own tuned default', () => {
    expect(tabHonorsPinnedSortDefault('uncertainty')).toBe(false);
    expect(tabHonorsPinnedSortDefault('model_disagreements')).toBe(false);
    expect(tabHonorsPinnedSortDefault('classifier_blind_spots')).toBe(false);
  });

  it('is false for a slot tab', () => {
    for (const t of installedQueueSlots()) {
      expect(tabHonorsPinnedSortDefault(slotTabId(t.key))).toBe(false);
    }
  });

  it('is false for a preset id — a preset reuses its own former tab default, not All', () => {
    for (const p of REVIEW_PRESETS) {
      expect(tabHonorsPinnedSortDefault(p.id as never)).toBe(false);
    }
  });
});

describe('resolveEffectiveTab', () => {
  it('passes non-all tabs straight through, ignoring any stale preset', () => {
    expect(resolveEffectiveTab('uncertainty', null)).toBe('uncertainty');
    expect(resolveEffectiveTab('slot:widget_tag', 'mismatches')).toBe('slot:widget_tag');
    expect(resolveEffectiveTab('classifier_blind_spots', 'vlm_low_conf')).toBe(
      'classifier_blind_spots',
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
 * passes to getReviewQueue, which hits `GET {API_PREFIX}/review/{tab}` — the same
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
    'the %s chip on the All tab fetches {API_PREFIX}/review/%s, not {API_PREFIX}/review/all',
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
      expect(url).toContain(`${API_PREFIX}/review/${presetId}`);
      expect(url).not.toContain(`${API_PREFIX}/review/all`);
      expect(res.total).toBe(42);
    },
  );

  it('plain All (no preset) still fetches {API_PREFIX}/review/all', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        jsonResponse({ total: 56797, page: 1, page_size: 30, items: [] }),
      );
    vi.stubGlobal('fetch', fetchMock);

    const effectiveTab = resolveEffectiveTab('all', null);
    const res = await getReviewQueue(effectiveTab, 1, 30, {});

    const url = fetchMock.mock.calls[0]?.[0] as string;
    expect(url).toContain(`${API_PREFIX}/review/all`);
    expect(res.total).toBe(56797);
  });
});

describe('reviewDeepLink', () => {
  it('opens the slot tab from its bookmark urlId, with the crop to jump to', () => {
    for (const s of installedQueueSlots()) {
      const d = reviewDeepLink(
        new URLSearchParams(`tab=${s.capabilities.queue!.urlId}&crop_id=abc`),
      );
      expect(d.tab).toBe(slotTabId(s.key));
      expect(d.cropId).toBe('abc');
    }
  });

  it('falls back to All for an absent or unknown tab, and no crop', () => {
    expect(reviewDeepLink(new URLSearchParams(''))).toEqual({
      tab: 'all',
      cropId: null,
      preset: null,
      importId: null,
      combineConflict: false,
      unavailableTab: null,
    });
    expect(reviewDeepLink(new URLSearchParams('tab=nope&crop_id=')).tab).toBe('all');
  });

  describe('preset (m31, 2026-09-24 interactive pass)', () => {
    it('reads a valid preset id on the all tab', () => {
      expect(reviewDeepLink(new URLSearchParams('preset=vlm_low_conf')).preset).toBe(
        'vlm_low_conf',
      );
    });

    it('rejects a garbage preset value rather than passing it through as state', () => {
      expect(
        reviewDeepLink(new URLSearchParams('preset=not_a_real_preset')).preset,
      ).toBeNull();
    });

    it('ignores a preset on any tab other than all — presets are all-only', () => {
      expect(
        reviewDeepLink(new URLSearchParams('tab=uncertainty&preset=mismatches')).preset,
      ).toBeNull();
    });

    it('absent preset param is null, not undefined-that-happens-to-be-falsy', () => {
      expect(reviewDeepLink(new URLSearchParams('tab=all')).preset).toBeNull();
    });
  });
});

describe('the imported tab (W10): offered only when the backend serves it', () => {
  it('visibleReviewTabs drops imported unless the served vocabulary has it', () => {
    const none = visibleReviewTabs(REVIEW_TABS, () => false).map((t) => t.id);
    expect(none).not.toContain('imported');
    // Every other tab is untouched by the served-only rule.
    expect(none).toContain('new_class_proposals');
    expect(none).toHaveLength(REVIEW_TABS.length - 1);
    const served = visibleReviewTabs(REVIEW_TABS, (id) => id === 'imported').map(
      (t) => t.id,
    );
    expect(served).toContain('imported');
    expect(served).toHaveLength(REVIEW_TABS.length);
  });

  it('asks the served vocabulary by the tab endpoint id', () => {
    const asked: string[] = [];
    visibleReviewTabs(REVIEW_TABS, (id) => {
      asked.push(id);
      return true;
    });
    expect(asked).toEqual(['imported']);
  });

  it('resolves ?tab=imported to the imported tab and its backend endpoint', () => {
    expect(tabFromUrlId('imported')).toBe('imported');
    expect(endpointForTab('imported')).toBe('imported');
  });
});

describe('reviewDeepLink: URL-seeded non-enum filters', () => {
  it('reads import_id and combine_conflict=true', () => {
    const d = reviewDeepLink(
      new URLSearchParams('tab=imported&import_id=imp_1&combine_conflict=true'),
    );
    expect(d.importId).toBe('imp_1');
    expect(d.combineConflict).toBe(true);
  });

  it('treats an empty import_id and any other combine_conflict value as unset', () => {
    const d = reviewDeepLink(new URLSearchParams('import_id=&combine_conflict=false'));
    expect(d.importId).toBeNull();
    expect(d.combineConflict).toBe(false);
    expect(
      reviewDeepLink(new URLSearchParams('combine_conflict=1')).combineConflict,
    ).toBe(false);
  });
});

describe('unavailable ?tab= (coordinator minor, 2026-09-25)', () => {
  it('reviewDeepLink reports a tab that resolved to nothing, and nothing otherwise', () => {
    expect(reviewDeepLink(new URLSearchParams('tab=no_such_tab')).unavailableTab).toBe(
      'no_such_tab',
    );
    expect(reviewDeepLink(new URLSearchParams('tab=all')).unavailableTab).toBeNull();
    expect(reviewDeepLink(new URLSearchParams('')).unavailableTab).toBeNull();
  });

  it('names the missing region profile for the region tab id', () => {
    expect(unavailableTabMessage('regions', 'regions', false)).toContain(
      'the backend reports no region profile',
    );
    expect(unavailableTabMessage('bogus', 'regions', false)).toContain(
      'no "bogus" review tab',
    );
    expect(unavailableTabMessage('regions', 'regions', false, true)).toContain(
      "hasn't loaded",
    );
  });
});
