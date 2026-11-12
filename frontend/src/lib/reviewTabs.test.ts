import { afterEach, describe, expect, it } from 'vitest';
import { WIDGET_TAG_PROFILE, widgetTagSlot } from '$lib/test/fixtures/regionSlot';
import { registeredSlots, resetDeploymentSlots } from './annotations/registeredSlots';
import {
  buildReviewTabs,
  endpointForTab,
  isSlotTab,
  REVIEW_TABS,
  slotTabId,
  tabForQueue,
} from './reviewTabs';

// The registered queue slots (whatever this build registers); each gets exactly one tab.
const queueSlots = registeredSlots.filter((s) => s.capabilities.queue);

afterEach(() => resetDeploymentSlots());

describe('REVIEW_TABS', () => {
  it('is one slot:<key> tab per queue slot and nothing else: the plain queues are served', () => {
    expect(REVIEW_TABS.map((t) => t.id)).toEqual(queueSlots.map((s) => `slot:${s.key}`));
    for (const s of queueSlots) {
      expect(REVIEW_TABS.find((t) => t.id === `slot:${s.key}`)?.urlId).toBe(
        s.capabilities.queue!.urlId,
      );
    }
  });

  it('no region profile: no tabs at all', async () => {
    const reg = await import('./annotations/registeredSlots');
    const tabs = await import('./reviewTabs');
    reg.installServedRegionProfile(null);
    expect(tabs.REVIEW_TABS).toEqual([]);
    expect(tabs.tabForQueue('regions')).toBe('regions');
  });

  it('a served region profile adds one region tab labelled by its display_name', async () => {
    const reg = await import('./annotations/registeredSlots');
    const tabs = await import('./reviewTabs');
    reg.installServedRegionProfile(WIDGET_TAG_PROFILE);
    expect(tabs.REVIEW_TABS).toHaveLength(1);
    expect(tabs.REVIEW_TABS.at(-1)).toMatchObject({
      id: `slot:${WIDGET_TAG_PROFILE.name}`,
      label: WIDGET_TAG_PROFILE.display_name,
      urlId: 'regions',
      endpointId: 'regions',
    });
    expect(tabs.tabForQueue('regions')).toBe(`slot:${WIDGET_TAG_PROFILE.name}`);
  });
});

describe('buildReviewTabs (P2.8 data-driving)', () => {
  it('derives a tab from a queue-capable slot, matching its urlId/endpointId/label', () => {
    const [tab] = buildReviewTabs([widgetTagSlot]);
    const q = widgetTagSlot.capabilities.queue!;
    expect(tab).toMatchObject({
      id: slotTabId(widgetTagSlot.key),
      label: q.tabLabel,
      urlId: q.urlId,
      endpointId: q.endpointId,
      slot: widgetTagSlot,
    });
  });

  it('skips a slot with no queue capability', () => {
    const noQueue = {
      ...widgetTagSlot,
      capabilities: { ...widgetTagSlot.capabilities, queue: undefined },
    };
    expect(buildReviewTabs([noQueue])).toEqual([]);
  });
});

describe('isSlotTab', () => {
  it('is structurally true for any slot: id, even an unregistered one', () => {
    expect(isSlotTab('slot:not_registered')).toBe(true);
  });

  it('is false for a served queue id', () => {
    for (const id of ['to_label', 'check_machine', 'disagreements', 'audit']) {
      expect(isSlotTab(id)).toBe(false);
    }
  });
});

describe('tabForQueue and endpointForTab', () => {
  it('a served queue no slot reads is its own tab and endpoint', () => {
    expect(tabForQueue('to_label')).toBe('to_label');
    expect(endpointForTab('to_label')).toBe('to_label');
  });

  it('no queue (walking /crops) has no tab', () => {
    expect(tabForQueue(null)).toBe('');
  });

  it("each registered slot's queue resolves to its slot tab and back to its endpoint", () => {
    for (const s of queueSlots) {
      const tab = tabForQueue(s.capabilities.queue!.endpointId);
      expect(tab).toBe(`slot:${s.key}`);
      expect(endpointForTab(tab)).toBe(s.capabilities.queue!.endpointId);
    }
  });
});
