/**
 * Mount-based behavior test for SlotCard (docs/design/test-audit-2026-09-24.md
 * P1-4): the "readers disagree" flag must render exactly when
 * `region_text_disagreement` (readSlot's `disagreementField`) is true, and
 * must never render when it's false/absent — asserted on the rendered DOM,
 * not a source scan.
 */
import { afterEach, describe, expect, it } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import SlotCard from './SlotCard.svelte';
import type { RegionBrowseItem } from '$lib/api';
import { regionSlotFromServedProfile } from '$lib/annotations/servedRegionSlot';
import { widgetTagSlot, WIDGET_TAG_PROFILE_NO_TEXT } from '$lib/test/fixtures/regionSlot';

function fakeRegionItem(overrides: Partial<RegionBrowseItem> = {}): RegionBrowseItem {
  return {
    crop_id: 'c1',
    id: 'c1',
    image_path: '/img.jpg',
    bbox_norm: [0.3, 0.3, 0.7, 0.7],
    region_bbox_norm: [0.4, 0.4, 0.6, 0.6],
    region_score: 0.9,
    region_status: 'detected',
    region_verified: true,
    region_validated: true,
    region_detector: 'tag_detector_v1',
    region_detector_version: null,
    region_detector_chain: null,
    region_bbox_frame: 'source',
    region_detected_at: null,
    region_verifier: null,
    region_verifier_version: null,
    region_verified_at: null,
    region_rejection_reason: null,
    region_visible: true,
    region_text: 'TAG-001',
    region_text_source: 'ocr',
    region_text_confidence: 0.8,
    class_id: 3,
    class_name: 'widget_a',
    cluster_id: null,
    updated_at: '',
    ...overrides,
  } as RegionBrowseItem;
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
  it('renders the readers-disagree chip when region_text_disagreement is true', () => {
    const el = renderCard(fakeRegionItem({ region_text_disagreement: true } as never));
    expect(el.textContent).toContain('readers disagree');
    // C6 (visual audit 2026-09-24): plain text, no emoji warning glyph.
    expect(el.textContent).not.toContain('⚠');
  });

  it('does not render the chip when region_text_disagreement is false', () => {
    const el = renderCard(fakeRegionItem({ region_text_disagreement: false } as never));
    expect(el.textContent).not.toContain('readers disagree');
  });

  it('does not render the chip when region_text_disagreement is absent', () => {
    const el = renderCard(fakeRegionItem());
    expect(el.textContent).not.toContain('readers disagree');
  });
});

describe('SlotCard — text-free region profile (OpenProcessor W1)', () => {
  it('renders no text value row at all for a slot with no text capability', () => {
    const noTextSlot = regionSlotFromServedProfile(WIDGET_TAG_PROFILE_NO_TEXT);
    expect(noTextSlot.capabilities.text).toBeUndefined();
    const el = renderCard(fakeRegionItem({ region_text: null } as never), noTextSlot);
    expect(el.querySelector('[data-testid="slot-text-value"]')).toBeNull();
  });

  it('still renders the text value row (even with no reading yet) for a text-reading slot', () => {
    const el = renderCard(fakeRegionItem({ region_text: null } as never));
    expect(el.querySelector('[data-testid="slot-text-value"]')).not.toBeNull();
  });
});
