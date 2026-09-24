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
import { widgetTagSlot } from '$lib/test/fixtures/regionSlot';

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
    class_name: 'sedan',
    cluster_id: null,
    updated_at: '',
    ...overrides,
  } as RegionBrowseItem;
}

let target: HTMLDivElement;
let instance: unknown;

function renderCard(crop: RegionBrowseItem) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(SlotCard, { target, props: { crop, slot: widgetTagSlot } });
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
  it('renders the ⚠ disagree chip when region_text_disagreement is true', () => {
    const el = renderCard(fakeRegionItem({ region_text_disagreement: true } as never));
    expect(el.textContent).toContain('⚠ disagree');
  });

  it('does not render the chip when region_text_disagreement is false', () => {
    const el = renderCard(fakeRegionItem({ region_text_disagreement: false } as never));
    expect(el.textContent).not.toContain('⚠ disagree');
  });

  it('does not render the chip when region_text_disagreement is absent', () => {
    const el = renderCard(fakeRegionItem());
    expect(el.textContent).not.toContain('⚠ disagree');
  });
});
