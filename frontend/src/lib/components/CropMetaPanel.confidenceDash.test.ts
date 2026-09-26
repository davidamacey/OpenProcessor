/**
 * OCR-sentinel follow-up (2026-09-26): the backend plans to serve a
 * failed OCR recognition as `confidence: null` (plus a separate failure
 * reason), never a negative sentinel value. This pins that a null
 * confidence on either surface -- the region-slot text reading
 * (`region_text_confidence`) and a per-item OCR line
 * (`item_text_lines[].confidence`) -- renders as "—", never "0%" or
 * "null%".
 */
import {
  afterEach,
  beforeAll,
  afterAll,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import CropMetaPanel from './CropMetaPanel.svelte';
import {
  installServedRegionProfile,
  resetDeploymentSlots,
} from '$lib/annotations/registeredSlots';
import { mapCropSlots } from '$lib/annotations/cropSlots';
import { WIDGET_TAG_PROFILE } from '$lib/test/fixtures/regionSlot';
import type { Crop } from '$lib/types';

let target: HTMLDivElement;
let instance: unknown;

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('no network in test')));
});

afterEach(() => {
  if (instance) {
    unmount(instance);
    instance = undefined;
  }
  target?.remove();
  vi.unstubAllGlobals();
});

function crop(over: Partial<Crop> = {}): Crop {
  return {
    id: 'crop-1',
    source_image_path: '/img.jpg',
    bbox_norm: { cx: 0.5, cy: 0.5, w: 0.5, h: 0.5 },
    class_id: 3,
    class_name: 'widget_a',
    cluster_id: 7,
    label_confidence: 0.9,
    label_source: 'model',
    class_source: 'model',
    updated_at: '',
    ...over,
  } as Crop;
}

function render(props: Record<string, unknown>): HTMLDivElement {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(CropMetaPanel, { target, props } as never);
  flushSync();
  return target;
}

describe('region-slot text confidence: null renders as "—"', () => {
  beforeAll(() => installServedRegionProfile(WIDGET_TAG_PROFILE));
  afterAll(() => resetDeploymentSlots());

  function withRegion(over: Record<string, unknown>): Crop {
    const raw = {
      region_bbox_norm: [0.4, 0.4, 0.6, 0.6],
      region_status: 'detected',
      region_score: 0.9,
      region_text: 'TAG-001',
      ...over,
    };
    return crop({ slots: mapCropSlots(raw, [0, 0, 1, 1]) } as never);
  }

  it('null region_text_confidence shows "—", never a percentage', () => {
    const el = render({ crop: withRegion({ region_text_confidence: null }) });
    const text = el.textContent ?? '';
    expect(text).toContain('TAG-001');
    expect(text).toContain('—');
    expect(text).not.toMatch(/NaN%|null%/i);
  });

  it('a real confidence still renders as a percentage', () => {
    const el = render({ crop: withRegion({ region_text_confidence: 0.92 }) });
    expect(el.textContent ?? '').toContain('92.0%');
  });
});

describe('item_text_lines confidence: null renders as "—"', () => {
  it('a line with confidence: null shows "—", not "0%"/"NaN%"', () => {
    const el = render({
      crop: crop({
        item_text_lines: [
          { text: 'PART-42', confidence: null, box_norm: [0, 0, 1, 1], rel_height: 0.1 },
        ],
      } as never),
    });
    const text = el.textContent ?? '';
    expect(text).toContain('PART-42');
    // The item-text-line confidence chip sits right after its text span.
    expect(text).toMatch(/PART-42\s*—/);
    expect(text).not.toMatch(/NaN%|null%/i);
  });

  it('a line with a real confidence still renders as a percentage', () => {
    const el = render({
      crop: crop({
        item_text_lines: [
          { text: 'PART-42', confidence: 0.81, box_norm: [0, 0, 1, 1], rel_height: 0.1 },
        ],
      } as never),
    });
    expect(el.textContent ?? '').toContain('81.0%');
  });
});
