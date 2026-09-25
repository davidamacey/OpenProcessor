/**
 * Mount-based tests for the CropMetaPanel fixes from
 * docs/design/visual-audit-2026-09-24.md:
 *
 *  - R6: a class-less crop read "Class (vlm_new_class_pending)" with a blank
 *    name and the raw source id; `vlm_class_empty_reason` rendered raw.
 *  - R7: Details' region Status used the slot profile's label while the
 *    review panel's dropdown used the served `/regions/statuses` label —
 *    two vocabularies for one status.
 *  - R11: embedded under /review, the panel repeated the Class / Detector
 *    score / VLM confidence rows sitting directly above it.
 */
import {
  afterAll,
  afterEach,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import CropMetaPanel from './CropMetaPanel.svelte';
import { mapCropSlots } from '$lib/annotations/cropSlots';
import { regionStatusesStore } from '$stores/regionStatuses.svelte';
import {
  installServedRegionProfile,
  resetDeploymentSlots,
} from '$lib/annotations/registeredSlots';
import { WIDGET_TAG_PROFILE } from '$lib/test/fixtures/regionSlot';
import type { Crop } from '$lib/types';

let target: HTMLDivElement;
let instance: unknown;

beforeEach(() => {
  // History / source-image lookups fire on mount; a rejected fetch renders
  // their inline "unavailable" text, which these tests don't look at.
  vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('no network in test')));
});

afterEach(() => {
  if (instance) {
    unmount(instance);
    instance = undefined;
  }
  target?.remove();
  regionStatusesStore.list = [];
  vi.unstubAllGlobals();
});

function crop(over: Partial<Crop> = {}): Crop {
  return {
    id: 'crop-1',
    source_image_path: '/img.jpg',
    bbox_norm: { cx: 0.5, cy: 0.5, w: 0.5, h: 0.5 },
    class_id: null,
    class_name: null,
    cluster_id: 7,
    label_confidence: 0.655,
    label_source: 'vlm',
    class_source: 'vlm_new_class_pending',
    vlm_confidence: 'medium',
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

function rowValue(el: HTMLElement, label: string): string | null {
  const dt = [...el.querySelectorAll('dt')].find((d) => d.textContent?.trim() === label);
  return dt?.nextElementSibling?.textContent?.replace(/\s+/g, ' ').trim() ?? null;
}

describe('R6: no blank class, no raw ids', () => {
  it('a class-less crop reads "no class yet", not a blank', () => {
    const el = render({ crop: crop() });
    expect(rowValue(el, 'Class')).toMatch(/^no class yet/);
  });

  it('vlm_class_empty_reason renders as prose, not the raw id', () => {
    const el = render({ crop: crop({ vlm_class_empty_reason: 'no_answer' }) });
    expect(rowValue(el, 'VLM empty reason')).toBe('No answer');
  });
});

describe('R11: embedded Details drops rows the review panel already shows', () => {
  it('standalone (the /clusters modal) keeps Class, Label source, score and VLM confidence', () => {
    const el = render({ crop: crop({ class_name: 'sedan', class_id: 3 }) });
    expect(rowValue(el, 'Class')).toMatch(/^sedan/);
    expect(rowValue(el, 'Label source')).not.toBeNull();
    expect(rowValue(el, 'VLM confidence')).toBe('medium');
  });

  it('embedded (/review) hides them, and keeps the rest (Cluster)', () => {
    const el = render({
      crop: crop({ class_name: 'sedan', class_id: 3 }),
      embedded: true,
    });
    expect(rowValue(el, 'Class')).toBeNull();
    expect(rowValue(el, 'Label source')).toBeNull();
    expect(rowValue(el, 'Detector score')).toBeNull();
    expect(rowValue(el, 'Confidence')).toBeNull();
    expect(rowValue(el, 'VLM confidence')).toBeNull();
    expect(rowValue(el, 'Cluster')).toMatch(/^#7/);
  });
});

const REGION_RAW = {
  region_bbox_norm: [0.4, 0.4, 0.6, 0.6],
  region_status: 'detected',
  region_score: 0.9,
};

describe('R7: region Status uses the served status vocabulary', () => {
  beforeAll(() => installServedRegionProfile(WIDGET_TAG_PROFILE));
  afterAll(() => resetDeploymentSlots());

  it('renders the served /regions/statuses label, not the slot profile label', () => {
    regionStatusesStore.list = [
      {
        value: 'detected',
        label: 'served-label-for-detected',
        role: 'positive',
        terminal: true,
        human_writable: true,
        clears_box: false,
        wants_reason: false,
      },
    ] as never;
    const slots = mapCropSlots(REGION_RAW, [0, 0, 1, 1]);
    expect(Object.keys(slots).length).toBeGreaterThan(0);
    const el = render({ crop: crop({ slots } as Partial<Crop>) });
    expect(el.textContent).toContain(WIDGET_TAG_PROFILE.display_name);
    expect(rowValue(el, 'Status')).toBe('served-label-for-detected');
  });
});

describe('no region profile (domain-neutral audit §5.4)', () => {
  beforeAll(() => installServedRegionProfile(null));
  afterAll(() => resetDeploymentSlots());

  it('renders no region section, even for an item whose wire carries region_* values', () => {
    const slots = mapCropSlots(REGION_RAW, [0, 0, 1, 1]);
    expect(slots).toEqual({});
    const el = render({ crop: crop({ slots } as Partial<Crop>) });
    expect(rowValue(el, 'Status')).toBeNull();
    expect(el.textContent).not.toContain(WIDGET_TAG_PROFILE.display_name);
  });
});
