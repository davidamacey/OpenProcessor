/**
 * Mount-based behavior tests for CropCard (docs/design/test-audit-2026-09-24.md
 * P1-4/2.4). Previously every "component test" in this directory was a
 * regex scan over the .svelte source text — it passed as long as a line of
 * code existed, not whether it ran correctly. These mount the real
 * component (Svelte 5 `mount`/`unmount`/`flushSync` under jsdom, enabled by
 * vite.config.ts's `resolve.conditions: ['browser']`) and assert on the
 * rendered DOM.
 */
import { afterEach, describe, expect, it } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import CropCard from './CropCard.svelte';
import { classSourcesStore } from '$stores/classSources.svelte';
import type { Crop } from '$lib/types';
import {
  installServedRegionProfile,
  resetDeploymentSlots,
} from '$lib/annotations/registeredSlots';
import { mapCropSlots } from '$lib/annotations/cropSlots';
import { WIDGET_TAG_PROFILE } from '$lib/test/fixtures/regionSlot';

function baseCrop(overrides: Partial<Crop> = {}): Crop {
  return {
    id: 'c1',
    source_image_path: '/img.jpg',
    bbox_norm: { cx: 0.5, cy: 0.5, w: 0.4, h: 0.4 },
    class_id: 3,
    class_name: 'sedan',
    class_source: null,
    label_source: 'model_x',
    label_validated: false,
    class_validated: false,
    label_confidence: null,
    cluster_id: null,
    similarity_to_centroid: null,
    cluster_subid: null,
    proposed_class_id: null,
    proposed_class_name: null,
    test_holdout: false,
    updated_at: '',
    ...overrides,
  } as Crop;
}

let target: HTMLDivElement;
let instance: unknown;

function renderCard(props: Record<string, unknown>) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(CropCard, { target, props } as never);
  flushSync();
  return target;
}

afterEach(() => {
  if (instance) {
    unmount(instance);
    instance = undefined;
  }
  target?.remove();
  classSourcesStore.list = [];
});

describe('CropCard — source badge', () => {
  const chip = (el: HTMLElement) => el.querySelector('[data-testid="source-chip"]');

  it('shows a short role code, with the served catalog label in the tooltip (K2)', () => {
    classSourcesStore.list = [
      { id: 'model_x', label: 'Model X Detector', role: 'model' },
    ] as never;
    const el = renderCard({
      crop: baseCrop({ label_source: 'model_x', class_validated: true }),
    });

    expect(chip(el)?.textContent?.trim()).toBe('M');
    expect(chip(el)?.getAttribute('title')).toBe('Label source: Model X Detector');
  });

  it('gives the class name its own element, never truncated by the source label (K2)', () => {
    classSourcesStore.list = [
      { id: 'vlm_write', label: 'Labeled by the VLM', role: 'vlm' },
    ] as never;
    const el = renderCard({
      crop: baseCrop({ label_source: 'vlm_write', class_name: 'class_b' }),
    });
    expect(el.querySelector('[data-testid="class-name"]')?.textContent?.trim()).toBe(
      'class_b',
    );
    expect(chip(el)?.textContent).not.toContain('Labeled by');
  });

  it('follows class_validated, not label_validated, for the vlm badge suffix', () => {
    classSourcesStore.list = [{ id: 'vlm_write', label: 'VLM', role: 'vlm' }] as never;

    // class_validated=false, label_validated=true (region validated, class
    // is not) — the badge must still render the unvalidated form.
    const el = renderCard({
      crop: baseCrop({
        label_source: 'vlm_write',
        class_validated: false,
        label_validated: true,
      }),
    });
    expect(chip(el)?.getAttribute('title')).toBe('Label source: VLM — not yet validated');
    expect(chip(el)?.textContent?.trim()).toBe('VLM·');
  });

  it('renders the validated form when class_validated is true regardless of label_validated', () => {
    classSourcesStore.list = [{ id: 'vlm_write', label: 'VLM', role: 'vlm' }] as never;

    const el = renderCard({
      crop: baseCrop({
        label_source: 'vlm_write',
        class_validated: true,
        label_validated: false,
      }),
    });
    expect(chip(el)?.getAttribute('title')).toBe('Label source: VLM');
    expect(chip(el)?.textContent?.trim()).toBe('VLM');
  });

  it('K3: a crop with no class reads Unlabeled with no "labeled by" chip, even with a label_source', () => {
    classSourcesStore.list = [
      { id: 'vlm', label: 'Labeled by the VLM', role: 'vlm' },
    ] as never;
    const el = renderCard({
      crop: baseCrop({
        class_id: null,
        class_name: null,
        label_source: 'vlm',
        class_confidence: 0.7,
        vlm_raw_class: 'car',
      } as Partial<Crop>),
    });
    expect(chip(el)).toBeNull();
    expect(el.textContent).not.toContain('Labeled by');
    expect(el.querySelector('[data-testid="class-name"]')?.textContent?.trim()).toBe(
      'Unlabeled',
    );
    expect(el.textContent).toContain('VLM said: car');
    expect(el.textContent).not.toContain('70%');
  });
});

describe('CropCard — VLM suggestion chip', () => {
  it('shows the suggestion chip when the crop is unvalidated', () => {
    const el = renderCard({
      crop: baseCrop({
        class_validated: false,
        vlm_suggested_class_id: 9,
        vlm_suggested_class_name: 'hatchback',
      }),
    });
    expect(el.textContent).toContain('VLM: hatchback');
  });

  it('hides the suggestion chip once the crop is validated', () => {
    const el = renderCard({
      crop: baseCrop({
        class_validated: true,
        vlm_suggested_class_id: 9,
        vlm_suggested_class_name: 'hatchback',
      }),
    });
    expect(el.textContent).not.toContain('VLM: hatchback');
  });
});

describe('CropCard — dq-queues cutover: class_confidence / vlm_raw_class', () => {
  it('renders class_confidence as a percentage next to the class name', () => {
    const el = renderCard({
      crop: baseCrop({ class_confidence: 0.92, class_confidence_source: 'vlm' }),
    });
    expect(el.textContent).toContain('92%');
  });

  it('omits the class_confidence chip when null (e.g. a human label)', () => {
    const el = renderCard({
      crop: baseCrop({ class_confidence: null }),
    });
    expect(el.textContent).not.toMatch(/\d+%/);
  });

  it('shows "VLM said: <raw>" when vlm_raw_class differs from the applied class', () => {
    const el = renderCard({
      crop: baseCrop({ class_name: 'sedan', vlm_raw_class: 'coupe' }),
    });
    expect(el.textContent).toContain('VLM said: coupe');
  });

  it('shows the VLM empty reason instead of "VLM said" when set', () => {
    const el = renderCard({
      crop: baseCrop({
        class_name: 'sedan',
        vlm_raw_class: null,
        vlm_class_empty_reason: 'no_match',
      }),
    });
    expect(el.textContent).toContain('VLM answer matched no class');
    expect(el.textContent).not.toContain('no_match');
    expect(el.textContent).not.toContain('VLM said');
  });
});

describe('CropCard — K4: readable VLM empty reason', () => {
  it('maps no_answer to a sentence, not the raw id', () => {
    const el = renderCard({
      crop: baseCrop({ vlm_raw_class: null, vlm_class_empty_reason: 'no_answer' }),
    });
    expect(el.textContent).toContain('VLM gave no answer');
    expect(el.textContent).not.toContain('no_answer');
  });
});

describe('CropCard — region sub-box editing follows the served region profile (audit §5.4)', () => {
  const editButton = (el: HTMLElement) => el.querySelector('button[aria-label^="Edit "]');

  afterEach(() => resetDeploymentSlots());

  it('no region profile: no ✎ button and no region ring, even with region_* values on the item', () => {
    installServedRegionProfile(null);
    const slots = mapCropSlots({ region_bbox_norm: [0.4, 0.4, 0.6, 0.6] }, [0, 0, 1, 1]);
    const el = renderCard({ crop: baseCrop({ slots } as Partial<Crop>) });
    expect(editButton(el)).toBeNull();
    expect(el.textContent).not.toContain('✎');
  });

  it('a served region profile: the ✎ button edits that region, labelled by the served singular noun (#36 item 10)', () => {
    installServedRegionProfile(WIDGET_TAG_PROFILE);
    const el = renderCard({ crop: baseCrop() });
    expect(editButton(el)?.getAttribute('aria-label')).toBe(
      `Edit ${WIDGET_TAG_PROFILE.display_name_singular.toLowerCase()}`,
    );
  });
});
