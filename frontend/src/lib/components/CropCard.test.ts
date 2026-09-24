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
  it('renders the class-source catalog label, not the raw source id', () => {
    classSourcesStore.list = [
      { id: 'model_x', label: 'Model X Detector', role: 'model' },
    ] as never;
    const el = renderCard({
      crop: baseCrop({ label_source: 'model_x', class_validated: true }),
    });

    const badge = el.querySelector('span[title="sedan"]');
    expect(badge?.textContent?.trim()).toBe('Model X Detector');
  });

  it('follows class_validated, not label_validated, for the vlm badge suffix', () => {
    classSourcesStore.list = [{ id: 'vlm_write', label: 'VLM', role: 'vlm' }] as never;

    // class_validated=false, label_validated=true (region validated, class
    // is not) — the badge must still render the unvalidated form. p4
    // (2026-09-24 interactive pass): no longer a literal "?" suffix — the
    // unvalidated state now lives in the title tooltip instead.
    const el = renderCard({
      crop: baseCrop({
        label_source: 'vlm_write',
        class_validated: false,
        label_validated: true,
      }),
    });
    const badge = el.querySelector('span[title="sedan — not yet validated"]');
    expect(badge?.textContent?.trim()).toBe('VLM ·');
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
    const badge = el.querySelector('span[title="sedan"]');
    expect(badge?.textContent?.trim()).toBe('VLM');
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
