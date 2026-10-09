/**
 * DetectorClass, ConfirmationChip and ValidatedRatioNotice (#119): mounted
 * against the served role catalog and served counts.
 */
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { classSourcesStore } from '$stores/classSources.svelte';
import ConfirmationChip from './ConfirmationChip.svelte';
import DetectorClass from './DetectorClass.svelte';
import ValidatedRatioNotice from './ValidatedRatioNotice.svelte';

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | undefined;

function render(component: unknown, props: Record<string, unknown>) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(component as never, { target, props } as never);
  flushSync();
  return target;
}

beforeEach(() => {
  classSourcesStore.list = [
    { id: 'vlm_write', label: 'VLM', role: 'vlm', short_label: 'VLM' },
    { id: 'human_label', label: 'Human', role: 'human', short_label: 'H' },
    {
      id: 'cluster_agree',
      label: 'Cluster agreement',
      role: 'cluster',
      short_label: 'C',
    },
    { id: 'det_model', label: 'Detector', role: 'model', short_label: 'M' },
  ] as never;
});

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

describe('DetectorClass', () => {
  it('shows the detector class and score on a VLM-sourced label', () => {
    const el = render(DetectorClass, {
      crop: {
        class_source: 'vlm_write',
        detector_class_name: 'widget_h',
        detector_confidence: 0.72,
      },
    });
    expect(el.querySelector('[data-testid="detector-class"]')?.textContent).toContain(
      'Detector:\u00a0widget_h 72%',
    );
  });

  it('shows the value alone when bare', () => {
    const el = render(DetectorClass, {
      bare: true,
      crop: {
        class_source: 'vlm_write',
        detector_class_name: 'widget_h',
        detector_confidence: 0.72,
      },
    });
    const text = el.querySelector('[data-testid="detector-class"]')?.textContent?.trim();
    expect(text).toBe('widget_h 72%');
  });

  it('renders nothing for a human-sourced label', () => {
    const el = render(DetectorClass, {
      crop: {
        class_source: 'human_label',
        detector_class_name: 'widget_h',
        detector_confidence: 0.72,
      },
    });
    expect(el.querySelector('[data-testid="detector-class"]')).toBeNull();
  });

  it('renders nothing when the item carries no detector class', () => {
    const el = render(DetectorClass, {
      crop: {
        class_source: 'vlm_write',
        detector_class_name: null,
        detector_confidence: null,
      },
    });
    expect(el.querySelector('[data-testid="detector-class"]')).toBeNull();
  });
});

describe('ConfirmationChip', () => {
  const chip = (el: HTMLElement) =>
    el.querySelector('[data-testid="confirmation-chip"]')?.textContent?.trim() ?? null;

  it('says "VLM suggestion" for an unvalidated VLM label', () => {
    expect(
      chip(render(ConfirmationChip, { source: 'vlm_write', validated: false })),
    ).toBe('VLM suggestion');
  });

  it('says "Human-confirmed" for a human label', () => {
    expect(
      chip(render(ConfirmationChip, { source: 'human_label', validated: true })),
    ).toBe('Human-confirmed');
  });

  it('says "Auto-validated" for a validated cluster label only', () => {
    expect(
      chip(render(ConfirmationChip, { source: 'cluster_agree', validated: true })),
    ).toBe('Auto-validated');
    unmount(instance!);
    target.remove();
    expect(
      chip(render(ConfirmationChip, { source: 'cluster_agree', validated: false })),
    ).toBeNull();
  });

  it('renders nothing for a model label or an unknown source', () => {
    expect(
      chip(render(ConfirmationChip, { source: 'det_model', validated: true })),
    ).toBeNull();
    unmount(instance!);
    target.remove();
    expect(
      chip(render(ConfirmationChip, { source: 'mystery', validated: false })),
    ).toBeNull();
  });
});

describe('ValidatedRatioNotice', () => {
  const root = (el: HTMLElement) =>
    el.querySelector('[data-testid="validated-ratio"]') as HTMLElement;

  it('warns that only validated crops are exported when some are not', () => {
    const el = render(ValidatedRatioNotice, { validated: 3000, total: 14000 });
    expect(root(el).dataset.level).toBe('partial');
    expect(el.querySelector('[data-testid="validated-ratio-counts"]')?.textContent).toBe(
      `${(3000).toLocaleString()} of ${(14000).toLocaleString()}`,
    );
    expect(root(el).textContent).toContain('Only validated crops are exported');
  });

  it('says nothing is exportable when no crop is validated', () => {
    const el = render(ValidatedRatioNotice, { validated: 0, total: 14000 });
    expect(root(el).dataset.level).toBe('none');
    expect(root(el).textContent).toContain('Nothing is exportable yet');
  });

  it('is calm when every crop is validated', () => {
    const el = render(ValidatedRatioNotice, { validated: 50, total: 50 });
    expect(root(el).dataset.level).toBe('full');
    expect(root(el).textContent).toContain('Every crop is validated');
  });
});
