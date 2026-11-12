import { flushSync } from 'svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { mountUi } from '$lib/test/mountUi';
import { createTrainCohortsController } from '$lib/train/trainCohortsController.svelte';
import TrainingCohortsPanel from './TrainingCohortsPanel.svelte';

const SUMMARY = {
  cohorts: [
    {
      id: 'validated',
      label: 'Validated',
      description: 'Human-validated',
      cutoffs: {},
      endpoint: '/crops',
      params: { label_validated: true },
      row_kind: 'crop',
      total: 12,
    },
    {
      id: 'detector_proposals',
      label: 'Detector proposals not decided',
      description: 'Nobody has decided it',
      cutoffs: {},
      endpoint: '/crops',
      params: { review_tab: 'to_label' },
      row_kind: 'crop',
      total: 4,
    },
  ],
  rows: [
    {
      class_id: 1,
      class_name: 'car',
      kind: 'class',
      counts: { validated: 9, detector_proposals: 1 },
    },
    {
      class_id: 2,
      class_name: 'bus',
      kind: 'class',
      counts: { validated: 0, detector_proposals: 0 },
    },
    {
      class_id: null,
      class_name: 'tram',
      kind: 'detector',
      counts: { validated: 0, detector_proposals: 3 },
    },
  ],
};

let urls: string[] = [];
function serve(): void {
  urls = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      urls.push(String(url));
      const body = String(url).includes('/training_cohorts/summary')
        ? SUMMARY
        : { items: [], total: 0, next_cursor: null, limit: 24 };
      return new Response(JSON.stringify(body), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      });
    }),
  );
}
afterEach(() => vi.unstubAllGlobals());

async function render(): Promise<HTMLElement> {
  serve();
  const el = mountUi(TrainingCohortsPanel, { ctl: createTrainCohortsController() });
  await vi.waitFor(() => {
    flushSync();
    expect(el.querySelector('table')).not.toBeNull();
  });
  return el;
}
const classes = (el: HTMLElement) =>
  [...el.querySelectorAll('tbody tr td:first-child')].map((c) =>
    c.textContent?.replace('detector label', '').trim(),
  );

describe('TrainingCohortsPanel', () => {
  it('reads the matrix in one request, never one per class, and hides empty rows', async () => {
    const el = await render();
    expect(urls.filter((u) => u.includes('training_cohorts'))).toHaveLength(1);
    expect(classes(el)).toEqual(['car', 'tram']);
    expect(el.querySelector('[data-testid="cohorts-hidden"]')!.textContent).toContain(
      '1 class with',
    );
  });

  it('turning "Hide empty" off shows every row', async () => {
    const el = await render();
    const box = el.querySelector<HTMLInputElement>('[data-testid="cohorts-hide-empty"]')!;
    box.click();
    flushSync();
    expect(classes(el)).toEqual(['bus', 'car', 'tram']);
  });

  it('the filter keeps the classes that match', async () => {
    const el = await render();
    const input = el.querySelector<HTMLInputElement>('[data-testid="cohorts-filter"]')!;
    input.value = 'tra';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    expect(classes(el)).toEqual(['tram']);
  });

  it('a class count opens the preview in a Drawer; a detector label or the undecided column does not', async () => {
    const el = await render();
    const buttons = [...el.querySelectorAll('tbody button')];
    expect(buttons.map((b) => b.getAttribute('aria-label'))).toEqual([
      'Preview 9 Validated for car',
    ]);
    (buttons[0] as HTMLButtonElement).click();
    await vi.waitFor(() => {
      flushSync();
      expect(document.querySelector('[data-testid="drawer"]')).not.toBeNull();
    });
    expect(document.querySelector('[data-testid="cohort-preview"]')).not.toBeNull();
    expect(urls.some((u) => u.includes('/crops'))).toBe(true);
  });
});
