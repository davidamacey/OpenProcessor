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
      class_name: 'cog',
      kind: 'class',
      counts: { validated: 9, detector_proposals: 1 },
    },
    {
      class_id: 2,
      class_name: 'bolt',
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
    expect(classes(el)).toEqual(['tram', 'cog']);
    expect(el.querySelector('[data-testid="cohorts-hidden"]')!.textContent).toContain(
      '1 class with',
    );
  });

  it('turning "Hide empty" off shows every row', async () => {
    const el = await render();
    const box = el.querySelector<HTMLInputElement>('[data-testid="cohorts-hide-empty"]')!;
    box.click();
    flushSync();
    expect(classes(el)).toEqual(['tram', 'cog', 'bolt']);
  });

  it('the filter keeps the classes that match', async () => {
    const el = await render();
    const input = el.querySelector<HTMLInputElement>('[data-testid="cohorts-filter"]')!;
    input.value = 'tra';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    expect(classes(el)).toEqual(['tram']);
  });

  it('a count opens the preview in a Drawer; a zero has no link', async () => {
    const el = await render();
    const buttons = [...el.querySelectorAll('tbody button')];
    expect(buttons.map((b) => b.getAttribute('aria-label'))).toEqual([
      'Preview 3 Detector proposals not decided for tram',
      'Preview 1 Detector proposals not decided for cog',
      'Preview 9 Validated for cog',
    ]);
    (buttons.at(-1) as HTMLButtonElement).click();
    await vi.waitFor(() => {
      flushSync();
      expect(document.querySelector('[data-testid="drawer"]')).not.toBeNull();
    });
    expect(document.querySelector('[data-testid="cohort-preview"]')).not.toBeNull();
    expect(urls.some((u) => u.includes('/crops'))).toBe(true);
  });

  it('a detector-only cell previews by the detector class name and the to_label queue, not a class', async () => {
    const el = await render();
    const btn = [...el.querySelectorAll('tbody button')].find((b) =>
      b.getAttribute('aria-label')?.includes('for tram'),
    ) as HTMLButtonElement;
    expect(btn).toBeDefined();
    btn.click();
    await vi.waitFor(() =>
      expect(urls.some((u) => u.includes('review_tab=to_label'))).toBe(true),
    );
    const u = urls.find((x) => x.includes('review_tab=to_label'))!;
    expect(u).toContain('detector_class_name=tram');
    expect(new URL(u, 'http://x').searchParams.has('class_name')).toBe(false);
  });

  it('opens sorted by what is still detector-only, largest first, and that column comes right after Class', async () => {
    const el = await render();
    const heads = [...el.querySelectorAll('thead th')].map((h) => h.textContent?.trim());
    expect(heads[1]).toContain('Detector proposals not decided');
    expect(classes(el)).toEqual(['tram', 'cog']);
  });
});
