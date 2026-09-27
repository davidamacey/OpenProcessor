/**
 * Mount tests for the /classes fixes from docs/design/visual-audit-2026-09-24.md:
 *
 *  - L1: "Total (in cluster)" showed `sample_count` (widget_a 271) though its
 *    header promised the cluster page's "in cluster" number (270).
 *  - L2: flagged terms (widget_c 156, ...) sat in a collapsed section
 *    with no action at all, below ~80 tiny create-able terms.
 *  - L5: Validated includes frozen test crops with no hint.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import ClassesPage from './+page.svelte';
import { classesStore } from '$stores/classes.svelte';
import { proposalRows } from '$lib/classes/proposalRows';
import type { NewClassProposalsSummary } from '$lib/api';

const CLASSES = {
  classes: [
    {
      class_id: 5,
      class_name: 'widget_a',
      group: 'cars',
      sample_count: 271,
      validated_count: 35,
      cluster_size: 270,
      deprecated: false,
      hotkey_letter: null,
      adequacy: 'warn',
      added_at: '2026-04-29',
    },
    {
      class_id: 6,
      class_name: 'widget_b',
      group: 'cars',
      sample_count: 12,
      validated_count: 0,
      cluster_size: 12,
      deprecated: false,
      hotkey_letter: null,
      adequacy: 'block',
      added_at: '2026-04-29',
    },
  ],
};

const SUMMARY = {
  total_pending: 170,
  without_term: 0,
  top_terms: [
    {
      label: 'classic_widget',
      count: 3,
      sample_crop_ids: [],
      flag: null,
      class_id: null,
    },
  ],
  flagged_terms: [
    {
      label: 'widget_c',
      count: 156,
      sample_crop_ids: [],
      flag: 'generic_parent',
      class_id: null,
    },
    {
      label: 'widget_bs',
      count: 9,
      sample_crop_ids: [],
      flag: 'existing_class',
      class_id: 6,
    },
  ],
  term_rules: null,
};

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });
}

let target: HTMLDivElement;
let instance: unknown;

afterEach(() => {
  if (instance) {
    unmount(instance as never);
    instance = undefined;
  }
  target?.remove();
  classesStore.classes = [];
  vi.unstubAllGlobals();
});

async function flush(): Promise<void> {
  for (let i = 0; i < 10; i++) await new Promise((r) => setTimeout(r, 0));
  flushSync();
}

async function render(): Promise<HTMLDivElement> {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockImplementation((url: string) => {
      if (url.includes('/new_class_proposals/summary'))
        return Promise.resolve(jsonResponse(SUMMARY));
      if (url.includes('/test_holdout/stats'))
        return Promise.resolve(
          jsonResponse({ total: 5, by_class: [{ key: 5, doc_count: 5 }] }),
        );
      if (url.includes('/classes')) return Promise.resolve(jsonResponse(CLASSES));
      return Promise.resolve(jsonResponse({}));
    }),
  );
  await classesStore.refresh();
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(ClassesPage, { target } as never);
  flushSync();
  await flush();
  return target;
}

describe('L1: Total (in cluster) is cluster_size', () => {
  it('shows 270 (cluster_size), not 271 (sample_count)', async () => {
    const el = await render();
    const cell = el.querySelector(
      '[data-testid="class-row-5"] [data-testid="in-cluster"]',
    );
    expect(cell?.textContent?.trim()).toBe('270');
  });
});

describe('L5: Validated says how many are frozen test crops', () => {
  it('adds "incl. 5 test" for a class with held-out crops, nothing otherwise', async () => {
    const el = await render();
    const widget_a = el.querySelector('[data-testid="class-row-5"]');
    expect(
      widget_a?.querySelector('[data-testid="validated-test-suffix"]')?.textContent,
    ).toBe('incl. 5 test');
    const widget_b = el.querySelector('[data-testid="class-row-6"]');
    expect(widget_b?.querySelector('[data-testid="validated-test-suffix"]')).toBeNull();
  });
});

describe('L2: every proposal term is actionable, biggest first', () => {
  it('lists flagged and un-flagged terms in one list, sorted by count', async () => {
    const el = await render();
    const labels = [...el.querySelectorAll('[data-testid="proposal-row"]')].map((r) =>
      r.querySelector('.text-sm')?.textContent?.trim(),
    );
    expect(labels).toEqual(['widget_c', 'widget_bs', 'classic_widget']);
  });

  it('a generic-parent term can be mapped to an existing class but not created', async () => {
    const el = await render();
    const row = [...el.querySelectorAll('[data-testid="proposal-row"]')].find((r) =>
      r.textContent?.includes('widget_c'),
    );
    expect(row?.querySelector('select[aria-label^="Map widget_c"]')).not.toBeNull();
    expect(row?.textContent).not.toContain('Create class & assign');
    expect(row?.textContent).toContain('generic parent');
  });

  it('an existing-class term keeps its one-click map to the served class', async () => {
    const el = await render();
    const row = [...el.querySelectorAll('[data-testid="proposal-row"]')].find((r) =>
      r.textContent?.includes('widget_bs'),
    );
    expect(row?.textContent).toContain('Map to widget_b');
    expect(row?.querySelector('select')).toBeNull();
  });

  it('proposalRows only lets an un-flagged term be created', () => {
    const rows = proposalRows(SUMMARY as NewClassProposalsSummary);
    expect(rows.map((r) => [r.term.label, r.canCreate])).toEqual([
      ['widget_c', false],
      ['widget_bs', false],
      ['classic_widget', true],
    ]);
  });
});
