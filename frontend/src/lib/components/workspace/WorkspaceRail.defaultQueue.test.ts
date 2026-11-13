import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, tick, unmount } from 'svelte';

vi.mock('$lib/api_review', async () => {
  const actual =
    await vi.importActual<typeof import('$lib/api_review')>('$lib/api_review');
  return { ...actual, getReviewTabs: vi.fn() };
});

import WorkspaceRail from './WorkspaceRail.svelte';
import { reviewTabsVocabularyStore } from '$stores/reviewTabsVocabulary.svelte';
import { getReviewTabs, type ReviewTabVocabularyEntry } from '$lib/api_review';

const tab = (id: string, count: number): ReviewTabVocabularyEntry => ({
  id,
  label: id,
  description: '',
  filters: [],
  filter_defaults: {},
  filter_specs: [],
  count,
});

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | null = null;

async function open(queue: string | null, defaultTab: string | null) {
  vi.mocked(getReviewTabs).mockResolvedValue({
    tabs: [tab('to_label', 0), tab('check_machine', 7), tab('disagreements', 3)],
    empty_state: {
      has_probe_predictions: false,
      has_item_scores: false,
      has_unembedded_items: false,
      suggested_reprocess: null,
    },
    default_tab: defaultTab,
  });
  const ws = {
    url: { queue },
    itemQuery: () => ({}),
    railOpen: true,
    filter: { classNames: [] },
    setQueue: vi.fn(),
    toggleRail: vi.fn(),
    applyFilter: vi.fn(),
  };
  instance = mount(WorkspaceRail, { target, props: { ws } as never });
  await tick();
  await Promise.resolve();
  await tick();
  flushSync();
}

const button = (id: string) =>
  target.querySelector<HTMLElement>(`[data-testid="queue-${id}"]`)!;

beforeEach(() => {
  reviewTabsVocabularyStore.resetForProjectChange();
  target = document.createElement('div');
  document.body.appendChild(target);
});
afterEach(() => {
  if (instance) unmount(instance);
  instance = null;
  target.remove();
});

describe('the rail marks the served default queue', () => {
  it('shows the first queue with work in bold and no other', async () => {
    await open(null, 'check_machine');
    expect(button('check_machine').className).toContain('font-semibold');
    expect(button('check_machine').dataset.defaultQueue).toBe('true');
    expect(button('to_label').className).not.toContain('font-semibold');
    expect(button('disagreements').className).not.toContain('font-semibold');
  });

  it('marks nothing when no queue holds work', async () => {
    await open(null, null);
    expect(target.querySelectorAll('[data-default-queue]').length).toBe(0);
  });
});
