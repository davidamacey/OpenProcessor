import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

vi.mock('$lib/api_review', async () => {
  const actual =
    await vi.importActual<typeof import('$lib/api_review')>('$lib/api_review');
  return { ...actual, getReviewTabs: vi.fn() };
});

import WorkspaceRail from './WorkspaceRail.svelte';
import { reviewTabsVocabularyStore } from '$stores/reviewTabsVocabulary.svelte';
import { classesStore } from '$stores/classes.svelte';
import type { RegistryClass } from '$lib/types_classes';
import { getReviewTabs, type ReviewTabVocabularyEntry } from '$lib/api_review';

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | null = null;
const prevClasses = classesStore.classes;

function tab(id: string, label: string, count: number): ReviewTabVocabularyEntry {
  return {
    id,
    label,
    description: '',
    filters: [],
    filter_defaults: {},
    filter_specs: [],
    count,
  };
}

const TABS = [tab('to_label', 'To label', 1234), tab('all', 'All', 5)];
const EMPTY_STATE = {
  has_probe_predictions: false,
  has_item_scores: false,
  has_unembedded_items: false,
  suggested_reprocess: null,
};

beforeEach(() => {
  reviewTabsVocabularyStore.resetForProjectChange();
  vi.mocked(getReviewTabs).mockReset();
  vi.mocked(getReviewTabs).mockResolvedValue({ tabs: TABS, empty_state: EMPTY_STATE });
  reviewTabsVocabularyStore.list = TABS;
  reviewTabsVocabularyStore.loaded = true;
  classesStore.classes = [
    { id: 2, name: 'gadget', count: 7, deprecated: false },
    { id: 1, name: 'widget', count: 4321, deprecated: false },
  ] as RegistryClass[];
  target = document.createElement('div');
  document.body.appendChild(target);
  const ws = {
    url: { queue: 'to_label' },
    itemQuery: () => ({ class_name: ['widget'], review_tab: 'to_label' }),
    railOpen: true,
    filter: { classNames: [] },
    setQueue: vi.fn(),
    toggleRail: vi.fn(),
    applyFilter: vi.fn(),
  };
  instance = mount(WorkspaceRail, { target, props: { ws } as never });
  flushSync();
});
afterEach(() => {
  if (instance) unmount(instance);
  instance = null;
  target.remove();
  classesStore.classes = prevClasses;
  reviewTabsVocabularyStore.list = [];
});

describe('WorkspaceRail', () => {
  it('asks for the counts under the filter, not under the queue it shows', () => {
    expect(vi.mocked(getReviewTabs)).toHaveBeenCalledWith({ class_name: ['widget'] });
  });

  it('queue counts are the served count', () => {
    const counts = [...target.querySelectorAll('[data-testid^="queue-"]')].map((b) =>
      b.textContent?.replace(/\s+/g, ' ').trim(),
    );
    expect(counts).toEqual(['To label 1,234', 'All 5']);
  });

  it('class counts are the served count, in the served order', () => {
    const rows = [...target.querySelectorAll('section[aria-label="Classes"] li button')]
      .map((b) => b.textContent?.replace(/\s+/g, ' ').trim())
      .filter((t) => t && t !== 'All classes');
    expect(rows).toEqual(['gadget 7', 'widget 4,321']);
  });

  it('says what the count colours mean, from the served thresholds', () => {
    expect(target.querySelector('[data-testid="class-count-legend"]')).toBeNull();
    classesStore.thresholds = {
      ...classesStore.thresholds,
      block_below: 10,
      warn_below: 30,
    };
    flushSync();
    const legend = target.querySelector('[data-testid="class-count-legend"]')!;
    const text = legend.textContent!.replace(/\s+/g, ' ');
    expect(text).toContain('under 10');
    expect(text).toContain('under 30');
    classesStore.thresholds = {
      ...classesStore.thresholds,
      block_below: 0,
      warn_below: 0,
    };
  });

  it('no client sort', () => {
    const here = path.dirname(fileURLToPath(import.meta.url));
    for (const f of ['WorkspaceRail.svelte', 'RailClasses.svelte']) {
      expect(readFileSync(path.join(here, f), 'utf-8')).not.toMatch(/\.sort\(/);
    }
  });
});
