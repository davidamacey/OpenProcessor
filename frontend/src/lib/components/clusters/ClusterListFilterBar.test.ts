/**
 * The filter bar renders the served sort options and flag vocabulary, and
 * sends each choice to the controller unchanged.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import ClusterListFilterBar from './ClusterListFilterBar.svelte';
import { EMPTY_CLUSTER_FILTER } from '$lib/clusters/clusterListFilter';
import { classesStore } from '$stores/classes.svelte';
import type { ClusterListFilter } from '$lib/types_clusterList';

let instance: ReturnType<typeof mount> | null = null;
let target: HTMLDivElement;

function fakeList(filter: ClusterListFilter = EMPTY_CLUSTER_FILTER) {
  return {
    sort: null as string | null,
    appliedSort: 'needs_review_desc',
    sortOptions: [
      { id: 'needs_review_desc', label: 'Most to review' },
      { id: 'size_asc', label: 'Smallest first' },
    ],
    flagVocabulary: [
      { id: 'unvalidated', label: 'Unvalidated', description: 'No member is validated.' },
      { id: 'promotable', label: 'Promotable', description: 'Clears the gate.' },
    ],
    detectorClassOptions: ['widget_x', 'widget_y'],
    filter,
    filterActive: JSON.stringify(filter) !== JSON.stringify(EMPTY_CLUSTER_FILTER),
    setFilter: vi.fn(),
    clearFilter: vi.fn(),
  };
}

function render(list = fakeList()) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(ClusterListFilterBar, { target, props: { list: list as never } });
  flushSync();
  return { list, root: target };
}

afterEach(() => {
  if (instance) unmount(instance);
  instance = null;
  target?.remove();
  classesStore.classes = [];
});

describe('ClusterListFilterBar', () => {
  it('lists the served sort options verbatim and selects the applied one', () => {
    const { root } = render();
    const select = root.querySelector(
      '[data-testid="cluster-sort"]',
    ) as HTMLSelectElement;
    expect([...select.options].map((o) => [o.value, o.textContent])).toEqual([
      ['needs_review_desc', 'Most to review'],
      ['size_asc', 'Smallest first'],
    ]);
    expect(select.value).toBe('needs_review_desc');
  });

  it('sends the chosen sort id', () => {
    const { root, list } = render();
    const select = root.querySelector(
      '[data-testid="cluster-sort"]',
    ) as HTMLSelectElement;
    select.value = 'size_asc';
    select.dispatchEvent(new Event('change', { bubbles: true }));
    expect(list.sort).toBe('size_asc');
  });

  it('renders a pressed button per served flag with its description as tooltip', () => {
    const { root, list } = render(
      fakeList({ ...EMPTY_CLUSTER_FILTER, flag: ['promotable'] }),
    );
    const un = root.querySelector('[data-testid="cluster-flag-filter-unvalidated"]')!;
    const pr = root.querySelector('[data-testid="cluster-flag-filter-promotable"]')!;
    expect(un.getAttribute('title')).toBe('No member is validated.');
    expect(un.getAttribute('aria-pressed')).toBe('false');
    expect(pr.getAttribute('aria-pressed')).toBe('true');
    (un as HTMLButtonElement).click();
    expect(list.setFilter).toHaveBeenCalledWith(
      expect.objectContaining({ flag: ['promotable', 'unvalidated'] }),
    );
    (pr as HTMLButtonElement).click();
    expect(list.setFilter).toHaveBeenLastCalledWith(
      expect.objectContaining({ flag: [] }),
    );
  });

  it('adds a detector class chip from the served options and removes it again', () => {
    const { root, list } = render();
    const select = root.querySelector(
      'select[aria-label="Add detector class"]',
    ) as HTMLSelectElement;
    select.value = 'widget_y';
    select.dispatchEvent(new Event('change', { bubbles: true }));
    expect(list.setFilter).toHaveBeenCalledWith(
      expect.objectContaining({ detector_class: ['widget_y'] }),
    );
  });

  it('offers only active classes as dominant class options', () => {
    classesStore.classes = [
      { id: 1, name: 'class_a' },
      { id: 2, name: 'class_old', deprecated: true },
    ] as never;
    const { root } = render();
    const select = root.querySelector(
      'select[aria-label="Add dominant class"]',
    ) as HTMLSelectElement;
    expect([...select.options].map((o) => o.value)).toEqual(['', 'class_a']);
  });

  it('commits a size bound on change and clears it when emptied', () => {
    const { root, list } = render();
    const min = root.querySelector(
      'input[aria-label="Minimum cluster size"]',
    ) as HTMLInputElement;
    min.value = '12';
    min.dispatchEvent(new Event('change', { bubbles: true }));
    expect(list.setFilter).toHaveBeenCalledWith(
      expect.objectContaining({ min_size: 12 }),
    );
    min.value = '';
    min.dispatchEvent(new Event('change', { bubbles: true }));
    expect(list.setFilter).toHaveBeenLastCalledWith(
      expect.objectContaining({ min_size: null }),
    );
  });

  it('shows Clear filters only while a filter is set', () => {
    expect(
      render().root.querySelector('[data-testid="cluster-filters-clear"]'),
    ).toBeNull();
    unmount(instance!);
    target.remove();
    const { root, list } = render(
      fakeList({ ...EMPTY_CLUSTER_FILTER, needs_review: true }),
    );
    (
      root.querySelector('[data-testid="cluster-filters-clear"]') as HTMLButtonElement
    ).click();
    expect(list.clearFilter).toHaveBeenCalled();
  });

  it('has no sort control before the server has served options', () => {
    const list = fakeList();
    list.sortOptions = [];
    expect(render(list).root.querySelector('[data-testid="cluster-sort"]')).toBeNull();
  });
});
