import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';

vi.mock('$lib/api', async () => {
  const actual = await vi.importActual<typeof import('$lib/api')>('$lib/api');
  return { ...actual, getMatchingItemCount: vi.fn() };
});

import { ApiError, getMatchingItemCount } from '$lib/api';
import ExportItemFilter from './ExportItemFilter.svelte';
import { ItemFilterState } from '$lib/itemFilter/itemFilterState.svelte';

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | null = null;
const tick = () => new Promise((r) => setTimeout(r, 0));
const text = () =>
  target.querySelector('[data-testid="export-matching-count"]')?.textContent ?? null;

beforeEach(() => {
  target = document.createElement('div');
  document.body.appendChild(target);
});
afterEach(() => {
  if (instance) unmount(instance);
  instance = null;
  target.remove();
  vi.clearAllMocks();
});

describe('ExportItemFilter', () => {
  it('an empty filter asks the server nothing and shows no count', async () => {
    instance = mount(ExportItemFilter, {
      target,
      props: { state: new ItemFilterState() },
    });
    flushSync();
    await tick();
    expect(getMatchingItemCount).not.toHaveBeenCalled();
    expect(text()).toBeNull();
  });

  it("shows the server's total_crops for the same filter", async () => {
    vi.mocked(getMatchingItemCount).mockResolvedValue(1234);
    const state = new ItemFilterState();
    state.classNames = ['widget'];
    state.origin = ['sam3'];
    instance = mount(ExportItemFilter, { target, props: { state } });
    flushSync();
    await tick();
    flushSync();
    expect(getMatchingItemCount).toHaveBeenCalledWith({
      class_name: ['widget'],
      origin: ['sam3'],
    });
    expect(text()).toContain('1,234');
  });

  it('never sends the open-vocabulary pair, which stats/dataset does not declare', async () => {
    vi.mocked(getMatchingItemCount).mockResolvedValue(1);
    const state = new ItemFilterState();
    state.classNames = ['widget'];
    state.openVocabSet = 'tags';
    instance = mount(ExportItemFilter, { target, props: { state } });
    flushSync();
    await tick();
    expect(vi.mocked(getMatchingItemCount).mock.calls[0]![0]).toEqual({
      class_name: ['widget'],
    });
  });

  it('a served refusal shows under the bar and no count', async () => {
    vi.mocked(getMatchingItemCount).mockRejectedValue(
      new ApiError(400, '/x', { detail: 'conf_min must be <= conf_max' }),
    );
    const state = new ItemFilterState();
    state.confMin = 0.9;
    state.confMax = 0.1;
    instance = mount(ExportItemFilter, { target, props: { state } });
    flushSync();
    await tick();
    flushSync();
    expect(
      target.querySelector('[data-testid="item-filter-error"]')?.textContent,
    ).toContain('conf_min must be <= conf_max');
    expect(text()?.trim()).toBe('');
  });
});
