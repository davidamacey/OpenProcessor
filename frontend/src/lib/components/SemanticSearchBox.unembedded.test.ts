/**
 * The served `unembedded_in_scope` of `GET /search/text`: a banner under the
 * search line when it is above zero, absent otherwise.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import SemanticSearchBox from './SemanticSearchBox.svelte';
import { searchCrops } from '$lib/api';

vi.mock('$lib/api', async () => {
  const actual = await vi.importActual<typeof import('$lib/api')>('$lib/api');
  return { ...actual, searchCrops: vi.fn() };
});

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | undefined;

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
  vi.restoreAllMocks();
});

async function search(unembedded: number | null) {
  vi.mocked(searchCrops).mockResolvedValue({
    items: [],
    total: 0,
    page: 1,
    page_size: 30,
    unembedded_in_scope: unembedded,
  });
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(SemanticSearchBox, { target, props: { initialQuery: 'blue widget' } });
  await vi.waitFor(() => {
    flushSync();
    expect(target.textContent).toContain('result');
  });
}

describe('SemanticSearchBox unembedded banner', () => {
  it('shows the served count of items that cannot match', async () => {
    await search(42);
    expect(
      target.querySelector('[data-testid="unembedded-banner"]')!.textContent,
    ).toContain('42 items in scope have no vector and cannot match a text search');
  });

  it('shows nothing when the count is 0 or not served', async () => {
    await search(0);
    expect(target.querySelector('[data-testid="unembedded-banner"]')).toBeNull();
    unmount(instance!);
    target.remove();
    await search(null);
    expect(target.querySelector('[data-testid="unembedded-banner"]')).toBeNull();
  });
});
