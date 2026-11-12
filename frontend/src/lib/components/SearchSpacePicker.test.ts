import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { getEmbeddingSpaces } from '$lib/api_embeddingSpaces';
import type { EmbeddingSpaceView } from '$lib/types_embeddingSpaces';
import SearchSpacePicker from './SearchSpacePicker.svelte';

vi.mock('$lib/api_embeddingSpaces', () => ({ getEmbeddingSpaces: vi.fn() }));

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | undefined;

const space = (
  key: string,
  over: Partial<EmbeddingSpaceView> = {},
): EmbeddingSpaceView => ({
  space_key: key,
  target: 'item',
  model_key: key,
  model_version: 1,
  dim: 8,
  role: 'optional',
  status: 'ready',
  ready_at: null,
  docs_with_vector: null,
  run_id: null,
  run_mode: null,
  run_status: null,
  ...over,
});

beforeEach(() => {
  target = document.createElement('div');
  document.body.appendChild(target);
});

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target.remove();
  vi.resetAllMocks();
});

const q = (id: string) => target.querySelector(`[data-testid="${id}"]`);

describe('SearchSpacePicker', () => {
  it('renders nothing with a single ready space', async () => {
    vi.mocked(getEmbeddingSpaces).mockResolvedValue([space('a', { role: 'primary' })]);
    instance = mount(SearchSpacePicker, { target, props: { onChange: vi.fn() } });
    await vi.waitFor(() => expect(getEmbeddingSpaces).toHaveBeenCalled());
    await Promise.resolve();
    flushSync();
    expect(q('search-space-picker')).toBeNull();
  });

  it('lists only ready spaces and reports the chosen space and the other ready spaces to fuse', async () => {
    vi.mocked(getEmbeddingSpaces).mockResolvedValue([
      space('a', { role: 'primary' }),
      space('b'),
      space('c', { status: 'building' }),
    ]);
    const onChange = vi.fn();
    instance = mount(SearchSpacePicker, { target, props: { onChange } });
    await vi.waitFor(() => {
      flushSync();
      expect(q('search-space-picker')).not.toBeNull();
    });
    const options = [...target.querySelectorAll('option')].map((o) => o.value);
    expect(options).toEqual(['a', 'b']);

    const fuse = q('search-space-fuse') as HTMLInputElement;
    fuse.click();
    flushSync();
    expect(onChange).toHaveBeenLastCalledWith({ space: null, fuse: ['b'] });

    const select = q('search-space-select') as HTMLSelectElement;
    select.value = 'b';
    select.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    expect(onChange).toHaveBeenLastCalledWith({ space: 'b', fuse: ['a'] });
  });

  it('stays hidden when the list cannot be loaded', async () => {
    vi.mocked(getEmbeddingSpaces).mockRejectedValue(new Error('down'));
    instance = mount(SearchSpacePicker, { target, props: { onChange: vi.fn() } });
    await vi.waitFor(() => expect(getEmbeddingSpaces).toHaveBeenCalled());
    await Promise.resolve();
    flushSync();
    expect(q('search-space-picker')).toBeNull();
  });
});
