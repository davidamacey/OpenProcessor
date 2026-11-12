import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import Filmstrip from './Filmstrip.svelte';

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | undefined;

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

const tiles = [
  { id: 'a', label: 'Car', thumbUrl: '/t/a' },
  { id: 'b', label: 'Bus', thumbUrl: '/t/b' },
  { id: 'c', label: 'Truck', thumbUrl: '/t/c' },
];

function render(props: Record<string, unknown> = {}) {
  target = document.createElement('div');
  document.body.appendChild(target);
  const onSelect = vi.fn();
  instance = mount(Filmstrip, {
    target,
    props: { tiles, focusedId: 'b', index: 11, total: 412, onSelect, ...props },
  });
  flushSync();
  return onSelect;
}

const items = () => [
  ...target.querySelectorAll<HTMLElement>('[data-testid="filmstrip-item"]'),
];

describe('Filmstrip (buttons, as Review uses it)', () => {
  it('says where the focus sits in the served order', () => {
    render();
    expect(
      target.querySelector('[data-testid="filmstrip-position"]')?.textContent?.trim(),
    ).toBe('12 of 412');
  });

  it('marks only the focused item as current and names each by its served label', () => {
    render();
    expect(items().map((b) => b.getAttribute('aria-label'))).toEqual([
      'Car',
      'Bus',
      'Truck',
    ]);
    expect(items().map((b) => b.getAttribute('aria-current'))).toEqual([
      null,
      'true',
      null,
    ]);
  });

  it('selects an item on click', () => {
    const onSelect = render();
    items()[2]!.click();
    expect(onSelect).toHaveBeenCalledWith('c');
  });

  it('shows no position when the server did not serve one', () => {
    render({ index: null, total: null });
    expect(target.querySelector('[data-testid="filmstrip-position"]')).toBeNull();
  });

  it('leaves an empty tile, not a broken image, for a thumbnail the server cannot serve', () => {
    render();
    const imgs = target.querySelectorAll('img');
    expect(imgs).toHaveLength(3);
    imgs[1]!.dispatchEvent(new Event('error'));
    flushSync();
    expect(target.querySelectorAll('img')).toHaveLength(2);
    expect(items()).toHaveLength(3);
  });
});

describe('Filmstrip (links, as the image editor uses it)', () => {
  const links = ['img-0', 'img-1', 'img-2', 'img-3'].map((id) => ({
    id,
    label: id === 'img-1' ? 'This image' : `Open image ${id}`,
    thumbUrl: `/images/${id}/thumbnail?size=96`,
    href: `/p/default/images/${id}?sort=indexed_at`,
  }));

  it('draws the tiles in the served order as links with the list query', () => {
    render({ tiles: links, focusedId: 'img-1', index: null, total: null });
    expect(
      [...target.querySelectorAll('a[data-image-id]')].map((a) =>
        a.getAttribute('data-image-id'),
      ),
    ).toEqual(['img-0', 'img-1', 'img-2', 'img-3']);
    expect(target.querySelector('a[data-image-id="img-2"]')!.getAttribute('href')).toBe(
      '/p/default/images/img-2?sort=indexed_at',
    );
  });

  it('marks the open image', () => {
    render({ tiles: links, focusedId: 'img-1', index: null, total: null });
    expect(
      target.querySelector('a[data-image-id="img-1"]')!.getAttribute('aria-current'),
    ).toBe('true');
    expect(
      target.querySelector('a[data-image-id="img-2"]')!.getAttribute('aria-current'),
    ).toBeNull();
  });

  it('draws each thumbnail from the given url', () => {
    render({ tiles: links, focusedId: 'img-1', index: null, total: null });
    expect(
      target.querySelector('a[data-image-id="img-2"] img')!.getAttribute('src'),
    ).toMatch(/\/images\/img-2\/thumbnail\?size=96$/);
  });
});
