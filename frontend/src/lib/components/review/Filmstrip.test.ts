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

const items = [
  { id: 'a', displayName: 'Car' },
  { id: 'b', displayName: 'Bus' },
  { id: 'c', displayName: 'Truck' },
];

interface Overrides {
  index?: number | null;
  total?: number | null;
}

function render(overrides: Overrides = {}) {
  target = document.createElement('div');
  document.body.appendChild(target);
  const onSelect = vi.fn();
  instance = mount(Filmstrip, {
    target,
    props: { items, focusedId: 'b', index: 11, total: 412, onSelect, ...overrides },
  });
  flushSync();
  return onSelect;
}

describe('Filmstrip', () => {
  it('says where the focus sits in the served order', () => {
    render();
    expect(
      target.querySelector('[data-testid="filmstrip-position"]')?.textContent?.trim(),
    ).toBe('12 of 412');
  });

  it('marks only the focused item as current and names each by its served name', () => {
    render();
    const buttons = [
      ...target.querySelectorAll<HTMLButtonElement>('[data-testid="filmstrip-item"]'),
    ];
    expect(buttons.map((b) => b.getAttribute('aria-label'))).toEqual([
      'Car',
      'Bus',
      'Truck',
    ]);
    expect(buttons.map((b) => b.getAttribute('aria-current'))).toEqual([
      null,
      'true',
      null,
    ]);
  });

  it('selects an item on click', () => {
    const onSelect = render();
    target
      .querySelectorAll<HTMLButtonElement>('[data-testid="filmstrip-item"]')[2]
      ?.click();
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
    expect(target.querySelectorAll('[data-testid="filmstrip-item"]')).toHaveLength(3);
  });
});
