import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { imageCard } from '$lib/test/fixtures/imagePage';
import ImageGrid from './ImageGrid.svelte';

class Stub {
  observe(): void {}
  unobserve(): void {}
  disconnect(): void {}
}
vi.stubGlobal('ResizeObserver', Stub);
vi.stubGlobal('IntersectionObserver', Stub);

let instance: Record<string, unknown> | undefined;
let target: HTMLDivElement;

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

describe('ImageGrid', () => {
  it('renders the cards in the order given and routes clicks by image id', () => {
    const onselect = vi.fn();
    const onopen = vi.fn();
    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(ImageGrid, {
      target,
      props: {
        cards: [
          imageCard({ image_id: 'z' }),
          imageCard({ image_id: 'a' }),
          imageCard({ image_id: 'm' }),
        ],
        isSelected: (id: string) => id === 'a',
        focusedId: 'm',
        hasMore: false,
        loadingMore: false,
        onselect,
        onopen,
        onloadmore: vi.fn(),
      } as never,
    });
    flushSync();
    const cards = [...target.querySelectorAll('[data-testid="image-card"]')];
    expect(cards.map((c) => c.getAttribute('data-image-id'))).toEqual(['z', 'a', 'm']);
    expect(cards.map((c) => c.getAttribute('data-selected'))).toEqual([
      'false',
      'true',
      'false',
    ]);
    (cards[1]!.querySelector('[data-testid="image-card-select"]') as HTMLElement).click();
    expect(onselect).toHaveBeenCalledWith('a', expect.any(MouseEvent));
    (cards[2]!.querySelector('[data-testid="image-card-open"]') as HTMLElement).click();
    expect(onopen).toHaveBeenCalledWith('m');
  });
});
