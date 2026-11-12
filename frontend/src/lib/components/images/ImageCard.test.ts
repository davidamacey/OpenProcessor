import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { imageCard } from '$lib/test/fixtures/imagePage';
import { imageBox } from '$lib/test/fixtures/imageDetail';
import ImageCard from './ImageCard.svelte';

class ResizeObserverStub {
  observe(): void {}
  unobserve(): void {}
  disconnect(): void {}
}
vi.stubGlobal('ResizeObserver', ResizeObserverStub);

let instance: Record<string, unknown> | undefined;
let target: HTMLDivElement;

function render(props: Record<string, unknown>): HTMLDivElement {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(ImageCard, {
    target,
    props: {
      selected: false,
      focused: false,
      onclick: vi.fn(),
      onopen: vi.fn(),
      ...props,
    } as never,
  });
  flushSync();
  return target;
}

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

describe('ImageCard', () => {
  it('count chips are the served counts', () => {
    const el = render({
      card: imageCard({
        counts: { items: 7, labeled: 5, validated: 2, unlabeled: 2, in_holdout: 0 },
      }),
    });
    expect(el.querySelector('[data-testid="chip-items"]')!.textContent).toContain('7');
    expect(el.querySelector('[data-testid="chip-labeled"]')!.textContent).toContain('5');
    expect(el.querySelector('[data-testid="chip-unlabeled"]')!.textContent).toContain(
      '2',
    );
  });

  it('near-dup badge only when served', () => {
    expect(
      render({ card: imageCard() }).querySelector('[data-testid="near-dup-badge"]'),
    ).toBeNull();
    unmount(instance!);
    target.remove();
    const el = render({
      card: imageCard({
        near_dup: { group_id: 'g1', purpose: 'dedup', size: 3, is_representative: false },
      }),
    });
    expect(el.querySelector('[data-testid="near-dup-badge"]')!.textContent).toContain(
      '3',
    );
  });

  it('boxes_truncated says so', () => {
    expect(
      render({ card: imageCard({ boxes_truncated: false }) }).querySelector(
        '[data-testid="boxes-truncated"]',
      ),
    ).toBeNull();
    unmount(instance!);
    target.remove();
    const el = render({ card: imageCard({ boxes_truncated: true }) });
    expect(el.querySelector('[data-testid="boxes-truncated"]')).not.toBeNull();
  });

  it('draws exactly the served boxes from the thumbnail url only', () => {
    const el = render({
      card: imageCard({
        boxes: [imageBox({ crop_id: 'a' }), imageBox({ crop_id: 'b' })],
      }),
    });
    expect(el.querySelectorAll('[data-testid="overlay-box"]')).toHaveLength(2);
    const img = el.querySelector('img')!;
    expect(img.getAttribute('src')).toContain('/thumbnail');
  });

  it('marks selection and focus distinctly and reports clicks', () => {
    const onclick = vi.fn();
    const onopen = vi.fn();
    const el = render({
      card: imageCard(),
      selected: true,
      focused: true,
      onclick,
      onopen,
    });
    const card = el.querySelector('[data-testid="image-card"]') as HTMLElement;
    expect(card.getAttribute('aria-current')).toBe('true');
    expect(card.getAttribute('data-selected')).toBe('true');
    (el.querySelector('[data-testid="image-card-select"]') as HTMLElement).click();
    expect(onclick).toHaveBeenCalledOnce();
    (el.querySelector('[data-testid="image-card-open"]') as HTMLElement).click();
    expect(onopen).toHaveBeenCalledOnce();
  });
});
