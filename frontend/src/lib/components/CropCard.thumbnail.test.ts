/** A thumbnail that fails to load is asked for once more, then replaced by a neutral tile. */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import type { Crop } from '$lib/types';

vi.mock('$lib/sourceMissing', () => ({ fetchSourceMissing: vi.fn(async () => null) }));

const { default: CropCard } = await import('./CropCard.svelte');

let target: HTMLDivElement;
let instance: unknown;

beforeEach(() => {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(CropCard, {
    target,
    props: {
      crop: {
        id: 'c1',
        class_id: 3,
        class_name: 'widget_a',
        label_source: 'model_x',
        bbox_norm: { cx: 0.5, cy: 0.5, w: 0.4, h: 0.4 },
      } as unknown as Crop,
    },
  } as never);
  flushSync();
});
afterEach(() => {
  unmount(instance as never);
  target.remove();
});

const img = () => target.querySelector<HTMLImageElement>('img');
const fail = () => {
  img()!.dispatchEvent(new Event('error'));
  flushSync();
};

describe('CropCard thumbnail', () => {
  it('retries once with a fresh request, then shows a "No preview" tile', () => {
    const first = img()!.getAttribute('src')!;
    expect(target.querySelector('[data-testid="no-preview"]')).toBeNull();

    fail();
    expect(img()).not.toBeNull();
    const second = img()!.getAttribute('src')!;
    expect(second).not.toBe(first);
    expect(second).toContain(first);
    expect(target.querySelector('[data-testid="no-preview"]')).toBeNull();

    fail();
    expect(img()).toBeNull();
    expect(target.querySelector('[data-testid="no-preview"]')!.textContent).toContain(
      'No preview',
    );
  });
});
