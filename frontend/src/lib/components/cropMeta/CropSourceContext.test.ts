import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import type { Crop } from '$lib/types';
import CropSourceContext from './CropSourceContext.svelte';

const crop = (id: string): Crop => ({ id, class_name: `cls-${id}` }) as unknown as Crop;

let instance: Record<string, unknown> | undefined;
let target: HTMLDivElement;

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

describe('CropSourceContext siblings', () => {
  it('a sibling click opens that crop', () => {
    const onopencrop = vi.fn();
    const sibling = crop('s1');
    const loader = {
      imageMeta: { width: 10, height: 10, source: 'x', indexed_at: null },
      siblings: [sibling, crop('s2')],
      imageError: null,
      imageLoading: false,
      contextRes: null,
    };
    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(CropSourceContext, {
      target,
      props: { crop: crop('c0'), loader, onopencrop } as never,
    });
    flushSync();
    const buttons = target.querySelectorAll('button[data-testid="sibling-crop"]');
    expect(buttons).toHaveLength(2);
    (buttons[0] as HTMLElement).click();
    expect(onopencrop).toHaveBeenCalledWith(sibling);
  });
});
