import { describe, expect, it, vi } from 'vitest';
import { flushSync, tick } from 'svelte';
import { mountUi } from '$lib/test/mountUi';

vi.mock('$lib/api_images', () => ({
  getImage: vi.fn().mockResolvedValue({ image: { width: 40, height: 30 }, boxes: [] }),
  getImageFileUrl: (id: string, max: number) => `/file/${id}?max_dim=${max}`,
}));

import Lightbox from './Lightbox.svelte';

const press = (key: string) => {
  window.dispatchEvent(
    new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true }),
  );
  flushSync();
};
const zoom = () => document.querySelector('[data-testid="lightbox-zoom"]')!.textContent;

describe('Lightbox', () => {
  it('is a labelled modal dialog over the full-size overlay at 2048 px', async () => {
    mountUi(Lightbox, { imageId: 'img-1', label: 'cat.jpg', onclose: vi.fn() });
    const d = document.querySelector('[data-testid="lightbox"]')!;
    expect(d.getAttribute('role')).toBe('dialog');
    expect(d.getAttribute('aria-label')).toBe('cat.jpg');
    await tick();
    expect(d.querySelector('img')?.getAttribute('src')).toBe('/file/img-1?max_dim=2048');
  });

  it('Escape closes it', () => {
    const onclose = vi.fn();
    mountUi(Lightbox, { imageId: 'img-1', onclose });
    document
      .querySelector('[data-testid="lightbox"]')!
      .dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    expect(onclose).toHaveBeenCalledOnce();
  });

  it('arrows step to a neighbour only when there is one', () => {
    const onprev = vi.fn();
    mountUi(Lightbox, { imageId: 'img-1', onclose: vi.fn(), onprev });
    press('ArrowLeft');
    press('ArrowRight');
    expect(onprev).toHaveBeenCalledOnce();
    const next = document.querySelector('[aria-label="Next image"]') as HTMLButtonElement;
    expect(next.disabled).toBe(true);
  });

  it('+ - 0 zoom within bounds and reset', () => {
    mountUi(Lightbox, { imageId: 'img-1', onclose: vi.fn() });
    expect(zoom()).toBe('100%');
    press('+');
    expect(zoom()).toBe('150%');
    press('-');
    press('-');
    expect(zoom()).toBe('100%');
    for (let i = 0; i < 10; i++) press('+');
    expect(zoom()).toBe('400%');
    press('0');
    expect(zoom()).toBe('100%');
  });
});
