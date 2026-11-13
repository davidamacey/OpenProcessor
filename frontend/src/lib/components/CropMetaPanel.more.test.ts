/** The crop panel keeps its rare actions, Assist and Reprocess, behind one More menu. */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import type { Crop } from '$lib/types';

vi.mock('$app/paths', () => ({ resolve: (p: string) => p }));
vi.mock('$lib/api_cropRead', () => ({
  getCropHistory: vi.fn(async () => ({ crop_id: 'c1', entries: [] })),
  getThumbUrl: (id: string) => `/thumb/${id}`,
}));
vi.mock('$lib/api_images', () => ({
  getImage: vi.fn(async () => ({ image: null, items: [] })),
}));

const { default: CropMetaPanel } = await import('./CropMetaPanel.svelte');

let instance: unknown;
let target: HTMLDivElement;
afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

function render(): HTMLDivElement {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(CropMetaPanel, {
    target,
    props: { crop: { id: 'c1', class_name: 'widget' } as unknown as Crop },
  });
  flushSync();
  return target;
}

describe('CropMetaPanel actions', () => {
  it('has no loose Assist button; Assist is in the More menu', () => {
    const el = render();
    expect(el.querySelector('[data-testid="assist-button"]')).toBeNull();
    const more = el.querySelector<HTMLButtonElement>('[data-testid="crop-meta-more"]')!;
    expect(more.textContent).toContain('More');
    more.click();
    flushSync();
    const items = [...el.querySelectorAll('[role="menuitem"]')].map((i) =>
      i.textContent?.trim(),
    );
    expect(items).toContain('Assist…');
  });
});
