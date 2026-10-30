/**
 * Cluster card thumbnails are decorative (the card's text names the cluster),
 * so they carry an empty alt: a thumbnail the server cannot serve must not
 * spill the filename-ish alt text into the grid. A failed load dims the tile.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';

vi.mock('$app/navigation', () => ({ goto: vi.fn() }));

import ClusterGridCard from './ClusterGridCard.svelte';
import type { Cluster } from '$lib/types';

let instance: ReturnType<typeof mount> | null = null;
let target: HTMLDivElement;

function render(): HTMLElement {
  target = document.createElement('div');
  document.body.appendChild(target);
  const c = {
    id: 7,
    size: 3,
    purity_tier: 'pure',
    representative_crop_ids: ['a', 'b'],
  } as unknown as Cluster;
  instance = mount(ClusterGridCard, { target, props: { c, flagVocabulary: [] } });
  flushSync();
  return target;
}

afterEach(() => {
  if (instance) unmount(instance);
  instance = null;
  target?.remove();
});

describe('ClusterGridCard thumbnails', () => {
  it('are decorative: empty alt, never the literal word "thumb"', () => {
    const imgs = [...render().querySelectorAll('img')];
    expect(imgs).toHaveLength(2);
    for (const img of imgs) expect(img.getAttribute('alt')).toBe('');
  });

  it('dim a tile whose image fails to load', () => {
    const img = render().querySelector('img') as HTMLImageElement;
    img.dispatchEvent(new Event('error'));
    expect(img.style.opacity).toBe('0.2');
  });
});
