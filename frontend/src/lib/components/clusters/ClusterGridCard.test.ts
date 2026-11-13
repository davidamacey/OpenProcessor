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

function render(over: Record<string, unknown> = {}): HTMLElement {
  target = document.createElement('div');
  document.body.appendChild(target);
  const c = {
    uid: '0b9f9a38-0000-4000-8000-000000000007',
    size: 3,
    representative_crop_ids: ['a', 'b'],
    display_no: 4,
    display_name: '',
    display_name_kind: 'none',
    ...over,
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

describe('ClusterGridCard name', () => {
  const name = (el: HTMLElement) => el.querySelector('[data-testid="cluster-name"]');

  it("shows the served detector name and says in words that it is the detector's", () => {
    const el = render({ display_name: 'bus', display_name_kind: 'detector' });
    expect(name(el)?.textContent).toBe('bus');
    expect(el.textContent).toContain('from the detector');
    expect(el.textContent).not.toContain('needs a label');
  });

  it('shows a class name without that note', () => {
    const el = render({ display_name: 'dog', display_name_kind: 'class' });
    expect(name(el)?.textContent).toBe('dog');
    expect(el.textContent).not.toContain("detector's");
  });

  it('keeps "Unlabeled #n, needs label" when no name was served', () => {
    const el = render();
    expect(name(el)).toBeNull();
    expect(el.textContent).toContain('Unlabeled #4');
    expect(el.textContent).toContain('needs a label');
  });
});

describe('ClusterGridCard duplicates', () => {
  it('keeps the duplicate-group count in the hover text, not as a chip', () => {
    const hover = (el: HTMLElement) => el.querySelector('button')!.getAttribute('title');
    expect(hover(render({ near_dup_group_count: 3 }))).toContain('3 duplicate groups');
    expect(hover(render({ near_dup_group_count: 1 }))).toContain('1 duplicate group');
    expect(hover(render({ near_dup_group_count: 0 }))).not.toContain('duplicate group');
    expect(render({ near_dup_group_count: 3 }).textContent).not.toContain('duplicate');
  });
});
