import { afterEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import Filmstrip from './Filmstrip.svelte';

let target: HTMLDivElement;
let instance: Record<string, unknown> | undefined;

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

function render(before: string | null) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(Filmstrip, {
    target,
    props: {
      before,
      after: ['img-2', 'img-3'],
      currentId: 'img-1',
      hrefFor: (id: string) => `/p/default/images/${id}?sort=indexed_at`,
    },
  });
  flushSync();
}

const ids = () =>
  [...target.querySelectorAll('a[data-image-id]')].map((a) =>
    a.getAttribute('data-image-id'),
  );

describe('Filmstrip', () => {
  it('shows the previous image, this one and the next ones in the served order', () => {
    render('img-0');
    expect(ids()).toEqual(['img-0', 'img-1', 'img-2', 'img-3']);
  });

  it('starts at this image when there is none before it', () => {
    render(null);
    expect(ids()).toEqual(['img-1', 'img-2', 'img-3']);
  });

  it('marks this image and links the others with the list query', () => {
    render('img-0');
    const current = target.querySelector('a[data-image-id="img-1"]')!;
    expect(current.getAttribute('aria-current')).toBe('true');
    expect(target.querySelector('a[data-image-id="img-2"]')!.getAttribute('href')).toBe(
      '/p/default/images/img-2?sort=indexed_at',
    );
  });

  it('draws each thumbnail from the image thumbnail route', () => {
    render(null);
    expect(
      target.querySelector('a[data-image-id="img-2"] img')!.getAttribute('src'),
    ).toMatch(/\/images\/img-2\/thumbnail\?size=96$/);
  });
});
