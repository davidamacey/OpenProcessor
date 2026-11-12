import { afterEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import AppLogo from './AppLogo.svelte';

let instance: ReturnType<typeof mount> | undefined;
let target: HTMLElement | undefined;

afterEach(() => {
  if (instance) unmount(instance);
  target?.remove();
  instance = undefined;
});

function render(props: { class?: string; label?: string }): SVGElement {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(AppLogo, { target, props });
  flushSync();
  return target.querySelector('svg')!;
}

describe('AppLogo', () => {
  it('is an image for assistive tech only when it has a label', () => {
    const labelled = render({ class: 'h-6 w-6', label: 'CW' });
    expect(labelled.getAttribute('role')).toBe('img');
    expect(labelled.getAttribute('aria-label')).toBe('CW');
    expect(labelled.getAttribute('class')).toBe('h-6 w-6');
  });

  it('is decoration without a label', () => {
    const bare = render({ class: 'h-8 w-8' });
    expect(bare.hasAttribute('role')).toBe(false);
    expect(bare.hasAttribute('aria-label')).toBe(false);
    expect(bare.getAttribute('viewBox')).toBe('0 0 128 128');
  });
});
