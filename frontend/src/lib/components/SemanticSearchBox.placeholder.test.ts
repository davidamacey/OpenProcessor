/** The default hint must fit a phone-width input (about 36 characters at 430 px). */
import { afterEach, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import SemanticSearchBox from './SemanticSearchBox.svelte';

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | undefined;

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

it('has a short default placeholder that still gives an example', () => {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(SemanticSearchBox, { target, props: {} });
  flushSync();
  const hint = target.querySelector('input')?.getAttribute('placeholder') ?? '';
  expect(hint.length).toBeGreaterThan(0);
  expect(hint.length).toBeLessThanOrEqual(36);
  expect(hint).toContain('e.g.');
});
