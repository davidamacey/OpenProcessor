/** The `/settings` card links every preset kind and the embedding spaces, and marks the deployment-wide one. */
import { afterEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { PRESET_KINDS } from '$lib/presets/presetKinds';
import PresetsCard from './PresetsCard.svelte';

let target: HTMLDivElement;
let instance: Record<string, unknown> | undefined;

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

describe('PresetsCard', () => {
  it('links each kind page and the embedding spaces page', () => {
    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(PresetsCard, { target });
    flushSync();
    const hrefs = [...target.querySelectorAll('a')].map((a) => a.getAttribute('href'));
    for (const k of PRESET_KINDS) {
      expect(hrefs.some((h) => h?.endsWith(`/settings/presets/${k.slug}`))).toBe(true);
    }
    expect(hrefs.some((h) => h?.endsWith('/settings/embedding-spaces'))).toBe(true);
    expect(target.textContent).toContain('deployment-wide');
  });
});
