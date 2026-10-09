import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import AboutModal from './AboutModal.svelte';

let instance: ReturnType<typeof mount> | undefined;
let target: HTMLElement | undefined;

afterEach(() => {
  if (instance) unmount(instance);
  target?.remove();
  instance = undefined;
  target = undefined;
});

function render(open: boolean): string {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(AboutModal, {
    target,
    props: { open, onclose: vi.fn(), appName: 'Cropwright' },
  });
  flushSync();
  return target.textContent ?? '';
}

describe('AboutModal license', () => {
  it('names the AGPL-3.0 license the repo ships', () => {
    const text = render(true);
    expect(text).toContain('AGPL-3.0');
    expect(text).not.toMatch(/\bMIT\b/);
  });

  it('renders nothing while closed', () => {
    expect(render(false)).not.toContain('AGPL-3.0');
  });
});
