import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import VlmQuickAction from './VlmQuickAction.svelte';

let instance: ReturnType<typeof mount> | null = null;
let target: HTMLDivElement | null = null;

async function render(active: unknown) {
  vi.stubGlobal(
    'fetch',
    vi.fn(
      async () =>
        new Response(JSON.stringify(active), {
          status: active ? 200 : 500,
          headers: { 'content-type': 'application/json' },
        }),
    ),
  );
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(VlmQuickAction, { target, props: { onrun: () => {} } });
  await new Promise((r) => setTimeout(r, 20));
  flushSync();
  return target;
}

afterEach(() => {
  if (instance) unmount(instance);
  instance = null;
  target?.remove();
  vi.unstubAllGlobals();
});

describe('VlmQuickAction', () => {
  it('a project whose served VLM source is off links to the Models settings', async () => {
    const t = await render({ source: 'off', active: { name: null, revision: null } });
    expect(t.querySelector('[data-testid="vlm-activate-first"]')).not.toBeNull();
    expect(t.querySelector('button.btn-primary')).toBeNull();
  });

  it('a project with an active VLM keeps the primary button', async () => {
    const t = await render({ source: 'saved', active: { name: 'x', revision: 1 } });
    expect(t.querySelector('button.btn-primary')?.textContent).toContain(
      'Run VLM Labeling',
    );
    expect(t.querySelector('[data-testid="vlm-activate-first"]')).toBeNull();
  });
});
