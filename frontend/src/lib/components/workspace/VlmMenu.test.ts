import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import VlmMenu from './VlmMenu.svelte';

let instance: ReturnType<typeof mount> | null = null;
let target: HTMLDivElement | null = null;

function fakeWs(selection: unknown = { kind: 'none' }) {
  return {
    selection,
    itemSelection: () =>
      (selection as { kind: string }).kind === 'items'
        ? { crop_ids: (selection as { cropIds: string[] }).cropIds }
        : null,
    itemBody: () => ({ review_tab: 'to_label' }),
  };
}

async function render(active: unknown, ws = fakeWs()) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      if (String(url).includes('/vlm/endpoints/active')) {
        return new Response(JSON.stringify(active), {
          status: active ? 200 : 500,
          headers: { 'content-type': 'application/json' },
        });
      }
      return new Response('{}', { status: 404 });
    }),
  );
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(VlmMenu, { target, props: { ws } as never });
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

describe('VlmMenu', () => {
  it('offers assist only when a VLM is active (served)', async () => {
    const off = await render({ source: 'off', active: { name: null, revision: null } });
    expect(off.querySelector('[data-testid="vlm-menu"]')).toBeNull();
    unmount(instance!);
    instance = null;
    off.remove();
    const on = await render({ source: 'saved', active: { name: 'x', revision: 1 } });
    expect(on.querySelector('[data-testid="vlm-menu-assist"]')).not.toBeNull();
  });

  it('an unreadable answer leaves the menu out', async () => {
    const t = await render(null);
    expect(t.querySelector('[data-testid="vlm-menu"]')).toBeNull();
  });

  it('names the selection when crops are selected, else the scope', async () => {
    const sel = await render(
      { source: 'saved', active: { name: 'x', revision: 1 } },
      fakeWs({ kind: 'items', cropIds: ['a'] }),
    );
    expect(sel.querySelector('[data-testid="vlm-menu-assist"]')?.textContent).toContain(
      'the selection',
    );
    unmount(instance!);
    instance = null;
    sel.remove();
    const scope = await render({ source: 'saved', active: { name: 'x', revision: 1 } });
    expect(scope.querySelector('[data-testid="vlm-menu-assist"]')?.textContent).toContain(
      'the scope',
    );
  });
});
