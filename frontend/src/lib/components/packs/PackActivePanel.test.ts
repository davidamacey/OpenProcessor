/**
 * The active-pack panel, mounted: the served active ref, stale flag and
 * applied runtimes (with the served `lagging`), and Rollback only when the
 * server names a previous pack, behind a confirm.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { PackActive } from '$lib/packs/packActive.svelte';
import { activeFixture } from '$lib/test/fixtures/promptPacks';
import type { ActiveConfigResponse } from '$lib/types_config';
import PackActivePanel from './PackActivePanel.svelte';

let target: HTMLDivElement;
let instance: Record<string, unknown> | undefined;

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
  document.querySelectorAll('[role="dialog"]').forEach((d) => d.remove());
});

async function render(
  active: ActiveConfigResponse,
  onrollback = vi.fn().mockResolvedValue(true),
) {
  const ctl = new PackActive({ getActivePromptPack: vi.fn().mockResolvedValue(active) });
  await ctl.load();
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(PackActivePanel, { target, props: { ctl, onrollback } });
  flushSync();
  return { ctl, onrollback };
}

const q = (id: string) => document.querySelector(`[data-testid="${id}"]`);
const button = (text: string) =>
  [...document.querySelectorAll('button')].find((b) =>
    b.textContent?.trim().startsWith(text),
  );

describe('PackActivePanel', () => {
  it('shows the served active ref, source and applied runtimes', async () => {
    await render(activeFixture());
    expect(q('active-ref')?.textContent).toBe('widget_tag r1');
    expect(target.textContent).toContain('(stored)');
    expect(q('active-applied')?.textContent).toContain('worker-1');
    expect(q('applied-lagging')).toBeNull();
    expect(q('active-stale')).toBeNull();
  });

  it('shows stale and lagging only when served', async () => {
    const a = activeFixture({ stale: true });
    a.applied![0]!.lagging = true;
    await render(a);
    expect(q('active-stale')).not.toBeNull();
    expect(q('applied-lagging')).not.toBeNull();
  });

  it('no active pack reads as the deployment default; no previous means no Rollback', async () => {
    await render(
      activeFixture({ active: { name: null, revision: null }, previous: null }),
    );
    expect(q('active-ref')?.textContent).toContain('the deployment default applies');
    expect(button('Roll back')).toBeUndefined();
  });

  it('Rollback opens a confirm naming both refs, and only the confirm runs it', async () => {
    const { onrollback } = await render(activeFixture());
    button('Roll back to generic_item_v1')!.click();
    flushSync();
    const dialog = document.querySelector('[role="dialog"]')!;
    expect(dialog.textContent).toContain('widget_tag r1');
    expect(dialog.textContent).toContain('generic_item_v1');
    expect(onrollback).not.toHaveBeenCalled();
    [...dialog.querySelectorAll('button')]
      .find((b) => b.textContent?.trim() === 'Roll back')!
      .click();
    await vi.waitFor(() => expect(onrollback).toHaveBeenCalledTimes(1));
    await vi.waitFor(() => expect(document.querySelector('[role="dialog"]')).toBeNull());
  });

  it('a refused rollback keeps the dialog open with the served message', async () => {
    const { ctl } = await render(
      activeFixture(),
      vi.fn().mockImplementation(async () => {
        ctl.actionError = 'Nothing to roll back to.';
        return false;
      }),
    );
    button('Roll back to')!.click();
    flushSync();
    const dialog = document.querySelector('[role="dialog"]')!;
    [...dialog.querySelectorAll('button')]
      .find((b) => b.textContent?.trim() === 'Roll back')!
      .click();
    await vi.waitFor(() =>
      expect(q('rollback-error')?.textContent).toContain('Nothing to roll back to.'),
    );
  });
});
