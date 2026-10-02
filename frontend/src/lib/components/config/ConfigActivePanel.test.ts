/**
 * The shared active panel with the region-profile words: "off" for a null
 * active ref, the applied runtimes' PROFILE ref, Turn off only when
 * something is active and the resource has the route, behind a confirm
 * that names both states, and an extra action rendered in the header.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { createRawSnippet, flushSync, mount, unmount } from 'svelte';
import { ConfigActive } from '$lib/config/configActive.svelte';
import { PROFILE_ACTIVE_COPY } from '$lib/profiles/profileCopy';
import { profileActiveFixture } from '$lib/test/fixtures/regionProfiles';
import type { ActiveConfigResponse } from '$lib/types_config';
import ConfigActivePanel from './ConfigActivePanel.svelte';

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
  opts: { deactivate?: boolean; ondeactivate?: () => Promise<boolean> } = {},
) {
  const ctl = new ConfigActive({
    getActive: vi.fn().mockResolvedValue(active),
    activate: vi.fn(),
    rollback: vi.fn(),
  });
  await ctl.load();
  const ondeactivate = opts.ondeactivate ?? vi.fn().mockResolvedValue(true);
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(ConfigActivePanel, {
    target,
    props: {
      ctl,
      copy:
        opts.deactivate === false
          ? { ...PROFILE_ACTIVE_COPY, deactivate: undefined }
          : PROFILE_ACTIVE_COPY,
      onrollback: vi.fn().mockResolvedValue(true),
      ondeactivate,
      actions: createRawSnippet(() => ({
        render: () => '<button data-testid="extra-action">Impact</button>',
      })),
    },
  });
  flushSync();
  return { ctl, ondeactivate };
}

const q = (id: string) => document.querySelector(`[data-testid="${id}"]`);

describe('ConfigActivePanel (region-profile words)', () => {
  it('shows the active ref, the applied profile ref and the extra action', async () => {
    await render(profileActiveFixture());
    expect(target.querySelector('section')?.getAttribute('aria-label')).toBe(
      'Active region profile',
    );
    expect(q('active-ref')?.textContent).toBe('widget_tag r2');
    expect(q('active-applied')?.textContent).toContain('Profile');
    expect(q('active-applied')?.textContent).toContain('widget_tag r2');
    // The served worker host and when it applied (both blank before f14f4ddc).
    expect(q('active-applied')?.textContent).toContain('worker-1');
    expect(q('applied-at')?.textContent).toBe('2026-09-26 12:05:02 UTC');
    expect(q('extra-action')).not.toBeNull();
  });

  it('nothing active reads as off, and there is nothing to turn off', async () => {
    await render(
      profileActiveFixture({ active: { name: null, revision: null }, previous: null }),
    );
    expect(q('active-ref')?.textContent).toBe('off: region detection is off');
    expect(q('active-deactivate')).toBeNull();
  });

  it('no deactivate copy (a pack) means no Turn off', async () => {
    await render(profileActiveFixture(), { deactivate: false });
    expect(q('active-deactivate')).toBeNull();
  });

  it('Turn off opens a confirm naming both states; only the confirm runs it', async () => {
    const { ondeactivate } = await render(profileActiveFixture());
    (q('active-deactivate') as HTMLButtonElement).click();
    flushSync();
    const dialog = document.querySelector('[role="dialog"]')!;
    expect(dialog.getAttribute('aria-label')).toBe('Turn off region detection');
    expect(dialog.textContent).toContain('widget_tag r2');
    expect(dialog.textContent).toContain('off: region detection is off');
    expect(ondeactivate).not.toHaveBeenCalled();
    [...dialog.querySelectorAll('button')]
      .find((b) => b.textContent?.trim() === 'Turn off')!
      .click();
    await vi.waitFor(() => expect(ondeactivate).toHaveBeenCalledTimes(1));
    await vi.waitFor(() => expect(document.querySelector('[role="dialog"]')).toBeNull());
  });

  it('a refused Turn off keeps the dialog with the served message', async () => {
    let ctlRef: ConfigActive | null = null;
    const { ctl } = await render(profileActiveFixture(), {
      ondeactivate: vi.fn().mockImplementation(async () => {
        ctlRef!.actionError = 'Another operator changed it.';
        return false;
      }),
    });
    ctlRef = ctl;
    (q('active-deactivate') as HTMLButtonElement).click();
    flushSync();
    const dialog = document.querySelector('[role="dialog"]')!;
    [...dialog.querySelectorAll('button')]
      .find((b) => b.textContent?.trim() === 'Turn off')!
      .click();
    await vi.waitFor(() =>
      expect(q('deactivate-error')?.textContent).toContain(
        'Another operator changed it.',
      ),
    );
  });
});
