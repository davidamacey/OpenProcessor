/**
 * The shared activate dialog: with no `ack` prop (packs, region profiles)
 * it renders exactly as before and never sends `acknowledge_external`; with
 * `ack` it shows the served warning in a red banner, the "I understand"
 * checkbox only when `required` (or after the server refused for a missing
 * acknowledgement), and sends `acknowledge_external: true` only when
 * checked. A served refusal's message stays in the dialog.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { ApiError } from '$lib/api';
import { ConfigActive } from '$lib/config/configActive.svelte';
import { activeFixture } from '$lib/test/fixtures/vlm';
import ConfigActivateDialog from './ConfigActivateDialog.svelte';

let instance: ReturnType<typeof mount> | null = null;

afterEach(() => {
  if (instance) unmount(instance);
  instance = null;
  document.body.innerHTML = '';
});

async function render(
  ack: { warning: string | null; required: boolean } | undefined,
  activate = vi.fn().mockResolvedValue(activeFixture()),
  onCloseHook?: () => void,
  targetOverride?: { name: string; revision: number | null; validation: null },
) {
  const active = new ConfigActive({
    getActive: vi.fn().mockResolvedValue(activeFixture()),
    activate,
    rollback: vi.fn(),
  });
  await active.load();
  const onclose = vi.fn(() => onCloseHook?.());
  const onactivated = vi.fn();
  instance = mount(ConfigActivateDialog, {
    target: document.body,
    props: {
      ed: { active, dirty: false, viewing: null },
      target: targetOverride ?? { name: 'cloud_vlm', revision: 1, validation: null },
      ack,
      blurb: 'Runs use it.',
      onclose,
      onactivated,
    },
  });
  flushSync();
  return { active, activate, onclose, onactivated };
}

const q = (id: string) => document.querySelector<HTMLElement>(`[data-testid="${id}"]`);
const confirm = () =>
  [...document.querySelectorAll('[role="dialog"] button')]
    .find((b) => b.textContent?.trim() === 'Activate')!
    .dispatchEvent(new MouseEvent('click', { bubbles: true }));

describe('ConfigActivateDialog acknowledgement', () => {
  it('without ack: no banner, no checkbox, and no acknowledge_external in the body', async () => {
    const { activate, onclose } = await render(undefined);
    expect(q('activate-external-warning')).toBeNull();
    expect(q('activate-ack')).toBeNull();
    confirm();
    await vi.waitFor(() => expect(onclose).toHaveBeenCalled());
    expect(activate.mock.calls[0]![1]).toEqual({
      revision: 1,
      expected_active: { name: 'local_vlm', revision: 3 },
      force: false,
    });
  });

  it('shows the served warning, and the checkbox only when required', async () => {
    await render({ warning: 'Crops leave the deployment.', required: false });
    expect(q('activate-external-warning')?.textContent?.trim()).toBe(
      'Crops leave the deployment.',
    );
    expect(q('activate-ack')).toBeNull();
    unmount(instance!);
    document.body.innerHTML = '';
    await render({ warning: 'Crops leave the deployment.', required: true });
    expect(q('activate-ack')).not.toBeNull();
  });

  it('sends acknowledge_external: true only when the box is checked', async () => {
    const first = await render({ warning: 'W', required: true });
    confirm();
    await vi.waitFor(() => expect(first.activate).toHaveBeenCalledTimes(1));
    expect(first.activate.mock.calls[0]![1]).not.toHaveProperty('acknowledge_external');
    unmount(instance!);
    document.body.innerHTML = '';

    const second = await render({ warning: 'W', required: true });
    q('activate-ack')!.click();
    flushSync();
    confirm();
    await vi.waitFor(() => expect(second.activate).toHaveBeenCalledTimes(1));
    expect(second.activate.mock.calls[0]![1]).toMatchObject({
      acknowledge_external: true,
    });
  });

  it('after a served ack refusal the checkbox appears even when the list said not external', async () => {
    const activate = vi.fn().mockRejectedValueOnce(
      new ApiError(422, '/x', {
        detail: {
          error: 'vlm_external_not_acknowledged',
          message: 'Acknowledge it first.',
          endpoint: 'cloud_vlm',
        },
      }),
    );
    await render({ warning: null, required: false }, activate);
    expect(q('activate-ack')).toBeNull();
    confirm();
    await vi.waitFor(() => expect(q('activate-error')).not.toBeNull());
    expect(q('activate-error')?.textContent).toBe('Acknowledge it first.');
    expect(q('activate-ack')).not.toBeNull();
  });
});

describe('ConfigActivateDialog target lifetime', () => {
  it("hands onactivated the target it activated even when onclose invalidates the caller's source", async () => {
    // The page passes `target` as an expression over state that onclose() nulls;
    // reading it after onclose() threw "Cannot read properties of null".
    let open = true;
    const target = {
      get name(): string {
        if (!open) throw new TypeError("Cannot read properties of null (reading 'name')");
        return 'cloud_vlm';
      },
      revision: 1,
      validation: null,
    };
    const { onactivated } = await render(
      undefined,
      undefined,
      () => (open = false),
      target,
    );
    confirm();
    await vi.waitFor(() => expect(onactivated).toHaveBeenCalled());
    expect(onactivated.mock.calls[0]![0]).toMatchObject({
      name: 'cloud_vlm',
      revision: 1,
    });
  });
});
