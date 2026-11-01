import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { assistRun } from '$lib/test/fixtures/assist';

vi.mock('$lib/api_assist', () => ({
  createAssistRun: vi.fn(async () => assistRun('dry_run')),
  getAssistRun: vi.fn(),
  cancelAssistRun: vi.fn(),
  applyAssistRun: vi.fn(),
}));

import AssistButton from '$components/assistRun/AssistButton.svelte';
import { createAssistRun } from '$lib/api_assist';

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | undefined;

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
  document.body.innerHTML = '';
  vi.mocked(createAssistRun).mockClear();
  vi.useRealTimers();
});

const q = (id: string) =>
  document.querySelector(`[data-testid="${id}"]`) as HTMLElement | null;

describe('AssistButton', () => {
  it('opens the dialog on a single box and plans a boxes request with only its own actions', async () => {
    // The dialog plans after a 250 ms debounce; fake timers keep the test off the host's clock.
    vi.useFakeTimers();
    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(AssistButton, {
      target,
      props: {
        opener: {
          title: 'This box',
          selected: { kind: 'boxes', boxes: [{ crop_id: 'c1', box_id: 'b1' }] },
        },
      },
    });
    flushSync();
    expect(q('assist-dialog')).toBeNull();
    (q('assist-button') as HTMLButtonElement).click();
    flushSync();
    expect(q('assist-dialog')).not.toBeNull();
    expect(q('assist-action-classify')).toBeNull();
    expect(q('assist-action-segment')).toBeNull();
    expect(q('assist-action-verify')).not.toBeNull();
    expect(createAssistRun).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(250);
    expect(createAssistRun).toHaveBeenCalledTimes(1);
    expect(vi.mocked(createAssistRun).mock.calls[0]![0]).toMatchObject({
      target: { kind: 'boxes', boxes: [{ crop_id: 'c1', box_id: 'b1' }] },
      actions: ['verify'],
      dry_run: true,
    });
  });
});
