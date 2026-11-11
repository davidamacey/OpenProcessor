/**
 * `POST /train/start` and `/train/start_campaign` take `force` (bypass blocking preflight
 * checks; the server still refuses the non-overridable ones). The form offers it only while the
 * served preflight is blocked, off by default, and sends it with the start call.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import TrainForm from './TrainForm.svelte';

let target: HTMLDivElement;
let instance: unknown;

function render(props: Record<string, unknown>): void {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(TrainForm, {
    target,
    props: {
      datasetExportDir: '/data/export',
      profiles: [],
      presets: [],
      preflight: null,
      preflighting: false,
      starting: false,
      onPreflight: () => {},
      onStart: () => {},
      onStartCampaign: () => {},
      ...props,
    },
  } as never);
  flushSync();
}

afterEach(() => {
  if (instance) unmount(instance as never);
  instance = undefined;
  target?.remove();
});

const BLOCKED = { blocked: true, checks: [], summary: 'blocked' };

function startButton(): HTMLButtonElement {
  return [...target.querySelectorAll('button')].find((b) =>
    /Start training/.test(b.textContent ?? ''),
  ) as HTMLButtonElement;
}

describe('TrainForm force', () => {
  it('has no bypass control while the preflight is clear', () => {
    render({ preflight: { blocked: false, checks: [] } });
    expect(target.querySelector('[data-testid="train-force"]')).toBeNull();
  });

  it('keeps Start disabled when blocked until the bypass is ticked, then sends force', () => {
    const onStart = vi.fn();
    render({ preflight: BLOCKED, onStart });
    expect(startButton().disabled).toBe(true);
    const box = target.querySelector('[data-testid="train-force"]') as HTMLInputElement;
    expect(box.checked).toBe(false);
    box.click();
    flushSync();
    expect(startButton().disabled).toBe(false);
    startButton().click();
    expect(onStart).toHaveBeenCalledTimes(1);
    expect(onStart.mock.calls[0][1]).toBe(true);
  });
});
