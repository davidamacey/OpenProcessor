/**
 * Mount test for `AutoLabelPanel`'s additive `gate` prop
 * (docs/design/ingest-ui-and-acceptance-plan-2026-09-24.md §A.5): with
 * no gate, the start button is enabled; with a blocked gate, it's
 * disabled and the reason renders.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import AutoLabelPanel from './AutoLabelPanel.svelte';

vi.mock('$lib/api', async () => {
  const actual = await vi.importActual<typeof import('$lib/api')>('$lib/api');
  return {
    ...actual,
    getAutoLabelStatus: vi.fn(async () => ({
      status: 'idle',
      stage: '',
      progress: null,
    })),
  };
});

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | undefined;

beforeEach(() => {
  target = document.createElement('div');
  document.body.appendChild(target);
});

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target.remove();
  vi.restoreAllMocks();
});

describe('AutoLabelPanel gate prop', () => {
  it('renders the start button enabled with no gate', () => {
    instance = mount(AutoLabelPanel, { target, props: {} });
    flushSync();
    const btn = [...target.querySelectorAll('button')].find((b) =>
      b.textContent?.includes('Recluster now'),
    );
    expect(btn).toBeTruthy();
    expect(btn?.disabled).toBe(false);
  });

  it('disables the start button and renders the reason when gate.blocked', () => {
    instance = mount(AutoLabelPanel, {
      target,
      props: { gate: { blocked: true, reason: 'Finish or cancel the upload first' } },
    });
    flushSync();
    const btn = [...target.querySelectorAll('button')].find((b) =>
      b.textContent?.includes('Recluster now'),
    );
    expect(btn?.disabled).toBe(true);
    expect(target.textContent).toContain('Finish or cancel the upload first');
  });
});
