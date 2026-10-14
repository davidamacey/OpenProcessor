import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import Harness from './ModalHarness.test.svelte';

let instance: ReturnType<typeof mount> | undefined;
let target: HTMLElement | undefined;
let opener: HTMLButtonElement | undefined;

function render(props: Record<string, unknown> = {}) {
  opener = document.createElement('button');
  opener.textContent = 'open';
  document.body.appendChild(opener);
  opener.focus();
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(Harness, { target, props } as never);
  flushSync();
  return target;
}

afterEach(() => {
  if (instance) unmount(instance);
  target?.remove();
  opener?.remove();
  instance = undefined;
  target = undefined;
  opener = undefined;
});

const dialog = () => target!.querySelector('[role="dialog"]') as HTMLElement;
const key = (node: Element, k: string, shift = false) =>
  node.dispatchEvent(
    new KeyboardEvent('keydown', {
      key: k,
      shiftKey: shift,
      bubbles: true,
      cancelable: true,
    }),
  );

describe('Modal', () => {
  it('is an accessible modal dialog with its label and passes data attributes through', () => {
    render();
    const d = dialog();
    expect(d.getAttribute('aria-modal')).toBe('true');
    expect(d.getAttribute('aria-label')).toBe('Harness dialog');
    expect(d.getAttribute('data-testid')).toBe('harness');
  });

  it('moves focus into the dialog and returns it to the opener when it closes', async () => {
    render();
    await Promise.resolve();
    expect(document.activeElement).toBe(dialog());
    unmount(instance!);
    instance = undefined;
    expect(document.activeElement).toBe(opener);
  });

  it('closes on Escape', () => {
    const onclose = vi.fn();
    render({ onclose });
    key(dialog(), 'Escape');
    expect(onclose).toHaveBeenCalledTimes(1);
  });

  it('closes on a backdrop click but not on a click inside the panel', () => {
    const onclose = vi.fn();
    render({ onclose });
    (target!.querySelector('#first') as HTMLElement).click();
    expect(onclose).not.toHaveBeenCalled();
    (target!.querySelector('[data-modal-backdrop]') as HTMLElement).click();
    expect(onclose).toHaveBeenCalledTimes(1);
  });

  it('does nothing on Escape or a backdrop click when there is no onclose', () => {
    render();
    key(dialog(), 'Escape');
    (target!.querySelector('[data-modal-backdrop]') as HTMLElement).click();
    expect(dialog()).not.toBeNull();
  });

  it('keeps Tab inside the dialog in both directions', () => {
    render();
    const first = target!.querySelector('#first') as HTMLElement;
    const last = target!.querySelector('#last') as HTMLElement;
    last.focus();
    key(last, 'Tab');
    expect(document.activeElement).toBe(first);
    key(first, 'Tab', true);
    expect(document.activeElement).toBe(last);
  });
});
