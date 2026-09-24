/**
 * m12 (2026-09-24 interactive pass): the Add Class modal didn't trap Tab
 * focus, so tabbing through it escaped into the nav links behind it, and
 * mounts the real component (Svelte 5 mount/flushSync, matches
 * CropCard.test.ts's pattern) rather than scanning source text.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import AddClassModal from './AddClassModal.svelte';

let target: HTMLDivElement;
let instance: unknown;
let outsideLink: HTMLAnchorElement;

function render(onclose: () => void) {
  outsideLink = document.createElement('a');
  outsideLink.href = '#';
  outsideLink.textContent = 'nav link outside the modal';
  document.body.appendChild(outsideLink);

  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(AddClassModal, { target, props: { open: true, onclose } } as never);
  flushSync();
  return target;
}

afterEach(() => {
  if (instance) unmount(instance as never);
  target?.remove();
  outsideLink?.remove();
  instance = undefined as unknown;
});

function tab(node: Element, shift = false): void {
  node.dispatchEvent(
    new KeyboardEvent('keydown', {
      key: 'Tab',
      shiftKey: shift,
      bubbles: true,
      cancelable: true,
    }),
  );
}

describe('AddClassModal focus trap (m12)', () => {
  it('never lets Tab move focus onto the nav link behind the modal', () => {
    render(() => {});
    const dialog = target.querySelector('[role="dialog"]') as HTMLElement;
    const buttons = target.querySelectorAll('button');
    const cancelBtn = Array.from(buttons).find(
      (b) => b.textContent?.trim() === 'Cancel',
    )!;
    cancelBtn.focus();
    tab(dialog); // last focusable (submit is disabled while name is empty) -> should wrap, not escape

    expect(document.activeElement).not.toBe(outsideLink);
    expect(target.contains(document.activeElement)).toBe(true);
  });

  it('closes on Escape', () => {
    const onclose = vi.fn();
    render(onclose);
    const dialog = target.querySelector('[role="dialog"]') as HTMLElement;
    dialog.dispatchEvent(
      new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }),
    );
    expect(onclose).toHaveBeenCalledTimes(1);
  });
});
