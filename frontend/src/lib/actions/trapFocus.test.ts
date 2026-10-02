import { afterEach, describe, expect, it, vi } from 'vitest';
import { trapFocus } from './trapFocus';

let root: HTMLDivElement | null = null;

afterEach(() => {
  root?.remove();
  root = null;
});

function buildDialog(): {
  outer: HTMLDivElement;
  input: HTMLInputElement;
  cancel: HTMLButtonElement;
  ok: HTMLButtonElement;
} {
  root = document.createElement('div');
  const trigger = document.createElement('button');
  trigger.textContent = 'open';
  root.appendChild(trigger);

  const outer = document.createElement('div');
  outer.tabIndex = -1;
  const input = document.createElement('input');
  const cancel = document.createElement('button');
  cancel.textContent = 'Cancel';
  const ok = document.createElement('button');
  ok.textContent = 'OK';
  outer.append(input, cancel, ok);
  root.appendChild(outer);

  const navLink = document.createElement('a');
  navLink.href = '#';
  navLink.textContent = 'nav (outside dialog)';
  root.appendChild(navLink);

  document.body.appendChild(root);
  trigger.focus();
  return { outer, input, cancel, ok };
}

function tab(target: HTMLElement, shift = false): void {
  target.dispatchEvent(
    new KeyboardEvent('keydown', {
      key: 'Tab',
      shiftKey: shift,
      bubbles: true,
      cancelable: true,
    }),
  );
}

describe('trapFocus action', () => {
  it('wraps Tab from the last focusable element back to the first, never reaching outside content', () => {
    const { outer, input, ok } = buildDialog();
    trapFocus(outer);
    outer.focus();

    input.focus();
    tab(outer); // input -> cancel
    tab(outer); // cancel -> ok
    expect(document.activeElement).not.toBe(ok); // jsdom doesn't auto-move focus on Tab

    // Simulate the browser having moved focus to `ok` already (jsdom does
    // not implement default Tab focus movement), then trap-check the wrap.
    ok.focus();
    tab(outer); // ok is last -> should wrap to input (first)
    expect(document.activeElement).toBe(input);
  });

  it('wraps Shift+Tab from the first element back to the last', () => {
    const { outer, input, ok } = buildDialog();
    trapFocus(outer);
    input.focus();
    tab(outer, true); // shift+tab from first -> last
    expect(document.activeElement).toBe(ok);
  });

  it('re-enters the dialog at the first element when the wrapper itself has focus (post focusOnMount)', () => {
    const { outer, input } = buildDialog();
    trapFocus(outer);
    outer.focus();
    tab(outer);
    expect(document.activeElement).toBe(input);
  });

  it('never lets Tab move focus onto content outside the dialog', () => {
    const { outer, ok } = buildDialog();
    const navLink = root!.querySelector('a')!;
    trapFocus(outer);
    ok.focus();
    tab(outer);
    expect(document.activeElement).not.toBe(navLink);
  });

  it('calls onEscape and prevents default when Escape is pressed', () => {
    const { outer } = buildDialog();
    const onEscape = vi.fn();
    trapFocus(outer, { onEscape });
    const evt = new KeyboardEvent('keydown', {
      key: 'Escape',
      bubbles: true,
      cancelable: true,
    });
    outer.dispatchEvent(evt);
    expect(onEscape).toHaveBeenCalledTimes(1);
    expect(evt.defaultPrevented).toBe(true);
  });

  it('restores focus to the previously-focused trigger on destroy', () => {
    const { outer } = buildDialog();
    const trigger = root!.querySelector('button')!;
    const handle = trapFocus(outer);
    outer.focus();
    handle.destroy();
    expect(document.activeElement).toBe(trigger);
  });
});

describe('trapFocus when focus has fallen out of the dialog', () => {
  // A focused Confirm button that turns `disabled` (busy) drops focus to
  // <body>; keys then never reach the dialog's own listener.
  it('still closes on Escape pressed with focus on <body>', () => {
    const { outer, ok } = buildDialog();
    const onEscape = vi.fn();
    const action = trapFocus(outer, { onEscape });
    ok.focus();
    ok.blur();
    expect(document.activeElement).toBe(document.body);

    document.body.dispatchEvent(
      new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }),
    );
    expect(onEscape).toHaveBeenCalledTimes(1);
    action.destroy();
  });

  it('pulls Tab from <body> back into the dialog', () => {
    const { outer, input } = buildDialog();
    const action = trapFocus(outer);
    (document.activeElement as HTMLElement | null)?.blur();

    document.body.dispatchEvent(
      new KeyboardEvent('keydown', { key: 'Tab', bubbles: true }),
    );
    expect(document.activeElement).toBe(input);
    action.destroy();
  });

  it('routes a stray Escape to the topmost dialog only, once', () => {
    const lower = buildDialog();
    const lowerEscape = vi.fn();
    const lowerAction = trapFocus(lower.outer, { onEscape: lowerEscape });
    const upperOuter = document.createElement('div');
    upperOuter.append(document.createElement('button'));
    document.body.appendChild(upperOuter);
    const upperEscape = vi.fn();
    const upperAction = trapFocus(upperOuter, { onEscape: upperEscape });
    (document.activeElement as HTMLElement | null)?.blur();

    document.body.dispatchEvent(
      new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }),
    );
    expect(upperEscape).toHaveBeenCalledTimes(1);
    expect(lowerEscape).not.toHaveBeenCalled();

    upperAction.destroy();
    upperOuter.remove();
    document.body.dispatchEvent(
      new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }),
    );
    expect(lowerEscape).toHaveBeenCalledTimes(1);
    lowerAction.destroy();
  });

  it('handles Escape inside the dialog exactly once', () => {
    const { outer, cancel } = buildDialog();
    const onEscape = vi.fn();
    const action = trapFocus(outer, { onEscape });
    cancel.focus();
    cancel.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    expect(onEscape).toHaveBeenCalledTimes(1);
    action.destroy();
  });

  it('stops listening on the document once destroyed', () => {
    const { outer } = buildDialog();
    const onEscape = vi.fn();
    trapFocus(outer, { onEscape }).destroy();
    document.body.dispatchEvent(
      new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }),
    );
    expect(onEscape).not.toHaveBeenCalled();
  });
});
