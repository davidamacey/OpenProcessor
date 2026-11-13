import { afterEach, describe, expect, it } from 'vitest';
import { gridArrowKeys } from './gridArrowKeys';

let host: HTMLElement;
let off: { destroy?: () => void } | void;

/** Six cards in rows of three: jsdom has no layout, so `offsetTop` is stubbed per card. */
function grid(): HTMLElement[] {
  host = document.createElement('ul');
  const cards = Array.from({ length: 6 }, (_, i) => {
    const b = document.createElement('button');
    b.dataset.gridItem = '';
    b.tabIndex = i === 0 ? 0 : -1;
    Object.defineProperty(b, 'offsetTop', { value: i < 3 ? 0 : 100 });
    host.append(b);
    return b;
  });
  document.body.append(host);
  off = gridArrowKeys(host);
  return cards;
}

const press = (el: HTMLElement, key: string) =>
  el.dispatchEvent(
    new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true }),
  );

afterEach(() => {
  off?.destroy?.();
  host.remove();
});

describe('gridArrowKeys', () => {
  it('Right and Left step one card, Down and Up one row', () => {
    const c = grid();
    c[0]!.focus();
    press(c[0]!, 'ArrowRight');
    expect(document.activeElement).toBe(c[1]);
    press(c[1]!, 'ArrowDown');
    expect(document.activeElement).toBe(c[4]);
    press(c[4]!, 'ArrowLeft');
    expect(document.activeElement).toBe(c[3]);
    press(c[3]!, 'ArrowUp');
    expect(document.activeElement).toBe(c[0]);
  });

  it('stops at the ends and Home/End jump', () => {
    const c = grid();
    c[0]!.focus();
    press(c[0]!, 'ArrowLeft');
    expect(document.activeElement).toBe(c[0]);
    press(c[0]!, 'End');
    expect(document.activeElement).toBe(c[5]);
    press(c[5]!, 'ArrowDown');
    expect(document.activeElement).toBe(c[5]);
    press(c[5]!, 'Home');
    expect(document.activeElement).toBe(c[0]);
  });

  it('roves the tab stop to the focused card', () => {
    const c = grid();
    c[0]!.focus();
    press(c[0]!, 'ArrowRight');
    expect(c.map((el) => el.tabIndex)).toEqual([-1, 0, -1, -1, -1, -1]);
  });

  it('leaves a key with a modifier to the app', () => {
    const c = grid();
    c[0]!.focus();
    const e = new KeyboardEvent('keydown', {
      key: 'ArrowRight',
      shiftKey: true,
      bubbles: true,
      cancelable: true,
    });
    c[0]!.dispatchEvent(e);
    expect(e.defaultPrevented).toBe(false);
    expect(document.activeElement).toBe(c[0]);
  });
});
