import { describe, expect, it, vi } from 'vitest';
import { flushSync } from 'svelte';
import { mountUi, snip } from '$lib/test/mountUi';
import Drawer from './Drawer.svelte';
import Menu from './Menu.svelte';

const items = [
  { id: 'a', label: 'Docs', href: '/p/default/dashboard' },
  { id: 'b', label: 'Delete', tone: 'danger' as const, onselect: vi.fn() },
  { id: 'c', label: 'Off', disabled: true, onselect: vi.fn() },
];

const key = (node: Element, k: string) =>
  node.dispatchEvent(
    new KeyboardEvent('keydown', { key: k, bubbles: true, cancelable: true }),
  );

describe('Drawer', () => {
  it('is a labelled modal dialog at the drawer layer, flush to its edge, and Escape closes it', () => {
    const onclose = vi.fn();
    mountUi(Drawer, {
      label: 'Menu',
      onclose,
      side: 'right',
      children: snip('<p>body</p>'),
    });
    const dialog = document.querySelector('[data-testid="drawer"]') as HTMLElement;
    expect(dialog.getAttribute('role')).toBe('dialog');
    expect(dialog.getAttribute('aria-modal')).toBe('true');
    expect(dialog.getAttribute('aria-label')).toBe('Menu');
    expect(dialog.className).toContain('z-50');
    expect(dialog.className).toContain('justify-end');
    expect(dialog.textContent).toContain('body');
    key(dialog, 'Escape');
    expect(onclose).toHaveBeenCalledOnce();
  });

  it('has a close button', () => {
    const onclose = vi.fn();
    mountUi(Drawer, { label: 'Menu', onclose, children: snip('<p>body</p>') });
    (
      document.querySelector(
        '[data-testid="drawer"] button[aria-label="Close"]',
      ) as HTMLElement
    ).click();
    expect(onclose).toHaveBeenCalledOnce();
  });
});

describe('Menu', () => {
  function open() {
    const el = mountUi(Menu, { label: 'Actions', items, current: 'a' });
    const trigger = el.querySelector('button[aria-haspopup]') as HTMLButtonElement;
    return { el, trigger };
  }

  it('is closed until its trigger is pressed, which reports aria-expanded', () => {
    const { el, trigger } = open();
    expect(trigger.getAttribute('aria-expanded')).toBe('false');
    expect(el.querySelector('[role="menu"]')).toBeNull();
    trigger.click();
    flushSync();
    expect(trigger.getAttribute('aria-expanded')).toBe('true');
    const menu = el.querySelector('[role="menu"]')!;
    expect(menu.querySelectorAll('[role="menuitem"]').length).toBe(3);
  });

  it('renders a link item as an anchor marked current, an action item as a button', () => {
    const { el, trigger } = open();
    trigger.click();
    flushSync();
    const link = el.querySelector('a[role="menuitem"]')!;
    expect(link.getAttribute('href')).toBe('/p/default/dashboard');
    expect(link.getAttribute('aria-current')).toBe('page');
    expect(el.querySelector('button[role="menuitem"][disabled]')).not.toBeNull();
  });

  it('an action item runs and closes the menu; Escape closes and returns focus', () => {
    const { el, trigger } = open();
    trigger.click();
    flushSync();
    (el.querySelectorAll('[role="menuitem"]')[1] as HTMLElement).click();
    flushSync();
    expect(items[1]!.onselect).toHaveBeenCalledOnce();
    expect(el.querySelector('[role="menu"]')).toBeNull();

    trigger.click();
    flushSync();
    key(el.querySelector('[role="menu"]')!, 'Escape');
    flushSync();
    expect(el.querySelector('[role="menu"]')).toBeNull();
    expect(document.activeElement).toBe(trigger);
  });

  it('a press outside closes it', () => {
    const { el, trigger } = open();
    trigger.click();
    flushSync();
    document.body.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }));
    flushSync();
    expect(el.querySelector('[role="menu"]')).toBeNull();
  });

  it('arrow keys move focus between enabled items only', () => {
    const { el, trigger } = open();
    trigger.click();
    flushSync();
    const menu = el.querySelector('[role="menu"]')!;
    const links = el.querySelectorAll<HTMLElement>('[role="menuitem"]:not([disabled])');
    links[0]!.focus();
    key(menu, 'ArrowDown');
    expect(document.activeElement).toBe(links[1]);
    key(menu, 'ArrowDown');
    expect(document.activeElement).toBe(links[0]);
    key(menu, 'End');
    expect(document.activeElement).toBe(links[1]);
  });
});
