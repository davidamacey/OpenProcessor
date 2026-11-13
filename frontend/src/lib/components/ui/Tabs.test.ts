import { describe, expect, it, vi } from 'vitest';
import { flushSync } from 'svelte';
import { mountUi } from '$lib/test/mountUi';
import Tabs from './Tabs.svelte';

const route = [
  { id: 'audit', label: 'Audit', href: '/p/default/quality/audit' },
  { id: 'history', label: 'History', href: '/p/default/quality/history', count: 4 },
];
const inPage = [
  { id: 'a', label: 'One' },
  { id: 'b', label: 'Two' },
  { id: 'c', label: 'Three', disabled: true },
];

describe('Tabs as route tabs', () => {
  it('are links in a nav, the active one aria-current and underlined in the accent', () => {
    const el = mountUi(Tabs, {
      items: route,
      value: 'history',
      label: 'Quality sections',
    });
    const links = el.querySelectorAll('nav[aria-label="Quality sections"] a');
    expect(links.length).toBe(2);
    expect(links[0]!.getAttribute('href')).toBe('/p/default/quality/audit');
    expect(links[0]!.getAttribute('aria-current')).toBeNull();
    expect(links[1]!.getAttribute('aria-current')).toBe('page');
    expect(links[1]!.className).toContain('border-accent');
    expect(links[1]!.textContent).toContain('4');
  });
});

describe('Tabs in a page', () => {
  it('are a tablist; only the selected tab is a tab stop, and a click selects', () => {
    const onselect = vi.fn();
    const el = mountUi(Tabs, { items: inPage, value: 'a', label: 'Sections', onselect });
    const tabs = el.querySelectorAll<HTMLButtonElement>('[role="tab"]');
    expect(tabs[0]!.getAttribute('aria-selected')).toBe('true');
    expect(tabs[0]!.tabIndex).toBe(0);
    expect(tabs[1]!.tabIndex).toBe(-1);
    tabs[1]!.click();
    expect(onselect).toHaveBeenCalledWith('b');
  });

  it('arrow keys move to the next enabled tab and wrap, skipping a disabled one', () => {
    const onselect = vi.fn();
    const el = mountUi(Tabs, { items: inPage, value: 'b', label: 'Sections', onselect });
    const list = el.querySelector('[role="tablist"]')!;
    list.dispatchEvent(
      new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }),
    );
    flushSync();
    expect(onselect).toHaveBeenLastCalledWith('a');
    list.dispatchEvent(new KeyboardEvent('keydown', { key: 'End', bubbles: true }));
    expect(onselect).toHaveBeenLastCalledWith('b');
  });

  it('marks the active tab for the strip to scroll into view', () => {
    const el = mountUi(Tabs, { items: inPage, value: 'b', label: 'Sections' });
    const tabs = el.querySelectorAll<HTMLElement>('[role="tab"]');
    expect(tabs[0]!.getAttribute('data-active')).toBeNull();
    expect(tabs[1]!.getAttribute('data-active')).toBe('true');
  });
});

describe('Tabs on a narrow screen', () => {
  const widths = { clientWidth: 100, scrollWidth: 300 };
  const stub = (name: 'clientWidth' | 'scrollWidth') =>
    Object.defineProperty(HTMLElement.prototype, name, {
      configurable: true,
      get: () => widths[name],
    });
  it('show a "more" button when tabs are cut off, for both tab kinds', () => {
    stub('clientWidth');
    stub('scrollWidth');
    try {
      for (const items of [route, inPage]) {
        const el = mountUi(Tabs, { items, value: items[0]!.id, label: 'Sections' });
        flushSync();
        expect(
          el.querySelector('[data-testid="tabs-strip-more-right"]'),
          items === route ? 'route tabs' : 'in-page tabs',
        ).not.toBeNull();
      }
    } finally {
      delete (HTMLElement.prototype as Partial<HTMLElement>).clientWidth;
      delete (HTMLElement.prototype as Partial<HTMLElement>).scrollWidth;
    }
  });
});
