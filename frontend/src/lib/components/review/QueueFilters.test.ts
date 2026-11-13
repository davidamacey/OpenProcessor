/**
 * Below `lg` the queue's filter fields start folded behind one button, so Review opens on the
 * picture at 430px; the button opens and folds them. (`lg:` is a CSS media query: jsdom checks the
 * state classes, the desktop layout is the e2e's.)
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import QueueFilters from './QueueFilters.svelte';

vi.mock('$lib/components/itemFilter/ServedFilterField.svelte', () => ({
  default: () => {},
}));

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | undefined;

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

function render(specs: { param: string }[]) {
  target = document.createElement('div');
  document.body.appendChild(target);
  const tabs = { queueFilterSpecs: specs, queueFilterValues: {} };
  instance = mount(QueueFilters, { target, props: { ws: {}, tabs } as never });
  flushSync();
}

describe('QueueFilters', () => {
  it('starts folded, counts the fields, and opens on the button', () => {
    render([{ param: 'a' }, { param: 'b' }]);
    const toggle = target.querySelector<HTMLButtonElement>(
      '[data-testid="queue-filters-toggle"]',
    )!;
    const fields = target.querySelector('[data-testid="queue-filters"]')!;
    expect(toggle.textContent).toContain('(2)');
    expect(toggle.getAttribute('aria-expanded')).toBe('false');
    expect(fields.classList.contains('hidden')).toBe(true);
    toggle.click();
    flushSync();
    expect(toggle.getAttribute('aria-expanded')).toBe('true');
    expect(fields.classList.contains('hidden')).toBe(false);
  });

  it('renders nothing when the queue serves no filter', () => {
    render([]);
    expect(target.querySelector('[data-testid="queue-filters"]')).toBeNull();
    expect(target.querySelector('[data-testid="queue-filters-toggle"]')).toBeNull();
  });
});
