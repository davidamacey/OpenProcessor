import { afterEach, describe, expect, it, vi } from 'vitest';
import { createRawSnippet, flushSync, mount, unmount } from 'svelte';
vi.mock('$lib/projectPaths', () => ({ projectHref: (p: string) => `/p/x${p}` }));

import ViewState from './ViewState.svelte';

let instance: ReturnType<typeof mount> | null = null;
let target: HTMLDivElement;

const body = createRawSnippet(() => ({
  render: () => '<p data-testid="body">cards</p>',
}));

function render(props: Record<string, unknown>): HTMLElement {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(ViewState, { target, props: { children: body, ...props } } as never);
  flushSync();
  return target;
}

afterEach(() => {
  if (instance) unmount(instance);
  instance = null;
  target?.remove();
});

describe('ViewState', () => {
  it('renders its children when ready', () => {
    const el = render({ status: 'ready' });
    expect(el.querySelector('[data-testid="body"]')).not.toBeNull();
  });

  it('shows a skeleton and no children while loading', () => {
    const el = render({ status: 'loading' });
    expect(el.querySelector('[data-testid="view-state-loading"]')).not.toBeNull();
    expect(el.querySelector('[data-testid="body"]')).toBeNull();
  });

  it('shows the served empty message and its one action', () => {
    const el = render({
      status: 'empty',
      message: 'No clusters yet. Run clustering first.',
      action: { label: 'Open Ingest', path: '/ingest' },
    });
    expect(el.textContent).toContain('No clusters yet. Run clustering first.');
    expect(el.querySelector('a')?.getAttribute('href')).toBe('/p/x/ingest');
    expect(el.querySelector('[data-testid="body"]')).toBeNull();
  });

  it('shows the error text with Retry', () => {
    const retry = vi.fn();
    const el = render({ status: 'error', message: 'Could not load.', onretry: retry });
    expect(el.querySelector('[role="alert"]')?.textContent).toContain('Could not load.');
    el.querySelector('button')!.click();
    expect(retry).toHaveBeenCalledTimes(1);
  });
});
