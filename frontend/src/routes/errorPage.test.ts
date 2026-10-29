/**
 * Root error page: an unknown URL (404) is not fixed by reloading, so it
 * offers the way out (the project list) instead of the generic reload
 * prompt; any other failure keeps the reload prompt and its details.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';

const state = vi.hoisted(() => ({
  page: { status: 500, error: { message: 'boom' } as { message: string } | null },
}));
vi.mock('$app/state', () => state);

import ErrorPage from './+error.svelte';

let instance: ReturnType<typeof mount> | null = null;
let target: HTMLDivElement;

function render(status: number, message: string): HTMLElement {
  state.page.status = status;
  state.page.error = { message };
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(ErrorPage, { target });
  flushSync();
  return target;
}

afterEach(() => {
  if (instance) unmount(instance);
  instance = null;
  target?.remove();
});

describe('+error.svelte', () => {
  it('a 404 says the page was not found and links to the project list', () => {
    const el = render(404, 'Not Found');
    expect(el.textContent).toContain('Page not found');
    expect(el.textContent).not.toContain('Reloading may help');
    expect(el.querySelector('a[href$="/projects"]')).not.toBeNull();
  });

  it('any other failure keeps the reload prompt and the details', () => {
    const el = render(500, 'boom');
    expect(el.textContent).toContain('Something went wrong');
    expect(el.textContent).toContain('Reload');
    expect(el.querySelector('[data-testid="app-error-detail"]')?.textContent).toContain(
      'boom',
    );
  });
});
