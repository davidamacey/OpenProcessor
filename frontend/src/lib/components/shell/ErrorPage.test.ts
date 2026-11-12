import { describe, expect, it, vi } from 'vitest';
import { mountUi } from '$lib/test/mountUi';

const state = vi.hoisted(() => ({
  page: { status: 404, error: { message: 'Not found' } as { message: string } | null },
}));
vi.mock('$app/state', () => state);

import ErrorPage from './ErrorPage.svelte';

describe('ErrorPage inside a project', () => {
  it('a 404 keeps the project: the way out is its dashboard, then the project list', () => {
    state.page.status = 404;
    const el = mountUi(ErrorPage, { inShell: true });
    expect(el.querySelector('h1')!.textContent).toBe('Page not found');
    expect(el.querySelector('a.btn-primary')!.getAttribute('href')).toBe(
      '/p/default/dashboard',
    );
    expect(el.querySelector('a[href$="/projects"]')).not.toBeNull();
    // Inside the shell it fills the pane under the bar instead of the whole screen.
    expect(el.firstElementChild!.className).toContain('min-h-full');
    expect(el.firstElementChild!.className).not.toContain('h-screen');
  });

  it('the app-root page is full screen and offers only the project list', () => {
    state.page.status = 404;
    const el = mountUi(ErrorPage, { inShell: false });
    expect(el.firstElementChild!.className).toContain('h-screen');
    expect(el.querySelectorAll('a').length).toBe(1);
  });
});
