/**
 * The Export button follows the served `can_export` of `GET /export/status`
 * (the same pre-scan the export 422 uses) and lists the served
 * `blocking_reasons`; `null` never blocks and there is no client rule over
 * the class rows (an empty class list with `can_export: true` stays enabled).
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import ExportPage from './+page.svelte';

let target: HTMLDivElement;
let instance: unknown;

const json = (body: unknown) =>
  new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });

function stubFetch(status: Record<string, unknown>): void {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockImplementation((url: string) => {
      if (url.includes('/export/status')) return Promise.resolve(json(status));
      if (url.includes('/test_holdout/stats')) {
        return Promise.resolve(json({ total: 0, by_class: [], min_test_per_class: 5 }));
      }
      if (url.includes('/export/datasets')) {
        return Promise.resolve(json({ datasets: [], count: 0 }));
      }
      if (url.includes('/stats/classes')) return Promise.resolve(json({ classes: [] }));
      return Promise.resolve(json({}));
    }),
  );
}

async function mountPage(): Promise<void> {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(ExportPage, { target } as never);
  await vi.waitFor(() => {
    flushSync();
    expect(target.textContent).toContain('Last status');
  });
}

const exportButton = () =>
  [...target.querySelectorAll('button')].find((b) =>
    /^(Export|Re-export)$/.test(b.textContent?.trim() ?? ''),
  ) as HTMLButtonElement;

afterEach(() => {
  if (instance) unmount(instance as never);
  instance = undefined;
  target?.remove();
  vi.unstubAllGlobals();
});

describe('/export follows the served can_export', () => {
  it('can_export false disables Export and lists the served blocking reasons', async () => {
    stubFetch({
      status: 'idle',
      last_run: null,
      can_export: false,
      blocking_reasons: [
        'nothing to export: 0 items are class_validated',
        'second reason',
      ],
    });
    await mountPage();
    expect(exportButton().disabled).toBe(true);
    const list = target.querySelector('[data-testid="export-blocking-reasons"]');
    expect(list?.textContent).toContain('nothing to export: 0 items are class_validated');
    expect(list?.textContent).toContain('second reason');
  });

  it('can_export null never blocks and shows nothing guessed', async () => {
    stubFetch({ status: 'idle', last_run: null, can_export: null, blocking_reasons: [] });
    await mountPage();
    expect(exportButton().disabled).toBe(false);
    expect(target.querySelector('[data-testid="export-blocking-reasons"]')).toBeNull();
  });

  it('can_export true stays enabled even with no class rows (no client rule)', async () => {
    stubFetch({ status: 'idle', last_run: null, can_export: true, blocking_reasons: [] });
    await mountPage();
    expect(exportButton().disabled).toBe(false);
  });
});
