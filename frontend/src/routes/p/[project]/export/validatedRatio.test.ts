/**
 * /export shows the served validated-vs-total counts (`GET /stats/dataset`)
 * and warns that only validated crops are exported (#119).
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import ExportPage from './+page.svelte';

let target: HTMLDivElement;
let instance: unknown;

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });

function stubFetch(dataset: () => Response): void {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockImplementation((url: string) => {
      if (url.includes('/stats/dataset')) return Promise.resolve(dataset());
      if (url.includes('/export/status')) {
        return Promise.resolve(json({ status: 'idle', last_run: null }));
      }
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
    expect(target.textContent).toContain('Test holdout');
  });
}

const notice = () =>
  target.querySelector('[data-testid="validated-ratio"]') as HTMLElement | null;

afterEach(() => {
  if (instance) unmount(instance as never);
  instance = undefined;
  target?.remove();
  vi.unstubAllGlobals();
});

describe('/export validated ratio', () => {
  it('warns with the served counts when most crops are machine labels', async () => {
    stubFetch(() =>
      json({ total_crops: 14023, validated: 120, test_holdout: 0, by_source: [] }),
    );
    await mountPage();
    await vi.waitFor(() => {
      flushSync();
      expect(notice()).not.toBeNull();
    });
    expect(notice()!.dataset.level).toBe('partial');
    expect(notice()!.textContent).toContain(
      `${(120).toLocaleString()} of ${(14023).toLocaleString()}`,
    );
  });

  it('is absent when the dataset stats could not be read (nothing is guessed)', async () => {
    stubFetch(() => json({ detail: 'region_status not aggregatable' }, 503));
    await mountPage();
    await vi.waitFor(
      () => {
        flushSync();
        expect(target.textContent).toContain('Dataset totals unavailable');
      },
      { timeout: 8000 },
    );
    expect(notice()).toBeNull();
  });
});
