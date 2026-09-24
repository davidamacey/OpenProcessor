/**
 * Mount-based coverage for the OpenProcessor 6c77deb `/export` adoption:
 *
 *   1. `GET {API_PREFIX}/export/status` now serves `image_count`/
 *      `class_count`/`split_counts`/`class_split_counts` — the page must
 *      render them, with a 0-train/0-val class row highlighted, using the
 *      served numbers only (no client threshold).
 *   2. `POST {API_PREFIX}/test_holdout/freeze`'s body is `{percent}` only
 *      — the Seed field is gone from the freeze modal, and the request
 *      actually sent carries no `seed` key.
 *
 * Real Svelte 5 mount (`mount`/`unmount`/`flushSync` under jsdom, no
 * `@testing-library/svelte` — same harness as
 * `TrainForm.gpuPicker.test.ts`). `fetch` is routed by URL since the page
 * fires several requests in parallel from `loadAll()`.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import ExportPage from './+page.svelte';
import { toastStore } from '$stores/toast.svelte';

let target: HTMLDivElement;
let instance: unknown;

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

async function flushMicrotasks(): Promise<void> {
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));
}

const EXPORT_STATUS_SUCCESS = {
  status: 'success',
  last_run: '2026-09-24T12:00:00Z',
  export_dir: '/data/export/yolo/v1',
  path: '/data/export/yolo/v1',
  version_tag: 'v1',
  dataset_sha: 'abc123',
  seed: null,
  group_key: 'image_id',
  image_count: 500,
  class_count: 2,
  split_counts: { train: 400, val: 80, test: 20 },
  class_split_counts: [
    { class_id: 1, export_id: 0, class_name: 'bmw', train: 200, val: 40, test: 10 },
    { class_id: 2, export_id: 1, class_name: 'audi', train: 0, val: 40, test: 10 },
  ],
};

function makeFetchMock(
  freezeCalls: Array<{ url: string; body: unknown }>,
): ReturnType<typeof vi.fn> {
  return vi.fn().mockImplementation((url: string, init?: RequestInit) => {
    if (url.includes('/test_holdout/freeze')) {
      freezeCalls.push({
        url,
        body: init?.body ? JSON.parse(init.body as string) : null,
      });
      return Promise.resolve(
        jsonResponse({
          n_frozen: 42,
          n_classes_covered: 2,
          test_holdout_sha: 'deadbeef',
          per_class_counts: { '1': 21, '2': 21 },
          selection: 'sha1_per_class',
          percent: 10,
          min_per_class: 5,
        }),
      );
    }
    if (url.includes('/export/status')) {
      return Promise.resolve(jsonResponse(EXPORT_STATUS_SUCCESS));
    }
    if (url.includes('/test_holdout/stats')) {
      return Promise.resolve(jsonResponse({ total: 0, by_class: [] }));
    }
    if (url.includes('/export/datasets')) {
      return Promise.resolve(jsonResponse({ datasets: [], count: 0 }));
    }
    if (url.includes('/stats/dataset')) {
      return Promise.resolve(
        jsonResponse({ total_crops: 0, validated: 0, test_holdout: 0, by_source: [] }),
      );
    }
    if (url.includes('/stats/classes')) {
      return Promise.resolve(jsonResponse({ classes: [] }));
    }
    return Promise.resolve(jsonResponse({}));
  });
}

afterEach(() => {
  if (instance) {
    unmount(instance as never);
    instance = undefined;
  }
  target?.remove();
  vi.unstubAllGlobals();
  toastStore.toasts = [];
});

describe('/export — served split counts (OpenProcessor 6c77deb)', () => {
  it('renders image/class totals, split totals, and a per-class table highlighting 0-train/0-val classes', async () => {
    const freezeCalls: Array<{ url: string; body: unknown }> = [];
    vi.stubGlobal('fetch', makeFetchMock(freezeCalls));

    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(ExportPage, { target } as never);
    flushSync();
    await flushMicrotasks();
    flushSync();

    expect(target.textContent).toContain('500');
    expect(target.textContent).toContain('by image_id');
    expect(target.textContent).toContain('train 400');
    expect(target.textContent).toContain('val 80');
    expect(target.textContent).toContain('test 20');

    const summary = Array.from(target.querySelectorAll('summary')).find((s) =>
      s.textContent?.includes('Per-class split counts'),
    );
    expect(summary).toBeTruthy();
    summary?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    (summary?.closest('details') as HTMLDetailsElement).open = true;
    flushSync();

    expect(target.textContent).toContain('audi');
    const audiRow = Array.from(target.querySelectorAll('tr')).find((tr) =>
      tr.textContent?.includes('audi'),
    );
    expect(audiRow?.className).toContain('bg-red-500/10');

    const bmwRow = Array.from(target.querySelectorAll('tr')).find((tr) =>
      tr.textContent?.includes('bmw'),
    );
    expect(bmwRow?.className ?? '').not.toContain('bg-red-500/10');
  });
});

describe('/export — freeze modal (OpenProcessor 6c77deb: no seed)', () => {
  it('has no Seed field and POSTs a body of exactly {percent}', async () => {
    const freezeCalls: Array<{ url: string; body: unknown }> = [];
    vi.stubGlobal('fetch', makeFetchMock(freezeCalls));
    vi.stubGlobal('confirm', vi.fn().mockReturnValue(true));

    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(ExportPage, { target } as never);
    flushSync();
    await flushMicrotasks();
    flushSync();

    const freezeButton = Array.from(target.querySelectorAll('button')).find(
      (b) => b.textContent?.trim() === 'Freeze test set',
    );
    expect(freezeButton).toBeTruthy();
    freezeButton?.click();
    flushSync();

    // No Seed field anywhere in the modal.
    const labels = Array.from(target.querySelectorAll('label')).map((l) => l.textContent);
    expect(labels.some((l) => l?.includes('Seed'))).toBe(false);
    expect(target.querySelector('input[type="number"]')).toBeTruthy();

    const submit = Array.from(target.querySelectorAll('button')).find(
      (b) => b.textContent?.trim() === 'Freeze',
    );
    expect(submit).toBeTruthy();
    submit?.click();
    flushSync();
    await flushMicrotasks();
    flushSync();

    expect(freezeCalls.length).toBeGreaterThan(0);
    const call = freezeCalls[0]!;
    expect(call.body).toEqual({ percent: 10 });
    expect(call.url).not.toContain('seed');

    const successToast = toastStore.toasts.find((t) => t.kind === 'success');
    expect(successToast?.text).toContain('sha1_per_class');
  });
});
