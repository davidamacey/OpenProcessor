/**
 * Mount-based coverage for the OpenProcessor df01309 + 4c9499a `/export`
 * adoption:
 *
 *   1. `GET {API_PREFIX}/export/status` serves `image_count`/`class_count`/
 *      `split_counts`/`class_split_counts` (df01309) plus `object_count`/
 *      `split_object_counts`/`require_fully_labeled_images`/partial-frame
 *      counts (4c9499a — one image + one label file per source image) —
 *      the page must render them, with a 0-train/0-val class row
 *      highlighted, using the served numbers only (no client threshold),
 *      and render a `null` count (an export written before it was
 *      recorded) as "—", never 0.
 *   2. `POST {API_PREFIX}/test_holdout/freeze`'s body is `{percent}` only
 *      — the Seed field is gone from the freeze modal, and the request
 *      actually sent carries no `seed` key.
 *   3. `POST {API_PREFIX}/export/yolo` accepts an opt-in
 *      `require_fully_labeled_images` — the export modal's checkbox
 *      must actually send it.
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
  object_count: 620,
  class_count: 2,
  split_counts: { train: 400, val: 80, test: 20 },
  split_object_counts: { train: 500, val: 100, test: 20 },
  require_fully_labeled_images: true,
  unlabeled_items_on_exported_images: 12,
  images_with_unlabeled_items: 9,
  images_dropped_not_fully_labeled: 30,
  class_split_counts: [
    { class_id: 1, export_id: 0, class_name: 'widget_a', train: 200, val: 40, test: 10 },
    { class_id: 2, export_id: 1, class_name: 'widget_b', train: 0, val: 40, test: 10 },
  ],
};

// An export written before 4c9499a recorded object/partial-frame counts
// — every 4c9499a field is null. Must render "—", never 0.
const EXPORT_STATUS_PRE_D5343CB = {
  status: 'success',
  last_run: '2026-09-20T12:00:00Z',
  export_dir: '/data/export/yolo/v0',
  path: '/data/export/yolo/v0',
  version_tag: 'v0',
  dataset_sha: 'old123',
  seed: 42,
  group_key: 'image_id',
  image_count: 300,
  object_count: null,
  class_count: 2,
  split_counts: { train: 240, val: 40, test: 20 },
  split_object_counts: null,
  require_fully_labeled_images: null,
  unlabeled_items_on_exported_images: null,
  images_with_unlabeled_items: null,
  images_dropped_not_fully_labeled: null,
  class_split_counts: null,
};

function makeFetchMock(
  freezeCalls: Array<{ url: string; body: unknown }>,
  exportStatusBody: unknown = EXPORT_STATUS_SUCCESS,
  exportYoloCalls: Array<{ url: string; body: unknown }> = [],
  // Non-empty by default so `isNothingExportable` doesn't disable the
  // Export button in tests that need to click it.
  statsClassesBody: unknown = {
    classes: [
      {
        class_id: 1,
        class_name: 'widget_a',
        count: 300,
        validated_count: 200,
        adequacy: 'ok',
      },
    ],
  },
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
    if (url.includes('/export/yolo')) {
      exportYoloCalls.push({
        url,
        body: init?.body ? JSON.parse(init.body as string) : null,
      });
      return Promise.resolve(
        jsonResponse({
          status: 'success',
          export_dir: '/data/export/yolo/v2',
          image_count: 500,
          object_count: 620,
          split_counts: { train: 400, val: 80, test: 20 },
          split_object_counts: { train: 500, val: 100, test: 20 },
          require_fully_labeled_images: true,
          unlabeled_items_on_exported_images: 0,
          images_with_unlabeled_items: 0,
          images_dropped_not_fully_labeled: 30,
          started_at: '2026-09-24T12:00:00Z',
          finished_at: '2026-09-24T12:00:05Z',
        }),
      );
    }
    if (url.includes('/export/status')) {
      return Promise.resolve(jsonResponse(exportStatusBody));
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
      return Promise.resolve(jsonResponse(statsClassesBody));
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

describe('/export — served split counts (OpenProcessor df01309)', () => {
  it('renders image/class totals, split totals, and a per-class table highlighting 0-train/0-val classes', async () => {
    const freezeCalls: Array<{ url: string; body: unknown }> = [];
    vi.stubGlobal('fetch', makeFetchMock(freezeCalls));

    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(ExportPage, { target } as never);
    flushSync();
    await flushMicrotasks();
    flushSync();

    expect(target.textContent).toContain('620');
    expect(target.textContent).toContain('objects in');
    expect(target.textContent).toContain('500');
    expect(target.textContent).toContain('grouped by image_id');
    expect(target.textContent).toContain('images: train 400');
    expect(target.textContent).toContain('objects: train 500');
    expect(target.textContent).toContain('images dropped');
    expect(target.textContent).toContain('30');

    const summary = Array.from(target.querySelectorAll('summary')).find((s) =>
      s.textContent?.includes('Per-class object counts'),
    );
    expect(summary).toBeTruthy();
    summary?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    (summary?.closest('details') as HTMLDetailsElement).open = true;
    flushSync();

    expect(target.textContent).toContain('widget_b');
    const audiRow = Array.from(target.querySelectorAll('tr')).find((tr) =>
      tr.textContent?.includes('widget_b'),
    );
    expect(audiRow?.className).toContain('bg-red-500/10');

    const bmwRow = Array.from(target.querySelectorAll('tr')).find((tr) =>
      tr.textContent?.includes('widget_a'),
    );
    expect(bmwRow?.className ?? '').not.toContain('bg-red-500/10');
  });

  it('renders a null 4c9499a field as "—", never 0, for an export written before it was recorded', async () => {
    const freezeCalls: Array<{ url: string; body: unknown }> = [];
    vi.stubGlobal('fetch', makeFetchMock(freezeCalls, EXPORT_STATUS_PRE_D5343CB));

    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(ExportPage, { target } as never);
    flushSync();
    await flushMicrotasks();
    flushSync();

    // object_count is null -> "—", not "0".
    expect(target.textContent).toContain('— objects in');
    expect(target.textContent).toContain('300');
    // split_object_counts is null -> that badge doesn't render at all.
    expect(target.textContent).not.toContain('objects: train');
    // No partial-frame block — every field on it is null.
    expect(target.textContent).not.toContain('require_fully_labeled_images');
    // class_split_counts is null -> no per-class table.
    expect(
      Array.from(target.querySelectorAll('summary')).some((s) =>
        s.textContent?.includes('Per-class object counts'),
      ),
    ).toBe(false);
    // skipped_items is absent on an older export -> no chip.
    expect(target.querySelector('[data-testid="export-skipped-items"]')).toBeNull();
  });

  it('renders the served skipped_items counts (OpenProcessor 536e000)', async () => {
    const freezeCalls: Array<{ url: string; body: unknown }> = [];
    vi.stubGlobal(
      'fetch',
      makeFetchMock(freezeCalls, {
        ...EXPORT_STATUS_SUCCESS,
        skipped_items: { no_image_id: 3, no_usable_box_or_class: 7 },
      }),
    );

    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(ExportPage, { target } as never);
    flushSync();
    await flushMicrotasks();
    flushSync();

    const chip = target.querySelector('[data-testid="export-skipped-items"]');
    expect(chip?.textContent?.replace(/\s+/g, ' ')).toContain(
      '3 no image id · 7 no usable box/class',
    );
  });
});

describe('/export — require_fully_labeled_images opt-in (OpenProcessor 4c9499a)', () => {
  it('checking "Only images whose every object is labeled" sends require_fully_labeled_images: true', async () => {
    const freezeCalls: Array<{ url: string; body: unknown }> = [];
    const exportYoloCalls: Array<{ url: string; body: unknown }> = [];
    vi.stubGlobal(
      'fetch',
      makeFetchMock(freezeCalls, EXPORT_STATUS_SUCCESS, exportYoloCalls),
    );

    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(ExportPage, { target } as never);
    flushSync();
    await flushMicrotasks();
    flushSync();

    const checkbox = Array.from(target.querySelectorAll('input[type="checkbox"]')).find(
      (el) =>
        el
          .closest('label')
          ?.textContent?.includes('Only images whose every object is labeled'),
    ) as HTMLInputElement | undefined;
    expect(checkbox).toBeTruthy();
    checkbox!.click();
    flushSync();

    const exportButton = Array.from(target.querySelectorAll('button')).find(
      (b) => b.textContent?.trim() === 'Export' || b.textContent?.trim() === 'Re-export',
    );
    expect(exportButton).toBeTruthy();
    exportButton?.click();
    flushSync();
    await flushMicrotasks();
    flushSync();

    expect(exportYoloCalls.length).toBeGreaterThan(0);
    expect(exportYoloCalls[0]!.body).toMatchObject({
      require_fully_labeled_images: true,
    });
  });

  it('leaves require_fully_labeled_images out of the request when unchecked', async () => {
    const freezeCalls: Array<{ url: string; body: unknown }> = [];
    const exportYoloCalls: Array<{ url: string; body: unknown }> = [];
    vi.stubGlobal(
      'fetch',
      makeFetchMock(freezeCalls, EXPORT_STATUS_SUCCESS, exportYoloCalls),
    );

    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(ExportPage, { target } as never);
    flushSync();
    await flushMicrotasks();
    flushSync();

    const exportButton = Array.from(target.querySelectorAll('button')).find(
      (b) => b.textContent?.trim() === 'Export' || b.textContent?.trim() === 'Re-export',
    );
    exportButton?.click();
    flushSync();
    await flushMicrotasks();
    flushSync();

    expect(exportYoloCalls.length).toBeGreaterThan(0);
    expect(exportYoloCalls[0]!.body).not.toHaveProperty('require_fully_labeled_images');
  });
});

describe('/export — freeze modal (OpenProcessor df01309: no seed)', () => {
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

describe('/export — class table puts classes with data first', () => {
  it('lists validated classes, folds empty ones behind a toggle that reveals them', async () => {
    vi.stubGlobal(
      'fetch',
      makeFetchMock([], EXPORT_STATUS_SUCCESS, [], {
        classes: [
          {
            class_id: 1,
            class_name: 'alpha_empty',
            count: 38,
            validated_count: 0,
            aug_target: 500,
          },
          {
            class_id: 2,
            class_name: 'beta_empty',
            count: 5,
            validated_count: 0,
            aug_target: 500,
          },
          {
            class_id: 3,
            class_name: 'gamma_full',
            count: 120,
            validated_count: 35,
            aug_target: 500,
          },
        ],
      }),
    );
    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(ExportPage, { target } as never);
    flushSync();
    await flushMicrotasks();
    flushSync();

    const rowNames = () =>
      Array.from(target.querySelectorAll('tbody tr td:first-child')).map((td) =>
        td.textContent?.trim(),
      );
    expect(rowNames()).toContain('gamma_full');
    expect(rowNames()).not.toContain('alpha_empty');
    const toggle = target.querySelector<HTMLButtonElement>(
      '[data-testid="export-empty-classes-toggle"]',
    );
    expect(toggle?.textContent).toContain('2 classes with no validated crops');
    toggle!.click();
    flushSync();
    expect(rowNames()).toEqual(
      expect.arrayContaining(['gamma_full', 'alpha_empty', 'beta_empty']),
    );
    expect(rowNames().indexOf('gamma_full')).toBeLessThan(
      rowNames().indexOf('alpha_empty'),
    );
  });
});
