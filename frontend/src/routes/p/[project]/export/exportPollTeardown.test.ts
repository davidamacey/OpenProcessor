/**
 * The export status poll (used only while the backend reports the export
 * still running) belongs to the page: unmounting /export stops it, so no
 * interval keeps reading `GET {API_PREFIX}/export/status` after the
 * operator navigates away. Same mount harness as `exportStatusContract.test.ts`.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import ExportPage from './+page.svelte';
import { toastStore } from '$stores/toast.svelte';

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });
}

async function flushMicrotasks(): Promise<void> {
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));
}

const RUNNING = { status: 'running', last_run: null, export_dir: null };

let target: HTMLDivElement | undefined;

afterEach(() => {
  target?.remove();
  target = undefined;
  vi.useRealTimers();
  vi.unstubAllGlobals();
  toastStore.toasts = [];
});

describe('/export status poll teardown', () => {
  it('polls while the export runs and stops when the page unmounts', async () => {
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] });
    const fetchMock = vi.fn().mockImplementation((url: string) => {
      if (url.includes('/export/yolo')) return Promise.resolve(jsonResponse(RUNNING));
      if (url.includes('/export/status')) return Promise.resolve(jsonResponse(RUNNING));
      if (url.includes('/export/datasets')) {
        return Promise.resolve(jsonResponse({ datasets: [], count: 0 }));
      }
      return Promise.resolve(jsonResponse({}));
    });
    vi.stubGlobal('fetch', fetchMock);
    const statusReads = () =>
      fetchMock.mock.calls.filter(([u]) => String(u).includes('/export/status')).length;

    target = document.createElement('div');
    document.body.appendChild(target);
    const instance = mount(ExportPage, { target } as never);
    flushSync();
    await flushMicrotasks();
    flushSync();

    const exportButton = Array.from(target.querySelectorAll('button')).find(
      (b) => b.textContent?.trim() === 'Export',
    );
    expect(exportButton).toBeTruthy();
    exportButton!.click();
    flushSync();
    await flushMicrotasks();

    const beforeTick = statusReads();
    await vi.advanceTimersByTimeAsync(5000);
    expect(statusReads()).toBe(beforeTick + 1);

    unmount(instance as never);
    const afterUnmount = statusReads();
    await vi.advanceTimersByTimeAsync(30_000);
    expect(statusReads()).toBe(afterUnmount);
  });
});
