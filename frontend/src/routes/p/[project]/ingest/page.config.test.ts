/**
 * `/ingest` renders from the served `GET {API_PREFIX}/ingest/config`:
 * nothing but a loading line until it loads, the error when it fails,
 * and the upload-caveat banner only when the backend serves
 * `upload.persists_bytes: false`.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { regionProfileStore } from '$stores/regionProfile.svelte';
import { WIDGET_TAG_PROFILE } from '$lib/test/fixtures/regionSlot';
import { servedIngestConfig } from '$lib/test/fixtures/ingestConfig';

const state = vi.hoisted(() => ({ page: { url: new URL('http://x/p/demo/ingest') } }));
vi.mock('$app/state', () => state);

vi.mock('$lib/api_ingest', async () => {
  const actual =
    await vi.importActual<typeof import('$lib/api_ingest')>('$lib/api_ingest');
  return { ...actual, getIngestConfig: vi.fn(), getRegionDrain: vi.fn() };
});

import { getIngestConfig, getRegionDrain } from '$lib/api_ingest';
import IngestPage from './+page.svelte';

const getConfig = vi.mocked(getIngestConfig);
const getDrain = vi.mocked(getRegionDrain);

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | undefined;

beforeEach(() => {
  state.page.url = new URL('http://x/p/demo/ingest');
  target = document.createElement('div');
  document.body.appendChild(target);
});

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target.remove();
  getConfig.mockReset();
});

async function render(): Promise<void> {
  instance = mount(IngestPage, { target, props: {} });
  flushSync();
  await vi.waitFor(() => expect(getConfig).toHaveBeenCalled());
  await Promise.resolve();
  flushSync();
}

describe('/ingest — served config', () => {
  it('renders no upload UI while the config is loading', async () => {
    getConfig.mockReturnValue(new Promise(() => {}));
    await render();
    expect(target.textContent).toContain('Loading');
    expect(target.querySelector('input[type=file]')).toBeNull();
  });

  it('renders the upload section with no caveat once the config loads', async () => {
    getConfig.mockResolvedValue(servedIngestConfig());
    await render();
    await vi.waitFor(() =>
      expect(target.querySelector('input[type=file]')).not.toBeNull(),
    );
    expect(target.textContent).not.toContain('without keeping the image');
    expect(target.textContent).not.toMatch(/server.path/i);
  });

  it('shows the caveat banner when the backend serves persists_bytes: false', async () => {
    const served = servedIngestConfig();
    getConfig.mockResolvedValue({
      ...served,
      upload: { ...served.upload, persists_bytes: false },
    });
    await render();
    await vi.waitFor(() =>
      expect(target.textContent).toContain('without keeping the image'),
    );
  });

  it('shows the error and no upload UI when the config fails to load', async () => {
    getConfig.mockRejectedValue(new Error('boom'));
    await render();
    await vi.waitFor(() => expect(target.textContent).toContain('boom'));
    expect(target.querySelector('input[type=file]')).toBeNull();
  });

  it('replaces the upload panel with one line when the backend serves upload.enabled: false', async () => {
    const served = servedIngestConfig();
    getConfig.mockResolvedValue({
      ...served,
      upload: { ...served.upload, enabled: false, persists_bytes: false },
    });
    await render();
    await vi.waitFor(() =>
      expect(
        target.querySelector('[data-testid="ingest-upload-disabled"]'),
      ).not.toBeNull(),
    );
    expect(target.textContent).toContain(
      'Browser uploads are disabled on this deployment.',
    );
    expect(target.querySelector('input[type=file]')).toBeNull();
    // The upload-only caveat goes with the panel.
    expect(target.textContent).not.toContain('without keeping the image');
  });

  it('shows the server-path panel only when batch is enabled and has source roots', async () => {
    const served = servedIngestConfig();
    getConfig.mockResolvedValue({
      ...served,
      batch: { ...served.batch, enabled: true, source_roots: ['/data/archive'] },
    });
    await render();
    await vi.waitFor(() =>
      expect(target.querySelector('[data-tab="folder"]')?.textContent).toContain(
        'Server folder',
      ),
    );
    expect(target.textContent).not.toContain('Server-path ingest');
  });

  it('the folder tab (the address) shows the bulk and server-path panels', async () => {
    state.page.url = new URL('http://x/p/demo/ingest?tab=folder');
    const served = servedIngestConfig();
    getConfig.mockResolvedValue({
      ...served,
      batch: { ...served.batch, enabled: true, source_roots: ['/data/archive'] },
    });
    await render();
    await vi.waitFor(() => expect(target.textContent).toContain('Server-path ingest'));
    expect(target.textContent).toContain('Bulk ingest');
    expect(target.querySelector('input[type=file]')).toBeNull();
    expect(
      target.querySelector('[data-tab="folder"]')?.getAttribute('aria-current'),
    ).toBe('page');
  });

  it('the Pipeline rail is on every tab', async () => {
    state.page.url = new URL('http://x/p/demo/ingest?tab=sources');
    getConfig.mockResolvedValue(servedIngestConfig());
    await render();
    await vi.waitFor(() =>
      expect(target.querySelector('[data-testid="ingest-pipeline-rail"]')).not.toBeNull(),
    );
    expect(
      target.querySelector('[data-testid="ingest-pipeline-rail"]')!.textContent,
    ).toContain('Cluster');
  });

  it('hides the server-path panel when the backend serves batch.enabled: false, roots or not', async () => {
    const served = servedIngestConfig();
    getConfig.mockResolvedValue({
      ...served,
      batch: { ...served.batch, enabled: false, source_roots: ['/data/archive'] },
    });
    await render();
    await vi.waitFor(() =>
      expect(target.querySelector('input[type=file]')).not.toBeNull(),
    );
    expect(target.textContent).not.toContain('Server-path ingest');
  });

  describe('the region worklog', () => {
    const DRAIN = {
      pending_detection: 7,
      pending_verification: 2,
      total_unfinished: 9,
      drained: false,
      stable_for_s: 0,
      observed_at: '2026-01-01T00:00:00Z',
      region_dependencies: [],
      stall_reason: null,
    };

    it('the rail shows only the pending-detection count; the full worklog is on the Activity tab, from the same read', async () => {
      getDrain.mockResolvedValue(DRAIN as never);
      getConfig.mockResolvedValue(servedIngestConfig());
      regionProfileStore.seed(WIDGET_TAG_PROFILE);
      await render();
      await vi.waitFor(() =>
        expect(
          target.querySelector('[data-testid="region-drain-pending"]')?.textContent,
        ).toBe('7'),
      );
      expect(target.textContent).not.toContain('Total unfinished');
      expect(
        target.querySelector('[data-testid="region-drain-total-unfinished"]'),
      ).toBeNull();
      unmount(instance!);
      state.page.url = new URL('http://x/p/demo/ingest?tab=activity');
      getDrain.mockClear();
      await render();
      await vi.waitFor(() =>
        expect(
          target
            .querySelector('[data-testid="region-drain-total-unfinished"]')
            ?.textContent?.trim(),
        ).toBe('9'),
      );
      expect(getDrain).toHaveBeenCalledTimes(1);
    });
  });
});
