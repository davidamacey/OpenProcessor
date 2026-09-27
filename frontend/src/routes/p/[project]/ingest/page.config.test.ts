/**
 * `/ingest` renders from the served `GET {API_PREFIX}/ingest/config`:
 * nothing but a loading line until it loads, the error when it fails,
 * and the upload-caveat banner only when the backend serves
 * `upload.persists_bytes: false`.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { servedIngestConfig } from '$lib/test/fixtures/ingestConfig';

vi.mock('$lib/api', async () => {
  const actual = await vi.importActual<typeof import('$lib/api')>('$lib/api');
  return { ...actual, getIngestConfig: vi.fn() };
});

import { getIngestConfig } from '$lib/api';
import IngestPage from './+page.svelte';

const getConfig = vi.mocked(getIngestConfig);

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | undefined;

beforeEach(() => {
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
});
