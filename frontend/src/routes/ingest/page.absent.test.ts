/**
 * `/ingest` — "absent, not disabled" (docs/design/
 * ingest-ui-and-acceptance-plan-2026-09-24.md §A.7):
 * `ingestAvailability.available === false` renders the absence copy and
 * no upload UI. This is exercised via the store directly (the page
 * reads it, doesn't fetch it — the layout's probe fires the fetch),
 * matching how ingestAvailability itself is store-tested, not through
 * a full route+layout mount.
 */
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import IngestPage from './+page.svelte';
import { ingestAvailability } from '$lib/ingest/ingestAvailability.svelte';

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
  ingestAvailability.available = null;
});

describe('/ingest — absence', () => {
  it('renders the absence copy and no upload UI when available === false', () => {
    ingestAvailability.available = false;
    instance = mount(IngestPage, { target, props: {} });
    flushSync();
    expect(target.textContent).toContain('This backend does not provide ingest.');
    expect(target.querySelector('input[type=file]')).toBeNull();
    expect(target.querySelector('button')).toBeNull();
  });

  it('renders the upload section when available !== false', () => {
    ingestAvailability.available = true;
    instance = mount(IngestPage, { target, props: {} });
    flushSync();
    expect(target.textContent).not.toContain('This backend does not provide ingest.');
    expect(target.querySelector('input[type=file]')).not.toBeNull();
  });

  it('never renders a server-path panel (absent without served batch config)', () => {
    ingestAvailability.available = true;
    instance = mount(IngestPage, { target, props: {} });
    flushSync();
    expect(target.textContent).not.toMatch(/server.path/i);
  });
});
