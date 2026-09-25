import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import IngestStatusTable from './IngestStatusTable.svelte';
import { ApiError, getIngestStatus } from '$lib/api';

vi.mock('$lib/api', async () => {
  const actual = await vi.importActual<typeof import('$lib/api')>('$lib/api');
  return { ...actual, getIngestStatus: vi.fn() };
});

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
  vi.restoreAllMocks();
});

describe('IngestStatusTable', () => {
  it('renders the served total and by_source rows', async () => {
    vi.mocked(getIngestStatus).mockResolvedValue({
      total: 42,
      by_source: [{ key: 'nas', doc_count: 42 }],
      by_day: [],
    });
    instance = mount(IngestStatusTable, { target, props: {} });
    await vi.waitFor(() => {
      flushSync();
      expect(target.textContent).toContain('42');
    });
    expect(target.textContent).toContain('nas');
  });

  it('shows the served error and never renders 0 on a 503', async () => {
    vi.mocked(getIngestStatus).mockRejectedValue(
      new ApiError(503, 'x', { detail: 'opensearch outage' }),
    );
    instance = mount(IngestStatusTable, { target, props: {} });
    await vi.waitFor(() => {
      flushSync();
      expect(target.textContent).toContain('opensearch outage');
    });
  });
});
