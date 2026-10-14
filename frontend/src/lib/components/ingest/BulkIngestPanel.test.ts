/**
 * The bulk-ingest panel renders the served run (status, counts, rates, stage
 * seconds, failures) and offers cancel / resume only as the backend allows.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import BulkIngestPanel from './BulkIngestPanel.svelte';
import { BulkIngest } from '$lib/ingest/bulkIngestController.svelte';
import {
  configFixture,
  progressFixture,
  runFixture,
} from '$lib/test/fixtures/bulkIngest';

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
});

function controller(run = runFixture(), over: Record<string, unknown> = {}) {
  const deps = {
    getBulkIngestConfig: vi.fn().mockResolvedValue(configFixture()),
    startBulkIngest: vi.fn().mockResolvedValue(run),
    listBulkIngests: vi
      .fn()
      .mockResolvedValue({ items: [run], total: 1, page: 1, page_size: 5 }),
    getBulkIngest: vi.fn().mockResolvedValue(run),
    getBulkIngestFailures: vi
      .fn()
      .mockResolvedValue({ items: [], total: 0, offset: 0, limit: 20 }),
    cancelBulkIngest: vi.fn().mockResolvedValue(run),
    ...over,
  };
  return { ctl: new BulkIngest(deps), deps };
}

async function render(ctl: BulkIngest): Promise<void> {
  instance = mount(BulkIngestPanel, { target, props: { controller: ctl } });
  await vi.waitFor(() =>
    expect(target.querySelector('[data-testid="bulk-path"]')).toBeTruthy(),
  );
  await vi.waitFor(() => flushSync());
}

const text = () => target.textContent ?? '';
const byId = (id: string) => target.querySelector(`[data-testid="${id}"]`);

describe('BulkIngestPanel', () => {
  it('lists the served roots and disables start until a path is typed', async () => {
    const { ctl } = controller(runFixture({ status: 'completed', poll_after_s: null }));
    await render(ctl);
    expect(byId('bulk-roots')?.textContent).toContain('/data/source');
    ctl.run = null;
    flushSync();
    expect((byId('bulk-start') as HTMLButtonElement).disabled).toBe(true);
    ctl.form.path = '/data/source/coco';
    flushSync();
    expect((byId('bulk-start') as HTMLButtonElement).disabled).toBe(false);
  });

  it('shows a live run with its rates and a cancel button', async () => {
    const run = runFixture({
      progress: progressFixture({
        discovered: 400,
        processed: 100,
        ingested: 98,
        duplicates: 2,
        images_per_s: 19.5,
        images_per_s_recent: 21.2,
        eta_s: 125,
        elapsed_s: 5,
      }),
      stage_s: { embed: 40.2, batch_requests: 12.5 },
    });
    const { ctl } = controller(run);
    await render(ctl);
    await vi.waitFor(() => expect(byId('bulk-run')).toBeTruthy());
    expect(byId('bulk-status')?.textContent).toContain('running');
    expect(byId('bulk-rate')?.textContent).toContain('21.2 img/s');
    expect(byId('bulk-eta')?.textContent).toContain('2m 05s');
    expect(byId('bulk-counts')?.textContent).toContain('100 of 400+ (still scanning)');
    expect(byId('bulk-stages')?.textContent).toContain('embed');
    expect(byId('bulk-cancel')).toBeTruthy();
    expect(byId('bulk-resume')).toBeNull();
    expect((byId('bulk-path') as HTMLInputElement).disabled).toBe(true);
    ctl.stop();
  });

  it('shows the run average, not the empty recent window, once a run has stopped', async () => {
    const run = runFixture({
      status: 'cancelled',
      poll_after_s: null,
      can_cancel: false,
      can_resume: true,
      progress: progressFixture({
        discovered: 464,
        processed: 320,
        ingested: 320,
        images_per_s: 18.2,
        images_per_s_recent: 0,
      }),
    });
    const { ctl } = controller(run);
    await render(ctl);
    await vi.waitFor(() => expect(byId('bulk-run')).toBeTruthy());
    expect(byId('bulk-rate')?.textContent).toContain('18.2 img/s');
    expect(byId('bulk-counts')?.textContent).toContain('found before it stopped');
    expect(byId('bulk-counts')?.textContent).not.toContain('still scanning');
  });

  it('cancel button calls the controller; a requested cancel reads as cancelling', async () => {
    const { ctl, deps } = controller();
    await render(ctl);
    await vi.waitFor(() => expect(byId('bulk-cancel')).toBeTruthy());
    (byId('bulk-cancel') as HTMLButtonElement).click();
    expect(deps.cancelBulkIngest).toHaveBeenCalled();
    ctl.run = runFixture({ cancel_requested: true });
    flushSync();
    expect(byId('bulk-status')?.textContent).toContain('cancelling');
    expect((byId('bulk-cancel') as HTMLButtonElement).disabled).toBe(true);
    ctl.stop();
  });

  it('offers resume only when the run says it can be resumed, and lists failures', async () => {
    const run = runFixture({
      status: 'completed_with_errors',
      poll_after_s: null,
      can_cancel: false,
      can_resume: true,
      progress: progressFixture({ discovered: 3, processed: 3, ingested: 2, failed: 1 }),
    });
    const { ctl, deps } = controller(run, {
      getBulkIngestFailures: vi.fn().mockResolvedValue({
        items: [
          {
            path: '/data/source/x.jpg',
            error: 'cannot read',
            error_kind: 'decode_failed',
          },
        ],
        total: 1,
        offset: 0,
        limit: 20,
      }),
    });
    await render(ctl);
    await vi.waitFor(() => expect(byId('bulk-failures')).toBeTruthy());
    expect(byId('bulk-failures')?.textContent).toContain('x.jpg');
    expect(byId('bulk-failures')?.textContent).toContain('decode_failed');
    expect(byId('bulk-cancel')).toBeNull();
    (byId('bulk-resume') as HTMLButtonElement).click();
    expect(deps.startBulkIngest).toHaveBeenCalledWith(
      expect.objectContaining({ folder: '/data/source/coco' }),
    );
  });

  it('shows the served error of a run that stopped', async () => {
    const run = runFixture({
      status: 'failed',
      poll_after_s: null,
      can_cancel: false,
      can_resume: true,
      error: '5 batches in a row failed entirely',
    });
    const { ctl } = controller(run);
    await render(ctl);
    await vi.waitFor(() => expect(byId('bulk-run-error')).toBeTruthy());
    expect(text()).toContain('5 batches in a row failed entirely');
  });

  it('shows the served reason a config read failed', async () => {
    const { ctl } = controller(undefined, {
      getBulkIngestConfig: vi.fn().mockRejectedValue(new Error('boom')),
    });
    instance = mount(BulkIngestPanel, { target, props: { controller: ctl } });
    await vi.waitFor(() => expect(text()).toContain('Could not load bulk ingest: boom'));
  });
});
