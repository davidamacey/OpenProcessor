/**
 * The sources panel renders the served status (counts, trigger times, the scan,
 * the last run, failures) and offers actions only as the backend allows them.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import IngestSourcesPanel from './IngestSourcesPanel.svelte';
import { IngestSources } from '$lib/ingest/sourcesController.svelte';
import { listFixture, sourceFixture } from '$lib/test/fixtures/ingestSources';

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

function controller(list = listFixture(), over: Record<string, unknown> = {}) {
  const deps = {
    listIngestSources: vi.fn().mockResolvedValue(list),
    createIngestSource: vi.fn().mockResolvedValue(sourceFixture()),
    updateIngestSource: vi.fn().mockResolvedValue(sourceFixture()),
    deleteIngestSource: vi.fn().mockResolvedValue(undefined),
    runIngestSource: vi.fn().mockResolvedValue({ run_id: 'bulk_1' }),
    retryDeadObjects: vi.fn().mockResolvedValue({ retried: 2, source: sourceFixture() }),
    ...over,
  };
  return { ctl: new IngestSources(deps), deps };
}

async function render(ctl: IngestSources): Promise<void> {
  instance = mount(IngestSourcesPanel, { target, props: { controller: ctl } });
  await vi.waitFor(() =>
    expect(target.querySelector('[data-testid="source-add"]')).toBeTruthy(),
  );
  flushSync();
}

const text = () => target.textContent ?? '';
const byId = (id: string) => target.querySelector(`[data-testid="${id}"]`);
const click = (el: Element | null) => {
  (el as HTMLElement).click();
  flushSync();
};

describe('IngestSourcesPanel', () => {
  it('renders the served counts, triggers, webhook path, last run and failures', async () => {
    const { ctl } = controller();
    await render(ctl);
    expect(text()).toContain('Incoming photos');
    expect(text()).toContain('s3://photos/incoming/');
    expect(byId('source-counts')?.textContent).toContain('pending 3');
    expect(byId('source-counts')?.textContent).toContain('ingested 120');
    expect(byId('source-counts')?.textContent).toContain('gave up 2');
    expect(byId('source-triggers')?.textContent).toContain('Watch every 60s');
    expect(byId('source-triggers')?.textContent).toContain('0 2 * * *');
    expect(byId('source-webhook-path')?.textContent).toContain('/events');
    expect(byId('source-last-run')?.textContent).toContain('ingested 118');
    expect(byId('source-errors')?.textContent).toContain('incoming/gone.jpg');
    ctl.stop();
  });

  it('shows the scan in progress and a failing watcher with its error, as served', async () => {
    const { ctl } = controller(
      listFixture([
        sourceFixture({
          scan: { active: true, pages: 12, listed: 11000, queued: 300 },
          watch: {
            last_poll_at: null,
            next_poll_at: null,
            error: 'AccessDenied',
            backoff_until: null,
          },
        }),
      ]),
    );
    await render(ctl);
    expect(byId('source-scan')?.textContent).toContain(
      '12 pages, 11000 listed, 300 queued',
    );
    expect(byId('source-watch-error')?.textContent).toContain('AccessDenied');
    ctl.stop();
  });

  it('disables Run now with the served reason, and hides retry when nothing gave up', async () => {
    const { ctl } = controller(
      listFixture([
        sourceFixture({
          can_run: false,
          can_retry_dead: false,
          blocked_reason: 'project_run_live',
          blocked_message: 'Another ingest run of this project is live',
        }),
      ]),
    );
    await render(ctl);
    expect((byId('source-run') as HTMLButtonElement).disabled).toBe(true);
    expect(byId('source-blocked')?.textContent).toContain('Another ingest run');
    expect(byId('source-retry')).toBeNull();
    ctl.stop();
  });

  it('Run now and Retry call the backend for that source', async () => {
    const { ctl, deps } = controller();
    await render(ctl);
    click(byId('source-run'));
    await vi.waitFor(() =>
      expect(deps.runIngestSource).toHaveBeenCalledWith('src_1a2b3c4d5e'),
    );
    await vi.waitFor(() => {
      flushSync();
      expect((byId('source-retry') as HTMLButtonElement).disabled).toBe(false);
    });
    click(byId('source-retry'));
    await vi.waitFor(() => expect(deps.retryDeadObjects).toHaveBeenCalledOnce());
    ctl.stop();
  });

  it('Remove asks first and deletes only on confirm', async () => {
    const { ctl, deps } = controller();
    await render(ctl);
    click(byId('source-remove'));
    expect(deps.deleteIngestSource).not.toHaveBeenCalled();
    const confirm = Array.from(document.querySelectorAll('button')).find(
      (b) => b.textContent?.trim() === 'Remove' && b !== byId('source-remove'),
    );
    click(confirm ?? null);
    await vi.waitFor(() =>
      expect(deps.deleteIngestSource).toHaveBeenCalledWith('src_1a2b3c4d5e'),
    );
    ctl.stop();
  });

  it('the create form disables S3 when the deployment has none and says why', async () => {
    const { ctl } = controller(listFixture([], { s3_configured: false, s3_allowed: [] }));
    await render(ctl);
    expect(byId('sources-empty')).toBeTruthy();
    click(byId('source-add'));
    expect((byId('source-backend-s3') as HTMLInputElement).disabled).toBe(true);
    expect(byId('s3-off')?.textContent).toContain('OP_S3_ALLOWED');
    expect(byId('source-folder')).toBeTruthy();
    ctl.stop();
  });

  it('a refused source keeps the form open with the served reason', async () => {
    const { ctl } = controller(listFixture([]), {
      createIngestSource: vi.fn().mockRejectedValue(new Error('outside OP_S3_ALLOWED')),
    });
    await render(ctl);
    click(byId('source-add'));
    const name = byId('source-name') as HTMLInputElement;
    name.value = 'x';
    name.dispatchEvent(new Event('input', { bubbles: true }));
    const bucket = byId('source-bucket') as HTMLInputElement;
    bucket.value = 'nope';
    bucket.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    click(byId('source-save'));
    await vi.waitFor(() =>
      expect(byId('source-form-error')?.textContent).toContain('outside OP_S3_ALLOWED'),
    );
    ctl.stop();
  });
});
