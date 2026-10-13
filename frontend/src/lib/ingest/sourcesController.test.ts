import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { IngestSources, toPatch, toRequest, blankForm } from './sourcesController.svelte';
import { listFixture, sourceFixture, SOURCE_ID } from '$lib/test/fixtures/ingestSources';
import { runFixture } from '$lib/test/fixtures/bulkIngest';

function deps(over: Record<string, unknown> = {}) {
  return {
    listIngestSources: vi.fn().mockResolvedValue(listFixture()),
    createIngestSource: vi.fn().mockResolvedValue(sourceFixture()),
    updateIngestSource: vi.fn().mockResolvedValue(sourceFixture()),
    deleteIngestSource: vi.fn().mockResolvedValue(undefined),
    runIngestSource: vi.fn().mockResolvedValue(runFixture()),
    retryDeadObjects: vi.fn().mockResolvedValue({ retried: 2, source: sourceFixture() }),
    ...over,
  };
}

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

describe('IngestSources', () => {
  it('re-reads the list after the interval the backend served, and stops when told', async () => {
    const d = deps({
      listIngestSources: vi
        .fn()
        .mockResolvedValueOnce(listFixture([sourceFixture()], { poll_after_s: 2 }))
        .mockResolvedValue(listFixture([sourceFixture()], { poll_after_s: 30 })),
    });
    const ctl = new IngestSources(d);
    await ctl.init();
    expect(d.listIngestSources).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1900);
    expect(d.listIngestSources).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(200);
    expect(d.listIngestSources).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(29000);
    expect(d.listIngestSources).toHaveBeenCalledTimes(2); // now 30 s apart
    ctl.stop();
    await vi.advanceTimersByTimeAsync(60000);
    expect(d.listIngestSources).toHaveBeenCalledTimes(2);
  });

  it('sends an S3 source exactly as the form states it and leaves unset options to the backend', () => {
    const f = blankForm();
    Object.assign(f, {
      name: ' Inbox ',
      bucket: 'photos',
      prefix: 'incoming/',
      watchEnabled: true,
      scheduleEnabled: true,
      cron: ' 0 2 * * * ',
      webhook: true,
    });
    expect(toRequest(f)).toEqual({
      name: 'Inbox',
      backend: 's3',
      source: 'source',
      s3: { bucket: 'photos', prefix: 'incoming/' },
      triggers: {
        watch: { enabled: true },
        schedule: { enabled: true, cron: '0 2 * * *' },
        webhook_enabled: true,
      },
    });
  });

  it('a folder source never claims a webhook, and a patch carries no location', () => {
    const f = Object.assign(blankForm(), {
      backend: 'folder',
      folder: '/data/in',
      webhook: true,
      name: 'F',
    });
    const body = toRequest(f);
    expect(body.folder).toBe('/data/in');
    expect(body.s3).toBeUndefined();
    expect(body.triggers?.webhook_enabled).toBe(false);
    expect(toPatch(f)).not.toHaveProperty('folder');
    expect(toPatch(f)).not.toHaveProperty('backend');
  });

  it('creating refreshes the list; a refusal is shown as served and keeps the form open', async () => {
    const d = deps();
    const ctl = new IngestSources(d);
    await ctl.init();
    ctl.openCreate();
    Object.assign(ctl.form, { name: 'x', bucket: 'b' });
    expect(ctl.canSave).toBe(true);
    expect(await ctl.save()).toBe(true);
    expect(d.createIngestSource).toHaveBeenCalledOnce();
    expect(ctl.editing).toBeNull();
    expect(d.listIngestSources).toHaveBeenCalledTimes(2);

    d.createIngestSource.mockRejectedValueOnce(
      new Error('s3://b/ is outside OP_S3_ALLOWED'),
    );
    ctl.openCreate();
    Object.assign(ctl.form, { name: 'y', bucket: 'b' });
    expect(await ctl.save()).toBe(false);
    expect(ctl.actionError).toContain('outside OP_S3_ALLOWED');
    expect(ctl.editing).toBe('new');
  });

  it('with S3 off the create form opens on a folder and cannot be saved without a path', async () => {
    const ctl = new IngestSources(
      deps({
        listIngestSources: vi
          .fn()
          .mockResolvedValue(listFixture([], { s3_configured: false })),
      }),
    );
    await ctl.init();
    ctl.openCreate();
    expect(ctl.form.backend).toBe('folder');
    ctl.form.name = 'n';
    expect(ctl.canSave).toBe(false);
  });

  it('run now and retry name what the backend did', async () => {
    const d = deps();
    const ctl = new IngestSources(d);
    await ctl.init();
    await ctl.run(SOURCE_ID);
    expect(d.runIngestSource).toHaveBeenCalledWith(SOURCE_ID);
    expect(ctl.notice).toContain('Started bulk_');
    await ctl.retryDead(SOURCE_ID);
    expect(ctl.notice).toBe('2 objects queued again');
  });

  it('a busy project is the backend answer and is surfaced, not retried', async () => {
    const d = deps({
      runIngestSource: vi
        .fn()
        .mockRejectedValue(new Error('a bulk ingest is already running')),
    });
    const ctl = new IngestSources(d);
    await ctl.init();
    expect(await ctl.run(SOURCE_ID)).toBe(false);
    expect(ctl.actionError).toContain('already running');
    expect(d.runIngestSource).toHaveBeenCalledOnce();
  });
});
