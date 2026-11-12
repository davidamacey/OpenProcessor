/**
 * `ConfigEditor` over a resource that serves only the document routes (the research presets serve
 * no schema, no live validation, no revision history and no events): load, edit, Save with
 * `expected_revision`, the served 409 and the served 422 `reasons`.
 */
import { describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api_errors';
import { ConfigEditor } from '$lib/config/configEditor.svelte';
import type { ConfigDocCore } from '$lib/types_config';

type Body = Record<string, unknown>;
type Doc = ConfigDocCore<Body>;

const doc = (over: Partial<Doc> = {}): Doc => ({
  name: 'strict',
  revision: 2,
  description: 'stricter',
  body: { version: 1, k: 5 },
  ...over,
});

function setup(update = vi.fn()) {
  const getDoc = vi.fn().mockResolvedValue(doc());
  const active = { load: vi.fn().mockResolvedValue(undefined) };
  const ed = new ConfigEditor<Body, Doc, null, typeof active>(
    'strict',
    { getDoc, update },
    active,
  );
  return { ed, getDoc, update, active };
}

describe('ConfigEditor with only the document routes', () => {
  it('loads the document with no schema, no revisions and no report', async () => {
    const { ed } = setup();
    await ed.load();
    expect(ed.loadError).toBeNull();
    expect(ed.doc?.revision).toBe(2);
    expect(ed.schema).toBeNull();
    expect(ed.revisions).toBeNull();
    expect(ed.report).toBeNull();
    expect(ed.editable).toBe(true);
  });

  it('an edit never posts a validation and Save claims the loaded revision', async () => {
    vi.useFakeTimers();
    const update = vi
      .fn()
      .mockResolvedValue(doc({ revision: 3, body: { version: 1, k: 9 } }));
    const { ed } = setup(update);
    await ed.load();
    ed.setField('k', 9);
    await vi.advanceTimersByTimeAsync(1000);
    expect(ed.validating).toBe(false);
    expect(ed.validateError).toBeNull();
    expect(ed.canSave).toBe(true);
    expect(await ed.save()).toBe(true);
    expect(update).toHaveBeenCalledWith('strict', {
      expected_revision: 2,
      description: 'stricter',
      body: { version: 1, k: 9 },
    });
    expect(ed.doc?.revision).toBe(3);
    vi.useRealTimers();
  });

  it('a stale Save is the served 409 and "keep my edits" claims the served revision', async () => {
    const update = vi.fn().mockRejectedValueOnce(
      new ApiError(409, '/x', {
        detail: {
          error: 'revision_conflict',
          message: "'strict' is at revision 4",
          current_revision: 4,
        },
      }),
    );
    const { ed } = setup(update);
    await ed.load();
    ed.setField('k', 9);
    expect(await ed.save()).toBe(false);
    expect(ed.conflict).toEqual({
      message: "'strict' is at revision 4",
      currentRevision: 4,
    });
    ed.keepMine();
    expect(ed.expectedRevision).toBe(4);
  });

  it('a refused body keeps the served reasons, one per field', async () => {
    const update = vi.fn().mockRejectedValueOnce(
      new ApiError(422, '/x', {
        detail: {
          error: 'validation_failed',
          message: 'invalid config body: 2 error(s)',
          reasons: [
            'item.dedup_cosine: Input should be <= 1',
            'k: Extra inputs are not permitted',
          ],
        },
      }),
    );
    const { ed } = setup(update);
    await ed.load();
    ed.setField('k', 'x');
    expect(await ed.save()).toBe(false);
    expect(ed.saveError).toBe('invalid config body: 2 error(s)');
    expect(ed.saveReasons).toEqual([
      'item.dedup_cosine: Input should be <= 1',
      'k: Extra inputs are not permitted',
    ]);
  });

  it('start() without events only loads, and viewRevision is a no-op', async () => {
    const { ed, getDoc } = setup();
    ed.start();
    await vi.waitFor(() => expect(ed.doc).not.toBeNull());
    expect(getDoc).toHaveBeenCalledTimes(1);
    await ed.viewRevision(1);
    expect(ed.viewing).toBeNull();
    ed.stop();
  });
});
