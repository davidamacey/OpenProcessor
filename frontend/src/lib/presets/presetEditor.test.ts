import { describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api_errors';
import type { PresetDoc } from '$lib/types_presets';
import { createPresetEditor } from './presetEditorController.svelte';
import { presetKindBySlug } from './presetKinds';

const kind = presetKindBySlug('near-dup-profiles')!;

const doc = (revision: number, body: Record<string, unknown>): PresetDoc => ({
  kind: 'near_dup_profile',
  name: 'strict',
  revision,
  description: 'stricter items',
  body,
  created_at: '2026-11-01T00:00:00Z',
  updated_at: '2026-11-02T00:00:00Z',
  updated_by: 'api',
});

function setup(put = vi.fn()) {
  const deps = {
    getPreset: vi
      .fn()
      .mockResolvedValue(doc(2, { version: 1, item: { dedup_cosine: 0.97 } })),
    putPreset: put,
    getActivePreset: vi.fn().mockResolvedValue({
      axis: 'near_dup',
      state: 'none',
      name: null,
      revision: null,
      activated_at: null,
      setting_revision: null,
    }),
  };
  return { ed: createPresetEditor(kind, 'strict', deps), deps };
}

const refusal = (status: number, detail: Record<string, unknown>) =>
  new ApiError(status, '/x', { detail });

describe('PresetEditor', () => {
  it('shows the served body as indented JSON and nothing to save', async () => {
    const { ed } = setup();
    await ed.load();
    expect(ed.bodyText).toBe(
      JSON.stringify({ version: 1, item: { dedup_cosine: 0.97 } }, null, 2),
    );
    expect(ed.dirty).toBe(false);
    expect(ed.canSave).toBe(false);
  });

  it('text that is not a JSON object cannot be saved and says why', async () => {
    const { ed } = setup();
    await ed.load();
    ed.setBodyText('{"version": ');
    expect(ed.bodyError).toMatch(/JSON/i);
    expect(ed.dirty).toBe(true);
    expect(ed.canSave).toBe(false);
    ed.setBodyText('[1]');
    expect(ed.bodyError).toBe('The body must be a JSON object.');
    expect(ed.canSave).toBe(false);
  });

  it('a valid edit is saved with the loaded revision and the parsed body', async () => {
    const put = vi
      .fn()
      .mockResolvedValue(doc(3, { version: 1, item: { dedup_cosine: 0.99 } }));
    const { ed } = setup(put);
    await ed.load();
    ed.setBodyText('{"version": 1, "item": {"dedup_cosine": 0.99}}');
    expect(ed.bodyError).toBeNull();
    expect(ed.canSave).toBe(true);
    expect(await ed.save()).toBe(true);
    expect(put).toHaveBeenCalledWith('near_dup_profile', 'strict', {
      body: { version: 1, item: { dedup_cosine: 0.99 } },
      description: 'stricter items',
      expected_revision: 2,
    });
    expect(ed.doc?.revision).toBe(3);
    expect(ed.bodyText).toContain('0.99');
  });

  it('a stale save is the served 409; reload drops the edit and the text with it', async () => {
    const put = vi.fn().mockRejectedValue(
      refusal(409, {
        error: 'revision_conflict',
        message: "'strict' is at revision 4",
        current_revision: 4,
      }),
    );
    const { ed, deps } = setup(put);
    await ed.load();
    ed.setBodyText('{"version": 1}');
    expect(await ed.save()).toBe(false);
    expect(ed.conflict).toEqual({
      message: "'strict' is at revision 4",
      currentRevision: 4,
    });
    deps.getPreset.mockResolvedValue(doc(4, { version: 1, item: { dedup_cosine: 0.5 } }));
    await ed.reloadLatest();
    expect(ed.conflict).toBeNull();
    expect(ed.bodyText).toContain('0.5');
    expect(ed.expectedRevision).toBe(4);
  });

  it('a refused body shows the served reasons and keeps the edit', async () => {
    const put = vi.fn().mockRejectedValue(
      refusal(422, {
        error: 'validation_failed',
        message: 'invalid config body: 1 error(s)',
        reasons: ['extra: Extra inputs are not permitted'],
      }),
    );
    const { ed } = setup(put);
    await ed.load();
    ed.setBodyText('{"extra": 1}');
    expect(await ed.save()).toBe(false);
    expect(ed.saveReasons).toEqual(['extra: Extra inputs are not permitted']);
    expect(ed.bodyText).toBe('{"extra": 1}');
  });

  it('format rewrites a parsed text and leaves a broken one alone', async () => {
    const { ed } = setup();
    await ed.load();
    ed.setBodyText('{"version":1}');
    ed.format();
    expect(ed.bodyText).toBe('{\n  "version": 1\n}');
    ed.setBodyText('{"version":');
    ed.format();
    expect(ed.bodyText).toBe('{"version":');
  });
});
