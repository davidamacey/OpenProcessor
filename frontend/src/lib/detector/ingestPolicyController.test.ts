import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api';
import type {
  IngestPolicy,
  IngestPolicyPreview,
  IngestPolicyPutResponse,
} from '$lib/types_detector';
import {
  IngestPolicyEditor,
  type IngestPolicyDeps,
} from './ingestPolicyController.svelte';

const POLICY: IngestPolicy = {
  detect: { class_resolution: 'proposal', classes: null, exclude_classes: [] },
  embedding: { mode: 'all', classes: [] },
  detector: null,
  revision: 4,
};

const PREVIEW: IngestPolicyPreview = {
  total_items: 100,
  scanned: 100,
  truncated: false,
  would_embed: 40,
  would_not_embed: 60,
  estimated_vector_mb: 1.5,
  by_class: [{ name: 'widget', would_embed: 40, would_not_embed: 60 }],
};

function deps(over: Partial<IngestPolicyDeps> = {}): IngestPolicyDeps {
  return {
    getPolicy: vi.fn(async () => structuredClone(POLICY)),
    getConfig: vi.fn(async () => ({
      detector: {
        model: 'm',
        version: '1',
        input_size: 640,
        assigns_class: false,
        confidence_floor_applies: false,
        n_labels: 2,
        labels: [
          { class_id: 0, name: 'widget', slug: 'widget' },
          { class_id: 1, name: 'tag', slug: 'tag' },
        ],
      },
    })),
    putPolicy: vi.fn(async (): Promise<IngestPolicyPutResponse> => ({
      ...structuredClone(POLICY),
      revision: 5,
      unknown_names: [],
    })),
    previewPolicy: vi.fn(async () => PREVIEW),
    debounceMs: 400,
    ...over,
  };
}

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

describe('IngestPolicyEditor', () => {
  it('loads the policy and the detector labels, draft = served minus revision', async () => {
    const d = deps();
    const ed = new IngestPolicyEditor(d);
    await ed.load();
    expect(ed.revision).toBe(4);
    expect(ed.labelNames).toEqual(['widget', 'tag']);
    expect(ed.draft.embedding?.mode).toBe('all');
    expect('revision' in ed.draft).toBe(false);
    expect(ed.dirty).toBe(false);
  });

  it('shows the served error when the load fails', async () => {
    const ed = new IngestPolicyEditor(
      deps({
        getPolicy: vi.fn(async () => {
          throw new ApiError(500, 'u', { detail: 'boom' });
        }),
      }),
    );
    await ed.load();
    expect(ed.loadError).toBe('boom');
    expect(ed.revision).toBeNull();
  });

  it('debounces the preview, sends the draft and aborts the previous request', async () => {
    const signals: AbortSignal[] = [];
    const previewPolicy = vi.fn((_b: unknown, s?: AbortSignal) => {
      signals.push(s!);
      // the first request stays in flight until the second one aborts it
      return signals.length === 1
        ? new Promise<IngestPolicyPreview>(() => {})
        : Promise.resolve(PREVIEW);
    });
    const ed = new IngestPolicyEditor(deps({ previewPolicy }));
    await ed.load();
    ed.draft.embedding = { ...ed.draft.embedding, mode: 'lazy' };
    ed.schedulePreview();
    ed.draft.embedding = { ...ed.draft.embedding, mode: 'selected' };
    ed.schedulePreview();
    expect(previewPolicy).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(400);
    expect(previewPolicy).toHaveBeenCalledTimes(1);
    expect(previewPolicy.mock.calls[0]![0]).toMatchObject({
      embedding: { mode: 'selected' },
    });
    ed.schedulePreview();
    await vi.advanceTimersByTimeAsync(400);
    expect(signals[0]!.aborted).toBe(true);
    expect(ed.preview).toEqual(PREVIEW);
  });

  it('never saves before save() is called and sends expected_revision', async () => {
    const d = deps();
    const ed = new IngestPolicyEditor(d);
    await ed.load();
    ed.draft.embedding = { ...ed.draft.embedding, mode: 'lazy' };
    expect(d.putPolicy).not.toHaveBeenCalled();
    await ed.save();
    expect(d.putPolicy).toHaveBeenCalledTimes(1);
    expect(vi.mocked(d.putPolicy).mock.calls[0]![0]).toMatchObject({
      embedding: { mode: 'lazy' },
      expected_revision: 4,
    });
    expect(ed.revision).toBe(5);
  });

  it('keeps unknown_names from the response for a warning', async () => {
    const ed = new IngestPolicyEditor(
      deps({
        putPolicy: vi.fn(async () => ({
          ...structuredClone(POLICY),
          revision: 5,
          unknown_names: ['gizmo'],
        })),
      }),
    );
    await ed.load();
    await ed.save();
    expect(ed.unknownNames).toEqual(['gizmo']);
  });

  it('a 409 offers Reload (discards edits) and Keep my edits (adopts the fresh revision)', async () => {
    let served = structuredClone(POLICY);
    const getPolicy = vi.fn(async () => structuredClone(served));
    const putPolicy = vi.fn(async () => {
      throw new ApiError(409, 'u', {
        detail: { error: 'revision_conflict', message: 'The policy changed.' },
      });
    });
    const ed = new IngestPolicyEditor(deps({ getPolicy, putPolicy }));
    await ed.load();
    ed.draft.embedding = { ...ed.draft.embedding, mode: 'lazy' };
    await ed.save();
    expect(ed.conflict).toBe(true);
    expect(ed.saveLines).toEqual(['The policy changed.']);

    served = { ...served, revision: 9, embedding: { mode: 'selected', classes: [] } };
    await ed.keepMyEdits();
    expect(ed.conflict).toBe(false);
    expect(ed.revision).toBe(9);
    expect(ed.draft.embedding?.mode).toBe('lazy');

    // conflict again, then Reload drops the edits
    await ed.save();
    expect(ed.conflict).toBe(true);
    await ed.reload();
    expect(ed.conflict).toBe(false);
    expect(ed.draft.embedding?.mode).toBe('selected');
    expect(ed.dirty).toBe(false);
  });

  it('shows each served reason of a 422 verbatim', async () => {
    const ed = new IngestPolicyEditor(
      deps({
        putPolicy: vi.fn(async () => {
          throw new ApiError(422, 'u', {
            detail: {
              error: 'detector_not_servable',
              message: 'Not servable.',
              reasons: ['no labels file'],
            },
          });
        }),
      }),
    );
    await ed.load();
    await ed.save();
    expect(ed.saveLines).toEqual(['Not servable.', 'no labels file']);
    expect(ed.conflict).toBe(false);
  });

  it('a failed preview shows the served error and keeps no stale result', async () => {
    const previewPolicy = vi
      .fn()
      .mockResolvedValueOnce(PREVIEW)
      .mockRejectedValueOnce(new ApiError(422, 'u', { detail: 'bad mode' }));
    const ed = new IngestPolicyEditor(deps({ previewPolicy }));
    await ed.load();
    ed.schedulePreview();
    await vi.advanceTimersByTimeAsync(400);
    expect(ed.preview).not.toBeNull();
    ed.schedulePreview();
    await vi.advanceTimersByTimeAsync(400);
    expect(ed.preview).toBeNull();
    expect(ed.previewError).toBe('bad mode');
  });
});
