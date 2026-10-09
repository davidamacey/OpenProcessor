import { describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api';
import { VlmPolicyEditor } from './vlmPolicyController.svelte';
import type { VlmPolicy, VlmPolicyUpdate } from '$lib/types_labelConfirmation';

const SERVED: VlmPolicy = {
  scope: 'all',
  conf_max: 0.8,
  per_cluster: 5,
  max_crops_per_day: 0,
  sample_frac: 1,
  revision: 4,
};

function conflict(): ApiError {
  return new ApiError(409, 'Conflict', {
    detail: { error: 'revision_conflict', message: 'vlm policy changed: 4 != 5' },
  });
}

function make(over: Partial<ConstructorParameters<typeof VlmPolicyEditor>[0]> = {}) {
  const getPolicy = vi.fn().mockResolvedValue(SERVED);
  const putPolicy = vi.fn(async (req: VlmPolicyUpdate) => ({
    ...req,
    revision: (req.expected_revision ?? 0) + 1,
  }));
  const editor = new VlmPolicyEditor({ getPolicy, putPolicy, ...over });
  return { editor, getPolicy, putPolicy };
}

describe('VlmPolicyEditor', () => {
  it('adopts the served policy as the draft, without the revision', async () => {
    const { editor } = make();
    await editor.load();
    expect(editor.revision).toBe(4);
    expect(editor.draft).toEqual({
      scope: 'all',
      conf_max: 0.8,
      per_cluster: 5,
      max_crops_per_day: 0,
      sample_frac: 1,
    });
    expect(editor.dirty).toBe(false);
  });

  it('is dirty after an edit and saves with the revision it read', async () => {
    const { editor, putPolicy } = make();
    await editor.load();
    editor.draft.scope = 'uncertain';
    expect(editor.dirty).toBe(true);
    expect(await editor.save()).toBe(true);
    expect(putPolicy).toHaveBeenCalledTimes(1);
    expect(putPolicy.mock.calls[0]![0]).toMatchObject({
      scope: 'uncertain',
      expected_revision: 4,
    });
    expect(editor.revision).toBe(5);
    expect(editor.saved).toBe(true);
    expect(editor.dirty).toBe(false);
  });

  it('does not save before a revision was read', async () => {
    const { editor, putPolicy } = make();
    expect(await editor.save()).toBe(false);
    expect(putPolicy).not.toHaveBeenCalled();
  });

  it('shows a refusal as served and flags a revision conflict', async () => {
    const { editor } = make({ putPolicy: vi.fn().mockRejectedValue(conflict()) });
    await editor.load();
    editor.draft.scope = 'off';
    expect(await editor.save()).toBe(false);
    expect(editor.conflict).toBe(true);
    expect(editor.saveLines).toEqual(['vlm policy changed: 4 != 5']);
    expect(editor.draft.scope).toBe('off');
  });

  it('a non-conflict refusal is not a conflict', async () => {
    const { editor } = make({
      putPolicy: vi.fn().mockRejectedValue(
        new ApiError(422, 'Unprocessable', {
          detail: { error: 'invalid', message: 'sample_frac must be positive' },
        }),
      ),
    });
    await editor.load();
    editor.draft.sample_frac = 0;
    await editor.save();
    expect(editor.conflict).toBe(false);
    expect(editor.saveLines).toEqual(['sample_frac must be positive']);
  });

  it('Keep my edits adopts the fresh revision and keeps the draft', async () => {
    const getPolicy = vi
      .fn()
      .mockResolvedValueOnce(SERVED)
      .mockResolvedValueOnce({ ...SERVED, scope: 'representatives', revision: 9 });
    const { editor } = make({
      getPolicy,
      putPolicy: vi.fn().mockRejectedValue(conflict()),
    });
    await editor.load();
    editor.draft.scope = 'off';
    await editor.save();
    await editor.keepMyEdits();
    expect(editor.revision).toBe(9);
    expect(editor.draft.scope).toBe('off');
    expect(editor.conflict).toBe(false);
  });

  it('Reload drops the edits and adopts the served policy', async () => {
    const getPolicy = vi
      .fn()
      .mockResolvedValueOnce(SERVED)
      .mockResolvedValueOnce({ ...SERVED, scope: 'representatives', revision: 9 });
    const { editor } = make({
      getPolicy,
      putPolicy: vi.fn().mockRejectedValue(conflict()),
    });
    await editor.load();
    editor.draft.scope = 'off';
    await editor.save();
    await editor.reload();
    expect(editor.draft.scope).toBe('representatives');
    expect(editor.revision).toBe(9);
    expect(editor.dirty).toBe(false);
  });

  it('shows the served error when the read fails', async () => {
    const { editor } = make({
      getPolicy: vi
        .fn()
        .mockRejectedValue(new ApiError(404, 'Not Found', { detail: 'Not Found' })),
    });
    await editor.load();
    expect(editor.loadError).toBe('Not Found');
    expect(editor.revision).toBeNull();
  });
});
