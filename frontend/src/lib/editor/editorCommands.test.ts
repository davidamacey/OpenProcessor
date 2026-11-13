import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api_errors';
import { imageDetail } from '$lib/test/fixtures/imageDetail';
import type { RegistryClass } from '$lib/types_classes';
import { editorApi as api, editorItem, resetEditorApi } from './editorApiMock';

vi.mock('$lib/api_images', async () =>
  (await import('./editorApiMock')).editorImagesModule(),
);
vi.mock('$lib/api_cropWrites', async () =>
  (await import('./editorApiMock')).editorWritesModule(),
);
vi.mock('$lib/labelOps/runLabelOp', async () =>
  (await import('./editorApiMock')).editorLabelOpsModule(),
);
vi.mock('$lib/api_regionBoxes', async () =>
  (await import('./editorApiMock')).editorRegionsModule(),
);

const {
  CommandStack,
  classCommand,
  deleteCommand,
  drawCommand,
  moveCommand,
  regionEditCommand,
} = await import('./editorCommands.svelte');

const item = editorItem();
const BOX: [number, number, number, number] = [0.1, 0.1, 0.5, 0.5];
const widget = { id: 7, name: 'widget' } as RegistryClass;

function conflict(): ApiError {
  return new ApiError(409, '/geometry', { detail: 'The box changed. Reload.' });
}

beforeEach(resetEditorApi);

describe("each command's revert inverts apply", () => {
  it('draw: creates the item, then excludes it as not an object', async () => {
    const cmd = drawCommand('img-1', BOX, 'widget');
    expect((await cmd.apply()).ok).toBe(true);
    expect(api.createImageItem).toHaveBeenCalledWith(
      'img-1',
      expect.objectContaining({ bbox_norm: BOX, class_name: 'widget' }),
    );
    await cmd.revert();
    expect(api.excludeCrops).toHaveBeenCalledWith(['new-1'], 'not_an_object');
  });

  it('draw: redo restores the item it created instead of creating a second', async () => {
    const cmd = drawCommand('img-1', BOX, 'widget');
    await cmd.apply();
    await cmd.revert();
    await cmd.apply();
    expect(api.createImageItem).toHaveBeenCalledTimes(1);
    expect(api.unexcludeCrops).toHaveBeenCalledWith(['new-1']);
  });

  it('move: patches with the held revision, then undoes with the new one', async () => {
    const cmd = moveCommand(item, BOX);
    await cmd.apply();
    expect(api.patchItemGeometry).toHaveBeenCalledWith('c1', {
      bbox_norm: BOX,
      expected_geometry_revision: 2,
    });
    await cmd.revert();
    expect(api.undoItemGeometry).toHaveBeenCalledWith('c1', 3);
  });

  it('delete: excludes as not an object, then unexcludes', async () => {
    const cmd = deleteCommand(item);
    await cmd.apply();
    expect(api.excludeCrops).toHaveBeenCalledWith(['c1'], 'not_an_object');
    await cmd.revert();
    expect(api.unexcludeCrops).toHaveBeenCalledWith(['c1']);
  });

  it('class: labels with the registry id, then undoes the recorded label operation', async () => {
    const cmd = classCommand(item, widget);
    await cmd.apply();
    expect(api.putCropLabel).toHaveBeenCalledWith('c1', 7);
    await cmd.revert();
    expect(api.undoLabelOpToEnd).toHaveBeenCalledWith('op-1', false);
  });

  it('class: a redo records a new operation, and the next undo reverts that one', async () => {
    api.putCropLabel
      .mockResolvedValueOnce({
        crop_id: 'c1',
        class_id: 7,
        class_name: 'widget',
        op_id: 'op-1',
      })
      .mockResolvedValueOnce({
        crop_id: 'c1',
        class_id: 7,
        class_name: 'widget',
        op_id: 'op-2',
      });
    const cmd = classCommand(item, widget);
    await cmd.apply();
    await cmd.revert();
    await cmd.apply();
    await cmd.revert();
    expect(api.undoLabelOpToEnd.mock.calls.map((c) => c[0])).toEqual(['op-1', 'op-2']);
  });

  it('class: a failed operation undo is reported, not swallowed', async () => {
    api.undoLabelOpToEnd.mockRejectedValueOnce(new Error('the undo did not finish'));
    const cmd = classCommand(item, widget);
    await cmd.apply();
    const r = await cmd.revert();
    expect(r.ok).toBe(false);
  });

  it('region edit: undo is the region undo, and it cannot be redone', async () => {
    const cmd = regionEditCommand(item);
    await cmd.revert();
    expect(api.undoCropRegion).toHaveBeenCalledWith('c1');
    expect(cmd.redoable).toBe(false);
    expect((await cmd.apply()).ok).toBe(false);
  });

  it('every successful write returns the served image', async () => {
    const served = imageDetail({ image: { image_id: 'img-9' } });
    api.getImage.mockResolvedValue(served);
    const r = await deleteCommand(item).apply();
    expect(r).toEqual({ ok: true, image: served });
    expect(api.getImage).toHaveBeenCalledWith('img-1');
  });
});

describe('draw retry', () => {
  it('reuses the request id after a failed attempt', async () => {
    api.createImageItem.mockRejectedValueOnce(new Error('network down'));
    const cmd = drawCommand('img-1', BOX, 'widget');
    expect((await cmd.apply()).ok).toBe(false);
    expect((await cmd.apply()).ok).toBe(true);
    const ids = api.createImageItem.mock.calls.map((c) => c[1].request_id);
    expect(ids).toHaveLength(2);
    expect(ids[0]).toBe(ids[1]);
    expect(ids[0]).toMatch(/^[0-9a-f-]{36}$/);
  });

  it('uses a new request id for the next box', () => {
    const a = drawCommand('img-1', BOX, 'widget');
    const b = drawCommand('img-1', BOX, 'widget');
    void a.apply();
    void b.apply();
    expect(api.createImageItem.mock.calls[0][1].request_id).not.toBe(
      api.createImageItem.mock.calls[1][1].request_id,
    );
  });
});

describe('CommandStack', () => {
  function stack() {
    const hooks = { onImage: vi.fn(), onConflict: vi.fn() };
    return { s: new CommandStack(hooks), hooks };
  }

  it('a 409 drops the entry and keeps the served item', async () => {
    api.patchItemGeometry.mockRejectedValueOnce(conflict());
    const { s, hooks } = stack();
    await s.run(moveCommand(item, BOX));
    expect(s.canUndo).toBe(false);
    expect(hooks.onImage).not.toHaveBeenCalled();
    expect(hooks.onConflict).toHaveBeenCalledOnce();
    expect(s.status).toBe('conflict');
    expect(s.message).toBe('The box changed. Reload.');
  });

  it('redo sends the fresh geometry revision', async () => {
    const { s } = stack();
    await s.run(moveCommand(item, BOX));
    await s.undo();
    await s.redo();
    expect(api.patchItemGeometry).toHaveBeenLastCalledWith('c1', {
      bbox_norm: BOX,
      expected_geometry_revision: 4,
    });
  });

  it('records a write made elsewhere so it can be undone, but never offers a redo', async () => {
    const { s, hooks } = stack();
    await s.record(regionEditCommand(item), 'img-1');
    expect(s.canUndo).toBe(true);
    expect(hooks.onImage).toHaveBeenCalledOnce();
    await s.undo();
    expect(api.undoCropRegion).toHaveBeenCalledWith('c1');
    expect(s.canUndo).toBe(false);
    expect(s.canRedo).toBe(false);
  });

  it('a new write clears the redo list', async () => {
    const { s } = stack();
    await s.run(deleteCommand(item));
    await s.undo();
    expect(s.canRedo).toBe(true);
    await s.run(deleteCommand(item));
    expect(s.canRedo).toBe(false);
    expect(s.canUndo).toBe(true);
  });

  it('an undo that conflicts drops that entry', async () => {
    const { s, hooks } = stack();
    await s.run(moveCommand(item, BOX));
    api.undoItemGeometry.mockRejectedValueOnce(conflict());
    await s.undo();
    expect(s.canUndo).toBe(false);
    expect(s.canRedo).toBe(false);
    expect(hooks.onConflict).toHaveBeenCalledOnce();
  });

  it('a failed write that is not a conflict keeps what was already undoable', async () => {
    const { s } = stack();
    await s.run(deleteCommand(item));
    api.excludeCrops.mockRejectedValueOnce(new Error('network down'));
    await s.run(deleteCommand(item));
    expect(s.status).toBe('failed');
    expect(s.canUndo).toBe(true);
  });

  it('is saving while a write is in flight and saved after', async () => {
    let release: (v: unknown) => void = () => {};
    api.excludeCrops.mockReturnValueOnce(new Promise((r) => (release = r)));
    const { s } = stack();
    const done = s.run(deleteCommand(item));
    expect(s.saving).toBe(true);
    expect(s.status).toBe('saving');
    release({});
    await done;
    expect(s.saving).toBe(false);
    expect(s.status).toBe('saved');
  });

  it('a turn that runs nothing leaves the status as it was, never stuck on saving', async () => {
    const { s } = stack();
    await s.run(() => null);
    expect(s.status).toBe('saved');
    expect(s.saving).toBe(false);
    api.excludeCrops.mockRejectedValueOnce(new Error('network down'));
    await s.run(deleteCommand(item));
    await s.run(() => null);
    expect(s.status).toBe('failed');
  });

  it('a command that throws, or a queued turn that runs nothing, never leaves it on saving', async () => {
    const { s } = stack();
    const throwing = {
      label: 'Boom',
      apply: () => Promise.reject(new Error('boom')),
      revert: () => Promise.reject(new Error('boom')),
    };
    await expect(s.run(throwing)).rejects.toThrow('boom');
    expect(s.saving).toBe(false);
    expect(s.status).not.toBe('saving');
    const first = s.run(deleteCommand(item));
    const second = s.run(() => null);
    await Promise.all([first, second]);
    expect(s.saving).toBe(false);
    expect(s.status).toBe('saved');
  });

  it('runs writes one after the other', async () => {
    const order: string[] = [];
    api.excludeCrops.mockImplementationOnce(async () => {
      await new Promise((r) => setTimeout(r, 5));
      order.push('first');
      return {};
    });
    api.excludeCrops.mockImplementationOnce(async () => {
      order.push('second');
      return {};
    });
    const { s } = stack();
    await Promise.all([s.run(deleteCommand(item)), s.run(deleteCommand(item))]);
    expect(order).toEqual(['first', 'second']);
  });

  it('names the next undo and redo for the tooltip', async () => {
    const { s } = stack();
    await s.run(drawCommand('img-1', BOX, 'widget'));
    expect(s.undoLabel).toBe('Draw widget');
    await s.undo();
    expect(s.redoLabel).toBe('Draw widget');
  });

  it('a move that would strand region boxes reports them and sends nothing else', async () => {
    api.patchItemGeometry.mockRejectedValueOnce(
      new ApiError(422, '/geometry', {
        detail: {
          error: 'regions_outside_item',
          message: 'region boxes would fall outside',
          box_ids: ['b1', 'b2'],
        },
      }),
    );
    const r = await moveCommand(item, BOX).apply();
    expect(r).toMatchObject({ ok: false, conflict: false, regionBoxIds: ['b1', 'b2'] });
  });

  it('the stack keeps the region boxes a move would strand until the next write', async () => {
    api.patchItemGeometry.mockRejectedValueOnce(
      new ApiError(422, '/geometry', {
        detail: { error: 'regions_outside_item', message: 'outside', box_ids: ['b1'] },
      }),
    );
    const { s } = stack();
    await s.run(moveCommand(item, BOX));
    expect(s.regionBoxIds).toEqual(['b1']);
    expect(s.canUndo).toBe(false);
    await s.run(deleteCommand(item));
    expect(s.regionBoxIds).toBeNull();
  });

  it('a move confirmed with region box ids sends them as the drop list', async () => {
    await moveCommand(item, BOX, ['b1']).apply();
    expect(api.patchItemGeometry).toHaveBeenCalledWith('c1', {
      bbox_norm: BOX,
      expected_geometry_revision: 2,
      drop_region_box_ids: ['b1'],
    });
  });
});
