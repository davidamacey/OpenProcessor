import { beforeEach, describe, expect, it, vi } from 'vitest';
import { mapRawCrop } from '$lib/api_cropMap';
import { ApiError } from '$lib/api_errors';
import { makeItem } from '$lib/test/makeItem';
import { imageDetail } from '$lib/test/fixtures/imageDetail';
import type { RegistryClass } from '$lib/types_classes';

const api = vi.hoisted(() => ({
  createImageItem: vi.fn(),
  getImage: vi.fn(),
  patchItemGeometry: vi.fn(),
  undoItemGeometry: vi.fn(),
  excludeCrops: vi.fn(),
  unexcludeCrops: vi.fn(),
  putCropLabel: vi.fn(),
  undoCropLabel: vi.fn(),
  undoCropRegion: vi.fn(),
  putRegionBoxes: vi.fn(),
}));
vi.mock('$lib/api_images', () => ({
  createImageItem: api.createImageItem,
  getImage: api.getImage,
}));
vi.mock('$lib/api_cropWrites', () => ({
  patchItemGeometry: api.patchItemGeometry,
  undoItemGeometry: api.undoItemGeometry,
  excludeCrops: api.excludeCrops,
  unexcludeCrops: api.unexcludeCrops,
  putCropLabel: api.putCropLabel,
  undoCropLabel: api.undoCropLabel,
  undoCropRegion: api.undoCropRegion,
}));
vi.mock('$lib/api_regionBoxes', () => ({ putRegionBoxes: api.putRegionBoxes }));

const {
  CommandStack,
  classCommand,
  deleteCommand,
  drawCommand,
  moveCommand,
  regionCommand,
} = await import('./editorCommands.svelte');

const item = mapRawCrop(
  makeItem({ crop_id: 'c1', image_id: 'img-1', geometry_revision: 2 }),
);
const BOX: [number, number, number, number] = [0.1, 0.1, 0.5, 0.5];
const car = { id: 7, name: 'car' } as RegistryClass;

function conflict(): ApiError {
  return new ApiError(409, '/geometry', { detail: 'The box changed. Reload.' });
}

beforeEach(() => {
  vi.clearAllMocks();
  api.getImage.mockResolvedValue(imageDetail());
  api.createImageItem.mockResolvedValue({ item: { ...item, id: 'new-1' } });
  api.patchItemGeometry.mockResolvedValue({ item: { ...item, geometry_revision: 3 } });
  api.undoItemGeometry.mockResolvedValue({ item: { ...item, geometry_revision: 4 } });
  for (const k of ['excludeCrops', 'unexcludeCrops', 'putCropLabel', 'undoCropLabel'])
    api[k as 'excludeCrops'].mockResolvedValue({});
  api.undoCropRegion.mockResolvedValue({});
  api.putRegionBoxes.mockResolvedValue({});
});

describe("each command's revert inverts apply", () => {
  it('draw: creates the item, then excludes it as not an object', async () => {
    const cmd = drawCommand('img-1', BOX, 'car');
    expect((await cmd.apply()).ok).toBe(true);
    expect(api.createImageItem).toHaveBeenCalledWith(
      'img-1',
      expect.objectContaining({ bbox_norm: BOX, class_name: 'car' }),
    );
    await cmd.revert();
    expect(api.excludeCrops).toHaveBeenCalledWith(['new-1'], 'not_an_object');
  });

  it('draw: redo restores the item it created instead of creating a second', async () => {
    const cmd = drawCommand('img-1', BOX, 'car');
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

  it('class: labels with the registry id, then undoes the label', async () => {
    const cmd = classCommand(item, car);
    await cmd.apply();
    expect(api.putCropLabel).toHaveBeenCalledWith('c1', 7);
    await cmd.revert();
    expect(api.undoCropLabel).toHaveBeenCalledWith('c1');
  });

  it('region: replaces the region boxes, then undoes the region write', async () => {
    const boxes = [{ box_id: null, bbox_norm: BOX }];
    const cmd = regionCommand(item, boxes);
    await cmd.apply();
    expect(api.putRegionBoxes).toHaveBeenCalledWith('c1', boxes);
    await cmd.revert();
    expect(api.undoCropRegion).toHaveBeenCalledWith('c1');
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
    const cmd = drawCommand('img-1', BOX, 'car');
    expect((await cmd.apply()).ok).toBe(false);
    expect((await cmd.apply()).ok).toBe(true);
    const ids = api.createImageItem.mock.calls.map((c) => c[1].request_id);
    expect(ids).toHaveLength(2);
    expect(ids[0]).toBe(ids[1]);
    expect(ids[0]).toMatch(/^[0-9a-f-]{36}$/);
  });

  it('uses a new request id for the next box', () => {
    const a = drawCommand('img-1', BOX, 'car');
    const b = drawCommand('img-1', BOX, 'car');
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
    await s.run(drawCommand('img-1', BOX, 'car'));
    expect(s.undoLabel).toBe('Draw car');
    await s.undo();
    expect(s.redoLabel).toBe('Draw car');
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

  it('a move confirmed with region box ids sends them as the drop list', async () => {
    await moveCommand(item, BOX, ['b1']).apply();
    expect(api.patchItemGeometry).toHaveBeenCalledWith('c1', {
      bbox_norm: BOX,
      expected_geometry_revision: 2,
      drop_region_box_ids: ['b1'],
    });
  });
});
