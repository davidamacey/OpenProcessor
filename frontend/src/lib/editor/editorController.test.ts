import { beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync } from 'svelte';
import { ApiError } from '$lib/api_errors';
import { imageBox, imageDetail } from '$lib/test/fixtures/imageDetail';
import type { RegistryClass } from '$lib/types_classes';
import type { ImageLocate } from '$lib/types_editor';
import { editorApi as api, editorItem, resetEditorApi } from './editorApiMock';
import type { CanvasApi } from './editorController.svelte';

vi.mock('$lib/api_images', async () =>
  (await import('./editorApiMock')).editorImagesModule(),
);
vi.mock('$lib/api_cropWrites', async () =>
  (await import('./editorApiMock')).editorWritesModule(),
);
vi.mock('$lib/api_regionBoxes', async () =>
  (await import('./editorApiMock')).editorRegionsModule(),
);

const { createEditorController } = await import('./editorController.svelte');

const car = { id: 7, name: 'car' } as RegistryClass;
const BOX: [number, number, number, number] = [0.1, 0.1, 0.5, 0.5];

const c1 = editorItem({ crop_id: 'c1' });
const c2 = editorItem({ crop_id: 'c2', class_excluded: false });
const gone = editorItem({ crop_id: 'c3', class_excluded: true });

function served() {
  return imageDetail({
    items: [c1, c2, gone],
    boxes: [
      imageBox({ crop_id: 'c1', bbox_norm: [0.1, 0.1, 0.3, 0.3] }),
      imageBox({ crop_id: 'c2', bbox_norm: [0.5, 0.5, 0.7, 0.7], label: 'bus' }),
      imageBox({ crop_id: 'c3', bbox_norm: [0.8, 0.8, 0.9, 0.9], tone: 'rejected' }),
      imageBox({ crop_id: 'c1', kind: 'region', box_id: 'r1' }),
    ],
  });
}

const where = (over: Partial<ImageLocate> = {}): ImageLocate => ({
  image_id: 'img-1',
  in_view: true,
  index: 4,
  total: 20,
  cursor_before: 'cb',
  cursor_after: 'ca',
  previous_image_id: 'img-0',
  reason: null,
  reason_message: null,
  ...over,
});

async function setup(locate: ImageLocate = where(), image = served()) {
  const deps = {
    imageId: 'img-1',
    locate: vi.fn().mockResolvedValue(locate),
    fetchPage: vi.fn().mockResolvedValue({
      items: [{ image_id: 'img-2' }, { image_id: 'img-3' }],
    }),
    navigate: vi.fn(),
  };
  api.getImage.mockResolvedValue(image);
  const c = createEditorController(deps);
  await vi.waitFor(() => expect(c.filmstrip).not.toBeNull());
  return { c, deps };
}

beforeEach(resetEditorApi);

describe('boxes', () => {
  it('draws served boxes only', async () => {
    const { c } = await setup();
    expect(c.boxes).toEqual(served().boxes);
    c.draw(BOX);
    expect(c.boxes).toEqual(served().boxes);
    // The write succeeds, but until the server's image is back nothing new is drawn.
    let release: (v: unknown) => void = () => {};
    api.createImageItem.mockReturnValueOnce(new Promise((r) => (release = r)));
    const picked = c.pick(car);
    await Promise.resolve();
    expect(c.boxes).toEqual(served().boxes);
    const after = imageDetail({
      items: [c1, editorItem({ crop_id: 'new-1' })],
      boxes: [imageBox({ crop_id: 'new-1', bbox_norm: BOX })],
    });
    api.getImage.mockResolvedValue(after);
    release({ item: editorItem({ crop_id: 'new-1' }) });
    await picked;
    expect(c.boxes).toEqual(after.boxes);
  });

  it('lists the item objects with the served lock and deleted marks', async () => {
    const { c } = await setup(
      where(),
      imageDetail({
        items: [
          editorItem({ crop_id: 'c1', geometry_locked: true, lock_label: 'Placed' }),
          gone,
        ],
        boxes: [
          imageBox({ crop_id: 'c1' }),
          imageBox({ crop_id: 'c3', tone: 'rejected' }),
        ],
      }),
    );
    expect(c.objects.map((o) => [o.cropId, o.locked, o.lockLabel, o.deleted])).toEqual([
      ['c1', true, 'Placed', false],
      ['c3', false, null, true],
    ]);
    expect(c.locks).toEqual({ c1: 'Placed' });
  });
});

describe('drawing', () => {
  it('after a draw the picker opens with the last class', async () => {
    const { c } = await setup();
    c.setTool('draw');
    c.draw(BOX);
    expect(c.picker).toEqual({ kind: 'draw', box: BOX });
    expect(c.tool).toBe('select');
    await c.pick(car);
    c.draw(BOX);
    expect(c.recentClasses).toEqual(['car']);
    expect(c.picker).toEqual({ kind: 'draw', box: BOX });
  });

  it('the sticky draw tool stays on after a box', async () => {
    const { c } = await setup();
    c.setTool('draw', true);
    c.draw(BOX);
    expect(c.tool).toBe('draw');
    c.cancelPicker();
    expect(c.picker).toBeNull();
  });

  it('picking a class saves the box with that class name and selects the new item', async () => {
    const { c } = await setup();
    const after = imageDetail({
      items: [c1, c2, gone, editorItem({ crop_id: 'new-1' })],
      boxes: [imageBox({ crop_id: 'new-1', bbox_norm: BOX })],
    });
    c.draw(BOX);
    api.getImage.mockResolvedValue(after);
    await c.pick(car);
    expect(api.createImageItem).toHaveBeenCalledWith(
      'img-1',
      expect.objectContaining({ bbox_norm: BOX, class_name: 'car' }),
    );
    expect(c.picker).toBeNull();
    expect(c.selectedCropId).toBe('new-1');
  });
});

describe('selecting and editing', () => {
  it('Tab cycles the live objects and skips the deleted one', async () => {
    const { c } = await setup();
    c.selectNext();
    expect(c.selectedCropId).toBe('c1');
    c.selectNext();
    expect(c.selectedCropId).toBe('c2');
    c.selectNext();
    expect(c.selectedCropId).toBe('c1');
    c.selectPrev();
    expect(c.selectedCropId).toBe('c2');
  });

  it('a class hotkey relabels the selected object', async () => {
    const { c } = await setup();
    c.select('c2');
    await c.setClass(car);
    expect(api.putCropLabel).toHaveBeenCalledWith('c2', 7);
  });

  it('a class hotkey with nothing selected does nothing', async () => {
    const { c } = await setup();
    await c.setClass(car);
    expect(api.putCropLabel).not.toHaveBeenCalled();
  });

  it('delete soft-deletes the selection and moves on to the next live object', async () => {
    const { c } = await setup();
    c.select('c1');
    api.getImage.mockResolvedValue(
      imageDetail({
        items: [editorItem({ crop_id: 'c1', class_excluded: true }), c2, gone],
        boxes: served().boxes,
      }),
    );
    await c.deleteSelected();
    expect(api.excludeCrops).toHaveBeenCalledWith(['c1'], 'not_an_object');
    expect(c.selectedCropId).toBe('c2');
  });

  it('a locked box is never sent', async () => {
    const { c } = await setup(
      where(),
      imageDetail({
        items: [editorItem({ crop_id: 'c1', geometry_locked: true })],
        boxes: [imageBox({ crop_id: 'c1' })],
      }),
    );
    await c.move('c1', BOX);
    expect(api.patchItemGeometry).not.toHaveBeenCalled();
  });

  it('a move sends the held revision and an undo reverts it', async () => {
    const { c } = await setup();
    await c.move('c1', BOX);
    expect(api.patchItemGeometry).toHaveBeenCalledWith('c1', {
      bbox_norm: BOX,
      expected_geometry_revision: 2,
    });
    expect(c.stack.canUndo).toBe(true);
    await c.undo();
    expect(api.undoItemGeometry).toHaveBeenCalledWith('c1', 3);
  });

  it('a conflict reloads the image and shows the served message', async () => {
    const { c } = await setup();
    api.patchItemGeometry.mockRejectedValueOnce(
      new ApiError(409, '/geometry', { detail: 'The box changed. Reload.' }),
    );
    const reloaded = imageDetail({ items: [c1], boxes: [imageBox({ crop_id: 'c1' })] });
    api.getImage.mockResolvedValue(reloaded);
    await c.move('c1', BOX);
    await vi.waitFor(() => expect(c.image).toEqual(reloaded));
    expect(c.stack.status).toBe('conflict');
    expect(c.stack.message).toBe('The box changed. Reload.');
  });

  it('a move that strands region boxes waits for a confirm, then deletes them with the box', async () => {
    const { c } = await setup();
    api.patchItemGeometry.mockRejectedValueOnce(
      new ApiError(422, '/geometry', {
        detail: { error: 'regions_outside_item', message: 'outside', box_ids: ['r1'] },
      }),
    );
    await c.move('c1', BOX);
    expect(c.drop).toEqual({ cropId: 'c1', box: BOX, regionBoxIds: ['r1'] });
    await c.confirmDrop();
    expect(api.patchItemGeometry).toHaveBeenLastCalledWith('c1', {
      bbox_norm: BOX,
      expected_geometry_revision: 2,
      drop_region_box_ids: ['r1'],
    });
    expect(c.drop).toBeNull();
  });

  it('nudge moves the box by whole screen pixels', async () => {
    const { c } = await setup();
    const canvas: CanvasApi = {
      zoomBy: vi.fn(),
      zoomFit: vi.fn(),
      zoomActual: vi.fn(),
      cancelGesture: vi.fn(() => false),
      pixelStep: () => ({ x: 0.001, y: 0.002 }),
    };
    c.attachCanvas(canvas);
    c.select('c1');
    await c.nudge(1, -1);
    const [, body] = api.patchItemGeometry.mock.calls[0];
    expect(body.bbox_norm.map((n: number) => +n.toFixed(6))).toEqual([
      0.101, 0.098, 0.301, 0.298,
    ]);
    await c.nudgeFar(-1, 0);
    const [, far] = api.patchItemGeometry.mock.calls[1];
    expect(far.bbox_norm.map((n: number) => +n.toFixed(6))).toEqual([
      0.09, 0.1, 0.29, 0.3,
    ]);
  });

  it('resize moves the bottom-right corner', async () => {
    const { c } = await setup();
    c.attachCanvas({
      zoomBy: vi.fn(),
      zoomFit: vi.fn(),
      zoomActual: vi.fn(),
      cancelGesture: vi.fn(() => false),
      pixelStep: () => ({ x: 0.01, y: 0.01 }),
    });
    c.select('c1');
    await c.resizeCorner(2, 3);
    const [, body] = api.patchItemGeometry.mock.calls[0];
    expect(body.bbox_norm.map((n: number) => +n.toFixed(6))).toEqual([
      0.1, 0.1, 0.32, 0.33,
    ]);
  });
});

describe('escape', () => {
  it('cancels a gesture, then the picker, then the selection, then leaves', async () => {
    const { c, deps } = await setup();
    const cancelGesture = vi.fn().mockReturnValueOnce(true).mockReturnValue(false);
    c.attachCanvas({
      zoomBy: vi.fn(),
      zoomFit: vi.fn(),
      zoomActual: vi.fn(),
      cancelGesture,
      pixelStep: () => ({ x: 1, y: 1 }),
    });
    c.select('c1');
    c.draw(BOX);
    c.escape();
    expect(c.picker).not.toBeNull();
    c.escape();
    expect(c.picker).toBeNull();
    expect(c.selectedCropId).toBe('c1');
    c.escape();
    expect(c.selectedCropId).toBeNull();
    expect(deps.navigate).not.toHaveBeenCalled();
    c.escape();
    expect(deps.navigate).toHaveBeenCalledWith(null);
    flushSync();
  });
});

describe('walking the list', () => {
  it('next image follows the served locate cursor', async () => {
    const { c, deps } = await setup();
    expect(deps.locate).toHaveBeenCalledOnce();
    expect(deps.fetchPage).toHaveBeenCalledWith('ca', expect.anything());
    c.nextImage();
    expect(deps.navigate).toHaveBeenCalledWith('img-2');
  });

  it('previous image is the served previous id', async () => {
    const { c, deps } = await setup();
    c.prevImage();
    expect(deps.navigate).toHaveBeenCalledWith('img-0');
  });

  it('the first image has no previous one', async () => {
    const { c, deps } = await setup(where({ index: 0, previous_image_id: null }));
    c.prevImage();
    expect(deps.navigate).not.toHaveBeenCalled();
    expect(c.filmstrip?.index).toBe(0);
  });

  it('an image outside the list shows the served reason and walks nowhere', async () => {
    const { c, deps } = await setup(
      where({
        in_view: false,
        index: null,
        cursor_before: null,
        cursor_after: null,
        previous_image_id: null,
        reason: 'filtered_out',
        reason_message: 'The image is not in the current filtered view.',
      }),
    );
    expect(c.filmstrip?.message).toBe('The image is not in the current filtered view.');
    expect(deps.fetchPage).not.toHaveBeenCalled();
    c.nextImage();
    expect(deps.navigate).not.toHaveBeenCalled();
  });

  it('the editor works when the list cannot be located', async () => {
    const deps = {
      imageId: 'img-1',
      locate: vi.fn().mockRejectedValue(new Error('down')),
      fetchPage: vi.fn(),
      navigate: vi.fn(),
    };
    api.getImage.mockResolvedValue(served());
    const c = createEditorController(deps);
    await vi.waitFor(() => expect(c.image).not.toBeNull());
    expect(c.filmstrip).toBeNull();
    expect(c.loadError).toBeNull();
  });

  it('shows the served error when the image cannot be read', async () => {
    api.getImage.mockRejectedValue(
      new ApiError(404, '/images/img-1', { detail: 'No such image.' }),
    );
    const c = createEditorController({
      imageId: 'img-1',
      locate: vi.fn().mockResolvedValue(where()),
      fetchPage: vi.fn(),
      navigate: vi.fn(),
    });
    await vi.waitFor(() => expect(c.loadError).toBe('No such image.'));
    expect(c.image).toBeNull();
  });
});

describe('view state', () => {
  it('toggles the labels and switches tools', async () => {
    const { c } = await setup();
    expect(c.labels).toBe(true);
    c.toggleLabels();
    expect(c.labels).toBe(false);
    c.setTool('pan');
    expect(c.tool).toBe('pan');
    c.setTool('draw', true);
    expect(c.sticky).toBe(true);
    c.setTool('select', true);
    expect(c.sticky).toBe(false);
  });
});
