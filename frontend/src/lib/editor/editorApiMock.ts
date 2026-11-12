/**
 * The mocked write and read clients the editor tests share. A test file mocks the two modules
 * with `editorImagesModule()` and `editorWritesModule()` and calls `resetEditorApi()` before
 * each test.
 */
import { vi } from 'vitest';
import { mapRawCrop } from '$lib/api_cropMap';
import { makeItem } from '$lib/test/makeItem';
import { imageDetail } from '$lib/test/fixtures/imageDetail';

export const editorApi = {
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
};

export const editorImagesModule = () => ({
  createImageItem: editorApi.createImageItem,
  getImage: editorApi.getImage,
});

export const editorWritesModule = () => ({
  patchItemGeometry: editorApi.patchItemGeometry,
  undoItemGeometry: editorApi.undoItemGeometry,
  excludeCrops: editorApi.excludeCrops,
  unexcludeCrops: editorApi.unexcludeCrops,
  putCropLabel: editorApi.putCropLabel,
  undoCropLabel: editorApi.undoCropLabel,
  undoCropRegion: editorApi.undoCropRegion,
});

export const editorRegionsModule = () => ({ putRegionBoxes: editorApi.putRegionBoxes });

/** An item of image `img-1` as the client maps it. */
export const editorItem = (over: Parameters<typeof makeItem>[0] = {}) =>
  mapRawCrop(
    makeItem({
      crop_id: 'c1',
      image_id: 'img-1',
      geometry_revision: 2,
      geometry_locked: false,
      lock_label: null,
      class_excluded: false,
      ...over,
    }),
  );

/** Clears every call and gives each client a successful default answer. */
export function resetEditorApi(): void {
  vi.clearAllMocks();
  const item = editorItem();
  editorApi.getImage.mockResolvedValue(imageDetail());
  editorApi.createImageItem.mockResolvedValue({ item: { ...item, id: 'new-1' } });
  editorApi.patchItemGeometry.mockResolvedValue({
    item: { ...item, geometry_revision: 3 },
  });
  editorApi.undoItemGeometry.mockResolvedValue({
    item: { ...item, geometry_revision: 4 },
  });
  for (const k of [
    'excludeCrops',
    'unexcludeCrops',
    'putCropLabel',
    'undoCropLabel',
  ] as const)
    editorApi[k].mockResolvedValue({});
  editorApi.undoCropRegion.mockResolvedValue({});
  editorApi.putRegionBoxes.mockResolvedValue({});
}
