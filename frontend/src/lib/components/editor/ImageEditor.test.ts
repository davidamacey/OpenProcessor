import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { imageBox, imageDetail } from '$lib/test/fixtures/imageDetail';
import { pointer, stubCanvasLayout } from '$lib/test/canvasLayout';
import { editorApi as api, editorItem, resetEditorApi } from '$lib/editor/editorApiMock';
import { classesStore } from '$stores/classes.svelte';
import type { RegistryClass } from '$lib/types_classes';

const nav = vi.hoisted(() => ({ goto: vi.fn() }));
vi.mock('$app/navigation', () => ({ goto: nav.goto }));
vi.mock('$app/paths', () => ({ resolve: (p: string) => p }));
vi.mock('$lib/api_images', async () =>
  (await import('$lib/editor/editorApiMock')).editorImagesModule(),
);
vi.mock('$lib/api_cropWrites', async () =>
  (await import('$lib/editor/editorApiMock')).editorWritesModule(),
);
vi.mock('$lib/api_regionBoxes', async () =>
  (await import('$lib/editor/editorApiMock')).editorRegionsModule(),
);

const { default: ImageEditor } = await import('./ImageEditor.svelte');

const W = 800;
const H = 600;
const cls = (id: number, name: string): RegistryClass =>
  ({ id, name, group: 'g', count: 1, validated_count: 0, kind: 'item' }) as RegistryClass;

let target: HTMLDivElement;
let instance: Record<string, unknown> | undefined;

const c1 = editorItem({ crop_id: 'c1' });
const served = () =>
  imageDetail({
    image: { image_id: 'img-1', file_name: 'street.jpg', width: W, height: H },
    items: [c1],
    boxes: [imageBox({ crop_id: 'c1', bbox_norm: [0.1, 0.1, 0.3, 0.3], label: 'bus' })],
  });

const SEARCH = '?view=gallery&class_name=bus';

async function render(search = SEARCH) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(ImageEditor, { target, props: { imageId: 'img-1', search } });
  flushSync();
  await vi.waitFor(() =>
    expect(target.querySelector('[data-testid="editor-canvas"]')).not.toBeNull(),
  );
  await vi.waitFor(() => expect(api.listImages).toHaveBeenCalled());
  await Promise.resolve();
  flushSync();
}

function press(init: KeyboardEventInit & { key: string }): void {
  window.dispatchEvent(
    new KeyboardEvent('keydown', { bubbles: true, cancelable: true, ...init }),
  );
  flushSync();
}

const canvas = () => target.querySelector('[data-testid="editor-canvas"]') as HTMLElement;

beforeEach(() => {
  stubCanvasLayout(W, H);
  resetEditorApi();
  nav.goto.mockReset();
  classesStore.classes = [cls(7, 'widget'), cls(8, 'bus')];
  api.getImage.mockImplementation(async () => served());
  api.locateImage.mockResolvedValue({
    image_id: 'img-1',
    in_view: true,
    index: 4,
    total: 20,
    cursor_before: 'cb',
    cursor_after: 'ca',
    previous_image_id: 'img-0',
    reason: null,
    reason_message: null,
  });
  api.listImages.mockResolvedValue({
    items: [{ image_id: 'img-2' }, { image_id: 'img-3' }],
  });
});

afterEach(() => {
  vi.restoreAllMocks();
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
  classesStore.classes = [];
  document.body.innerHTML = '';
});

describe('ImageEditor', () => {
  it('shows the served image, its objects, the position and the neighbours', async () => {
    await render();
    expect(target.querySelector('h1')!.textContent).toBe('street.jpg');
    expect(
      target.querySelector('[data-testid="editor-position"]')!.textContent?.trim(),
    ).toBe('5 of 20');
    expect(
      target.querySelectorAll('[data-testid="object-list"] li[data-crop-id]'),
    ).toHaveLength(1);
    expect(
      [...target.querySelectorAll('a[data-image-id]')].map((a) =>
        a.getAttribute('data-image-id'),
      ),
    ).toEqual(['img-0', 'img-1', 'img-2', 'img-3']);
    expect(api.locateImage).toHaveBeenCalledWith(
      expect.objectContaining({ class_name: ['bus'] }),
      'img-1',
      expect.anything(),
    );
    expect(api.listImages).toHaveBeenCalledWith(
      expect.objectContaining({ class_name: ['bus'] }),
      'ca',
      expect.anything(),
    );
  });

  it('draws a box, opens the picker, and Enter saves it with the first class', async () => {
    await render();
    press({ key: 'b' });
    pointer(canvas(), 'pointerdown', 400, 300);
    pointer(canvas(), 'pointermove', 600, 450);
    pointer(canvas(), 'pointerup', 600, 450);
    flushSync();
    const input = document.querySelector('input[role="combobox"]') as HTMLInputElement;
    expect(input).not.toBeNull();
    expect(target.querySelector('[data-testid="pending-box"]')).not.toBeNull();
    input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    flushSync();
    await vi.waitFor(() => expect(api.createImageItem).toHaveBeenCalledOnce());
    const [imageId, body] = api.createImageItem.mock.calls[0];
    expect(imageId).toBe('img-1');
    expect(body.bbox_norm).toEqual([0.5, 0.5, 0.75, 0.75]);
    expect(typeof body.class_name).toBe('string');
    expect(body.request_id).toMatch(/^[0-9a-f-]{36}$/);
    expect(document.querySelector('input[role="combobox"]')).toBeNull();
  });

  it('delete then undo restores through exclude and unexclude', async () => {
    await render();
    press({ key: 'Tab' });
    press({ key: 'Delete' });
    await vi.waitFor(() =>
      expect(api.excludeCrops).toHaveBeenCalledWith(['c1'], 'not_an_object'),
    );
    press({ key: 'z', ctrlKey: true });
    await vi.waitFor(() => expect(api.unexcludeCrops).toHaveBeenCalledWith(['c1']));
  });

  it('walks to the next and previous image, keeping the Workspace query', async () => {
    await render();
    press({ key: '.' });
    expect(nav.goto).toHaveBeenLastCalledWith(`/p/default/images/img-2${SEARCH}`);
    press({ key: ',' });
    expect(nav.goto).toHaveBeenLastCalledWith(`/p/default/images/img-0${SEARCH}`);
  });

  it('Escape with nothing selected goes back to the same Workspace view', async () => {
    await render();
    press({ key: 'Escape' });
    expect(nav.goto).toHaveBeenLastCalledWith(`/p/default/workspace${SEARCH}`);
  });

  it('says why when the image cannot be read', async () => {
    api.getImage.mockRejectedValue(new Error('No such image.'));
    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(ImageEditor, { target, props: { imageId: 'img-1', search: '' } });
    await vi.waitFor(() => expect(target.textContent).toContain('No such image.'));
  });
});
