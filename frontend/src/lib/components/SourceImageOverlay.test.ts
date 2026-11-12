/**
 * Mount test for `SourceImageOverlay` (K6,
 * K6 frontend overlay plan) — the boxes/
 * labels this component draws are the ONLY place they exist once the
 * backend serves only clean pixels at `/images/{id}/file`, so every claim
 * here is pinned against the actual rendered `%` geometry, not just
 * "it mounted".
 *
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import type { Crop } from '$lib/types';
import { imageBox, imageDetail } from '$lib/test/fixtures/imageDetail';
import { regionStatusesStore } from '$stores/regionStatuses.svelte';

vi.mock('$lib/api', () => ({ activeProjectKey: vi.fn(() => 'default') }));
vi.mock('$lib/api_cropRead', () => ({
  getImage: vi.fn(),
  getImageFileUrl: (id: string) => `/image/${id}`,
}));

const { getImage } = await import('$lib/api_cropRead');
const { default: SourceImageOverlay, resetForProjectChange } =
  await import('./SourceImageOverlay.svelte');

function item(overrides: Partial<Crop> = {}): Crop {
  return {
    id: 'crop-1',
    class_name: null,
    proposed_class_name: null,
    display_name: '',
    display_name_kind: 'none',
    bbox_norm: { cx: 0.3, cy: 0.3, w: 0.2, h: 0.2 },
    ...overrides,
  } as unknown as Crop;
}

let instance: unknown;
let target: HTMLDivElement;

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
  // The detail cache is keyed by image id and every test draws on the same one.
  resetForProjectChange();
  vi.mocked(getImage).mockReset();
});

async function render(
  props: { imageId: string } & Record<string, unknown>,
): Promise<HTMLDivElement> {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(SourceImageOverlay, { target, props: props as never });
  flushSync();
  // Let the $effect's context-fetch promise resolve, then flush again.
  await Promise.resolve();
  await Promise.resolve();
  flushSync();
  return target;
}

describe('SourceImageOverlay', () => {
  it('draws the served boxes, not item-derived labels', async () => {
    // The items carry names that must NOT appear: the boxes are the only source.
    const ctx = imageDetail({
      items: [
        item({ id: 'crop-pos', display_name: 'from_items', display_name_kind: 'class' }),
      ],
      boxes: [
        imageBox({
          crop_id: 'crop-pos',
          bbox_norm: [0.4, 0.35, 0.6, 0.45],
          label: 'served_name',
          tone: 'machine',
          label_kind: 'proposed',
        }),
      ],
    });
    vi.mocked(getImage).mockResolvedValue(ctx);

    const el = await render({ imageId: 'img-1', selectedCropId: 'crop-pos' });

    const boxes = el.querySelectorAll<HTMLElement>('[data-testid="overlay-box"]');
    expect(boxes).toHaveLength(1);
    expect(boxes[0]!.style.left).toBe('40%');
    expect(boxes[0]!.style.top).toBe('35%');
    expect(boxes[0]!.style.width).toBe('20%');
    expect(boxes[0]!.style.height).toBe('10%');
    expect(boxes[0]!.className).toContain('border-amber-400');
    expect(boxes[0]!.getAttribute('title')).toBe('served_name');
    expect(el.textContent).not.toContain('from_items');
  });

  it('draws a region box with its served name and dashes it by its served state', async () => {
    regionStatusesStore.boxStates = [
      {
        value: 'rejected',
        label: 'rejected',
        role: 'rejected',
        human_writable: true,
        exported: false,
        dashed: true,
        dim: false,
        badge: null,
        tone: 'rejected',
      },
    ];
    const ctx = imageDetail({
      boxes: [
        imageBox({
          crop_id: 'region-a',
          box_id: 'b1',
          kind: 'region',
          bbox_norm: [0.1, 0.2, 0.3, 0.4],
          label: 'Tag 1',
          label_kind: 'region',
          tone: 'rejected',
          state: 'rejected',
        }),
      ],
    });
    vi.mocked(getImage).mockResolvedValue(ctx);

    const el = await render({ imageId: 'img-1', selectedCropId: 'region-a' });

    const box = el.querySelector<HTMLElement>('[data-kind="region"]')!;
    expect(box.className).toContain('border-red-400');
    expect(box.className).toContain('border-dashed');
    expect(box.getAttribute('title')).toBe('Tag 1');
    regionStatusesStore.boxStates = [];
  });

  it('fires onselect when a sibling box is clicked, not for the selected one', async () => {
    const ctx = imageDetail({
      boxes: [imageBox({ crop_id: 'click-a' }), imageBox({ crop_id: 'click-b' })],
    });
    vi.mocked(getImage).mockResolvedValue(ctx);
    const onselect = vi.fn();

    const el = await render({ imageId: 'img-1', selectedCropId: 'click-a', onselect });
    const byId = (id: string) => el.querySelector<HTMLElement>(`[data-crop-id="${id}"]`)!;

    expect(byId('click-a').tagName).toBe('DIV');
    expect(byId('click-b').tagName).toBe('BUTTON');
    byId('click-b').click();
    expect(onselect).toHaveBeenCalledWith('click-b');
  });

  it('the boxes toggle hides and re-shows the overlay layer', async () => {
    vi.mocked(getImage).mockResolvedValue(imageDetail({ boxes: [imageBox()] }));

    const el = await render({ imageId: 'img-1', selectedCropId: 'crop-1' });
    expect(el.querySelector('[data-testid="overlay-layer"]')).toBeTruthy();

    const toggle = el.querySelector('button') as HTMLButtonElement;
    toggle.click();
    flushSync();
    expect(el.querySelector('[data-testid="overlay-layer"]')).toBeFalsy();

    toggle.click();
    flushSync();
    expect(el.querySelector('[data-testid="overlay-layer"]')).toBeTruthy();
  });

  it('skips its own fetch when a pre-fetched context is provided', async () => {
    const el = await render({
      imageId: 'img-1',
      selectedCropId: 'crop-1',
      context: imageDetail({ boxes: [imageBox()] }),
    });

    expect(getImage).not.toHaveBeenCalled();
    expect(el.querySelectorAll('[data-testid="overlay-box"]').length).toBe(1);
  });
});

describe('SourceImageOverlay extraShapes (W5 test-on-crop candidates)', () => {
  const ctx = () => imageDetail({ boxes: [imageBox()] });

  const shapes = [
    {
      key: 'detector:0:box',
      kind: 'box' as const,
      box: [0.1, 0.2, 0.5, 0.6] as [number, number, number, number],
      dimmed: false,
      label: 'detector #0',
      title: 'detector #0',
    },
    {
      key: 'detector:1:box',
      kind: 'box' as const,
      box: [0.6, 0.6, 0.9, 0.9] as [number, number, number, number],
      dimmed: true,
      label: 'detector #1',
      title: 'detector #1 · dropped: Below min score',
    },
    {
      key: 'segmenter:0:poly',
      kind: 'polygon' as const,
      points: [
        [0.2, 0.2],
        [0.6, 0.2],
        [0.4, 0.5],
      ] as [number, number][],
      dimmed: true,
      label: 'segmenter #0',
      title: 'segmenter #0',
    },
    {
      key: 'segmenter:1:poly',
      kind: 'polygon' as const,
      points: [
        [0.3, 0.3],
        [0.7, 0.3],
        [0.5, 0.8],
      ] as [number, number][],
      dimmed: false,
      label: 'segmenter #1',
      title: 'segmenter #1',
    },
  ];

  it('draws extra boxes at the served percentages, dimmed ones faint and dashed', async () => {
    vi.mocked(getImage).mockResolvedValue(ctx());
    const el = await render({
      imageId: 'img-1',
      selectedCropId: 'crop-1',
      extraShapes: shapes,
    });
    const boxes = [
      ...el.querySelectorAll('[data-testid="overlay-extra-box"]'),
    ] as HTMLElement[];
    expect(boxes).toHaveLength(2);
    expect(boxes[0]!.style.left).toBe('10%');
    expect(boxes[0]!.style.top).toBe('20%');
    expect(boxes[0]!.style.width).toBe('40%');
    expect(boxes[0]!.style.height).toBe('40%');
    expect(boxes[0]!.dataset.dimmed).toBe('false');
    expect(boxes[0]!.className).not.toContain('opacity-40');
    expect(boxes[1]!.dataset.dimmed).toBe('true');
    expect(boxes[1]!.className).toContain('opacity-40');
    expect(boxes[1]!.className).toContain('border-dashed');
    expect(boxes[1]!.getAttribute('title')).toContain('Below min score');
    expect(boxes[0]!.textContent).toContain('detector #0');
  });

  it('draws mask outlines as polygons over a unit viewBox, dimmed ones faint', async () => {
    vi.mocked(getImage).mockResolvedValue(ctx());
    const el = await render({
      imageId: 'img-1',
      selectedCropId: 'crop-1',
      extraShapes: shapes,
    });
    const polys = [
      ...el.querySelectorAll('[data-testid="overlay-extra-polygon"]'),
    ] as SVGPolygonElement[];
    expect(polys).toHaveLength(2);
    expect(polys[0]!.getAttribute('points')).toBe('0.2,0.2 0.6,0.2 0.4,0.5');
    expect(polys[0]!.dataset.dimmed).toBe('true');
    expect(polys[0]!.closest('svg')!.getAttribute('class')).toContain('opacity-40');
    expect(polys[1]!.dataset.dimmed).toBe('false');
    expect(polys[1]!.closest('svg')!.getAttribute('class')).not.toContain('opacity-40');
    expect(polys[0]!.closest('svg')!.getAttribute('viewBox')).toBe('0 0 1 1');
  });

  it('draws no extra shapes by default, and hides them with the boxes toggle', async () => {
    vi.mocked(getImage).mockResolvedValue(ctx());
    const plain = await render({ imageId: 'img-1', selectedCropId: 'crop-1' });
    expect(plain.querySelector('[data-testid="overlay-extra-box"]')).toBeNull();
    expect(plain.querySelector('[data-testid="overlay-extra-polygon"]')).toBeNull();
    unmount(instance as never);
    instance = undefined;
    plain.remove();

    const el = await render({
      imageId: 'img-1',
      selectedCropId: 'crop-1',
      extraShapes: shapes,
    });
    expect(el.querySelector('[data-testid="overlay-extra-box"]')).not.toBeNull();
    const toggle = [...el.querySelectorAll('button')].find((b) =>
      b.textContent?.includes('hide boxes'),
    ) as HTMLButtonElement;
    toggle.click();
    flushSync();
    expect(el.querySelector('[data-testid="overlay-extra-box"]')).toBeNull();
    expect(el.querySelector('[data-testid="overlay-extra-polygon"]')).toBeNull();
  });
});
