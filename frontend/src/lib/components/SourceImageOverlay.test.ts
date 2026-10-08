/**
 * Mount test for `SourceImageOverlay` (K6,
 * docs/design/k6-frontend-overlay-plan-2026-09-24.md) — the boxes/
 * labels this component draws are the ONLY place they exist once the
 * backend stops burning them into `/crops/{id}/image`, so every claim
 * here is pinned against the actual rendered `%` geometry, not just
 * "it mounted".
 *
 * Uses the neutral `widgetTagSlot` fixture, not a real registered slot —
 * this file is scanned by `domainNeutral.scan.test.ts`.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import type { Crop, CropContextResponse } from '$lib/types';
import {
  installDeploymentSlots,
  resetDeploymentSlots,
} from '$lib/annotations/registeredSlots';
import { widgetTagSlot } from '$lib/test/fixtures/regionSlot';
import { aircraftTailNumberSlot } from '$lib/test/fixtures/aircraftTailNumberSlot';
import { makeSlotBox } from '$lib/test/fixtures/slotBox';
import { regionStatusesStore } from '$stores/regionStatuses.svelte';

vi.mock('$lib/api', () => ({
  getCropContext: vi.fn(),
  getSourceImageScaled: (id: string) => `/image/${id}`,
  activeProjectKey: vi.fn(() => 'default'),
}));

const { getCropContext } = await import('$lib/api');
const { default: SourceImageOverlay } = await import('./SourceImageOverlay.svelte');

function item(overrides: Partial<Crop> = {}): Crop {
  return {
    id: 'crop-1',
    class_name: null,
    proposed_class_name: null,
    bbox_norm: { cx: 0.3, cy: 0.3, w: 0.2, h: 0.2 },
    ...overrides,
  } as unknown as Crop;
}

let instance: unknown;
let target: HTMLDivElement;

beforeEach(() => {
  installDeploymentSlots([widgetTagSlot]);
});

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
  resetDeploymentSlots();
  vi.mocked(getCropContext).mockReset();
});

async function render(
  props: { cropId: string } & Record<string, unknown>,
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
  it('positions the item box at the bbox_norm percentages', async () => {
    const ctx: CropContextResponse = {
      image: {
        image_id: 'img-1',
        image_path: '/img-1.jpg',
        width: 1000,
        height: 1000,
        source: 'test',
        indexed_at: null,
      },
      items: [
        item({
          id: 'crop-pos',
          class_name: 'widget_a',
          bbox_norm: { cx: 0.5, cy: 0.4, w: 0.2, h: 0.1 },
        }),
      ],
    };
    vi.mocked(getCropContext).mockResolvedValue(ctx);

    const el = await render({ cropId: 'crop-pos' });

    const boxes = el.querySelectorAll('[data-testid="overlay-box"][data-kind="item"]');
    expect(boxes.length).toBe(1);
    const style = (boxes[0] as HTMLElement).style;
    // cx=0.5,w=0.2 -> left = (0.5-0.1)*100 = 40%; cy=0.4,h=0.1 -> top = 35%
    expect(style.left).toBe('40%');
    expect(style.top).toBe('35%');
    expect(style.width).toBe('20%');
    expect(style.height).toBe('10%');
  });

  it('labels a labelled item, a proposed item, and an unlabeled item distinctly', async () => {
    const ctx: CropContextResponse = {
      image: {
        image_id: 'i',
        image_path: '/i.jpg',
        width: 100,
        height: 100,
        source: null,
        indexed_at: null,
      },
      items: [
        item({ id: 'lbl-a', class_name: 'widget_a' }),
        item({ id: 'lbl-b', class_name: null, proposed_class_name: 'widget_b' }),
        item({ id: 'lbl-c', class_name: null, proposed_class_name: null }),
      ],
    };
    vi.mocked(getCropContext).mockResolvedValue(ctx);

    const el = await render({ cropId: 'lbl-a', selectedCropId: 'lbl-a' });

    const boxes = Array.from(
      el.querySelectorAll('[data-testid="overlay-box"][data-kind="item"]'),
    );
    const byId = (id: string) =>
      boxes.find((b) => b.getAttribute('data-crop-id') === id) as HTMLElement;

    expect(byId('lbl-a').className).toContain('border-emerald-400');
    expect(byId('lbl-a').getAttribute('title')).toContain('widget_a');

    expect(byId('lbl-b').className).toContain('border-amber-400');
    expect(byId('lbl-b').getAttribute('title')).toContain('widget_b (proposed)');

    expect(byId('lbl-c').className).toContain('border-zinc-500');
    expect(byId('lbl-c').getAttribute('title')).toContain('unlabeled');
  });

  it('draws every served box at its source-frame position, solid or dashed by state', async () => {
    regionStatusesStore.boxStates = [
      {
        value: 'accepted',
        label: 'accepted',
        role: 'accepted',
        human_writable: true,
        exported: true,
        dashed: false,
        dim: false,
        badge: null,
        tone: 'accepted',
      },
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
    const ctx: CropContextResponse = {
      image: {
        image_id: 'i',
        image_path: '/i.jpg',
        width: 100,
        height: 100,
        source: null,
        indexed_at: null,
      },
      items: [
        {
          ...item({
            id: 'region-a',
            class_name: 'widget_a',
            bbox_norm: { cx: 0.5, cy: 0.5, w: 0.4, h: 0.4 },
          }),
          slots: {
            widget_tag: {
              key: 'widget_tag',
              subBoxes: [
                // bbox_norm is already source-image-normalized: drawn as is,
                // never re-projected through the item's own box.
                makeSlotBox({
                  boxId: 'b1',
                  state: 'accepted',
                  rawXyxy: [0.1, 0.2, 0.3, 0.4],
                  parent: { cx: 0.25, cy: 0.25, w: 0.2, h: 0.2 },
                  score: 0.9,
                }),
                makeSlotBox({
                  boxId: 'b2',
                  state: 'rejected',
                  rawXyxy: [0.6, 0.6, 0.8, 0.9],
                  parent: { cx: 0.75, cy: 0.75, w: 0.2, h: 0.2 },
                  score: 0.4,
                }),
              ],
            },
          },
        } as unknown as Crop,
      ],
    };
    vi.mocked(getCropContext).mockResolvedValue(ctx);

    const el = await render({ cropId: 'region-a' });

    const [first, second] = Array.from(
      el.querySelectorAll('[data-testid="overlay-box"][data-kind="region-box"]'),
    ) as HTMLElement[];
    expect(first).toBeTruthy();
    expect(second).toBeTruthy();
    expect(first.className).not.toContain('border-dashed');
    expect(second.className).toContain('border-dashed');
    expect(first.style.left).toBe('10%');
    expect(first.style.top).toBe('20%');
    expect(first.style.width).toBe('20%');
    expect(first.getAttribute('title')).toContain('Tag 1');
    expect(second.getAttribute('title')).toContain('Tag 2');
    regionStatusesStore.boxStates = [];
  });

  it('draws a read-only scalar-box slot through the item box when it is stored in the parent frame', async () => {
    const ctx: CropContextResponse = {
      image: {
        image_id: 'i',
        image_path: '/i.jpg',
        width: 100,
        height: 100,
        source: null,
        indexed_at: null,
      },
      items: [
        {
          ...item({
            id: 'tail-a',
            class_name: 'widget_a',
            // Item box: x 0.2..0.6, y 0.2..0.6.
            bbox_norm: { cx: 0.4, cy: 0.4, w: 0.4, h: 0.4 },
          }),
          slots: {
            aircraft_tail_number: {
              key: 'aircraft_tail_number',
              subBox: {
                // Parent-frame box: right half of the item box.
                rawXyxy: [0.5, 0, 1, 1],
                frame: 'parent',
                parent: { cx: 0.75, cy: 0.5, w: 0.5, h: 1 },
                score: 0.8,
                visible: true,
              },
            },
          },
        } as unknown as Crop,
      ],
    };
    vi.mocked(getCropContext).mockResolvedValue(ctx);
    installDeploymentSlots([widgetTagSlot, aircraftTailNumberSlot]);

    const el = await render({ cropId: 'tail-a' });
    resetDeploymentSlots();

    const region = el.querySelector(
      '[data-testid="overlay-box"][data-kind="region"]',
    ) as HTMLElement;
    expect(region).toBeTruthy();
    expect(region.style.left).toBe('40%');
    expect(region.style.width).toBe('20%');
  });

  it('renders the selected item with a highlight ring and dims the rest', async () => {
    const ctx: CropContextResponse = {
      image: {
        image_id: 'i',
        image_path: '/i.jpg',
        width: 100,
        height: 100,
        source: null,
        indexed_at: null,
      },
      items: [
        item({ id: 'sel-a', class_name: 'widget_a' }),
        item({ id: 'sel-b', class_name: 'widget_b' }),
      ],
    };
    vi.mocked(getCropContext).mockResolvedValue(ctx);

    const el = await render({ cropId: 'sel-a', selectedCropId: 'sel-a' });
    const boxes = Array.from(
      el.querySelectorAll('[data-testid="overlay-box"][data-kind="item"]'),
    );
    const byId = (id: string) =>
      boxes.find((b) => b.getAttribute('data-crop-id') === id) as HTMLElement;

    expect(byId('sel-a').className).toContain('ring-2');
    expect(byId('sel-a').className).not.toContain('opacity-60');
    expect(byId('sel-b').className).toContain('opacity-60');
  });

  it('fires onselect when a clickable sibling box is clicked, not for the selected one', async () => {
    const ctx: CropContextResponse = {
      image: {
        image_id: 'i',
        image_path: '/i.jpg',
        width: 100,
        height: 100,
        source: null,
        indexed_at: null,
      },
      items: [
        item({ id: 'click-a', class_name: 'widget_a' }),
        item({ id: 'click-b', class_name: 'widget_b' }),
      ],
    };
    vi.mocked(getCropContext).mockResolvedValue(ctx);
    const onselect = vi.fn();

    const el = await render({ cropId: 'click-a', selectedCropId: 'click-a', onselect });
    const boxes = Array.from(
      el.querySelectorAll('[data-testid="overlay-box"][data-kind="item"]'),
    );
    const byId = (id: string) =>
      boxes.find((b) => b.getAttribute('data-crop-id') === id) as HTMLElement;

    expect(byId('click-a').tagName).toBe('DIV'); // selected/unclickable stays a div
    expect(byId('click-b').tagName).toBe('BUTTON');
    byId('click-b').click();
    expect(onselect).toHaveBeenCalledWith('click-b');
  });

  it('the boxes toggle hides and re-shows the overlay layer', async () => {
    const ctx: CropContextResponse = {
      image: {
        image_id: 'i',
        image_path: '/i.jpg',
        width: 100,
        height: 100,
        source: null,
        indexed_at: null,
      },
      items: [item({ id: 'toggle-a', class_name: 'widget_a' })],
    };
    vi.mocked(getCropContext).mockResolvedValue(ctx);

    const el = await render({ cropId: 'toggle-a' });
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
    const ctx: CropContextResponse = {
      image: {
        image_id: 'i',
        image_path: '/i.jpg',
        width: 100,
        height: 100,
        source: null,
        indexed_at: null,
      },
      items: [item({ id: 'prefetch-a', class_name: 'widget_a' })],
    };

    const el = await render({ cropId: 'prefetch-a', context: ctx });

    expect(getCropContext).not.toHaveBeenCalled();
    expect(el.querySelectorAll('[data-testid="overlay-box"]').length).toBe(1);
  });

  it('draws a W8 multi-box region ring in the served box_states tone, not the client role fallback', async () => {
    // Backend follow-up to W8.7 (feat/w8-multibox-lockstep,
    // docs/design/w8-multibox-frontend-plan-2026-09-26.md): each
    // box_states entry now serves `tone`. A served `rejected` tone must
    // win over the role→color guess this file used before (which mapped
    // a `rejected` box to the same neutral zinc as `false_positive`).
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

    const ctx: CropContextResponse = {
      image: {
        image_id: 'i',
        image_path: '/i.jpg',
        width: 100,
        height: 100,
        source: null,
        indexed_at: null,
      },
      items: [
        {
          ...item({
            id: 'region-mb',
            class_name: 'widget_a',
            bbox_norm: { cx: 0.5, cy: 0.5, w: 0.4, h: 0.4 },
          }),
          slots: {
            widget_tag: {
              key: 'widget_tag',
              subBoxes: [
                makeSlotBox({
                  boxId: 'b1',
                  state: 'rejected',
                  rawXyxy: [0.1, 0.1, 0.3, 0.3],
                }),
              ],
            },
          },
        } as unknown as Crop,
      ],
    };
    vi.mocked(getCropContext).mockResolvedValue(ctx);

    const el = await render({ cropId: 'region-mb' });

    const box = el.querySelector(
      '[data-testid="overlay-box"][data-kind="region-box"]',
    ) as HTMLElement;
    expect(box).toBeTruthy();
    expect(box.className).toContain('border-red-400');
    expect(box.className).not.toContain('border-zinc-500');

    regionStatusesStore.boxStates = [];
  });
});

describe('SourceImageOverlay extraShapes (W5 test-on-crop candidates)', () => {
  const ctx = (): CropContextResponse => ({
    image: {
      image_id: 'i',
      image_path: '/i.jpg',
      width: 100,
      height: 100,
      source: null,
      indexed_at: null,
    },
    items: [item({ id: 'crop-1' })],
  });

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
    vi.mocked(getCropContext).mockResolvedValue(ctx());
    const el = await render({ cropId: 'crop-1', extraShapes: shapes });
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
    vi.mocked(getCropContext).mockResolvedValue(ctx());
    const el = await render({ cropId: 'crop-1', extraShapes: shapes });
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
    vi.mocked(getCropContext).mockResolvedValue(ctx());
    const plain = await render({ cropId: 'crop-1' });
    expect(plain.querySelector('[data-testid="overlay-extra-box"]')).toBeNull();
    expect(plain.querySelector('[data-testid="overlay-extra-polygon"]')).toBeNull();
    unmount(instance as never);
    instance = undefined;
    plain.remove();

    const el = await render({ cropId: 'crop-1', extraShapes: shapes });
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
