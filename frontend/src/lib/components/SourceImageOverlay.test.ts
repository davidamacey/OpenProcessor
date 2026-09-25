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

vi.mock('$lib/api', () => ({
  getCropContext: vi.fn(),
  getSourceImageScaled: (id: string) => `/image/${id}`,
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

  it('draws a solid region box and a dashed candidate box from slot data', async () => {
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
              subBox: {
                // Region box centered in the top-left quadrant of the item,
                // parent-frame-normalized (cx/cy/w/h relative to the item box).
                parent: { cx: 0.25, cy: 0.25, w: 0.2, h: 0.2 },
                rawXyxy: null,
                frame: 'source',
                score: 0.9,
                visible: true,
                candidate: {
                  parent: { cx: 0.75, cy: 0.75, w: 0.2, h: 0.2 },
                  rawXyxy: null,
                  score: 0.4,
                  detector: null,
                  detectorVersion: null,
                  source: null,
                },
              },
            },
          },
        } as unknown as Crop,
      ],
    };
    vi.mocked(getCropContext).mockResolvedValue(ctx);

    const el = await render({ cropId: 'region-a' });

    const region = el.querySelector(
      '[data-testid="overlay-box"][data-kind="region"]',
    ) as HTMLElement;
    const candidate = el.querySelector(
      '[data-testid="overlay-box"][data-kind="region-candidate"]',
    ) as HTMLElement;
    expect(region).toBeTruthy();
    expect(candidate).toBeTruthy();
    expect(region.className).not.toContain('border-dashed');
    expect(candidate.className).toContain('border-dashed');
    expect(region.getAttribute('title')).toContain('Tag');
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
});
