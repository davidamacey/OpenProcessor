/**
 * `SourceImageOverlay`'s module-level crop-context cache must be keyed by
 * (project, crop id), not crop id alone — crop ids are content-derived,
 * so the same image gets the same `crop_id` in every project
 * (`docs/design/any-domain-rev3-and-projects-contract-review-2026-09-26.md`
 * §7). Otherwise switching projects would render another project's
 * cached source-image context for a colliding id.
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
  activeProjectKey: vi.fn(() => 'default'),
}));

const { getCropContext, activeProjectKey } = await import('$lib/api');
const { default: SourceImageOverlay, resetForProjectChange } =
  await import('./SourceImageOverlay.svelte');

function item(overrides: Partial<Crop> = {}): Crop {
  return {
    id: 'crop-1',
    class_name: null,
    proposed_class_name: null,
    bbox_norm: { cx: 0.3, cy: 0.3, w: 0.2, h: 0.2 },
    ...overrides,
  } as unknown as Crop;
}

function ctxFor(label: string): CropContextResponse {
  return {
    image: {
      image_id: `img-${label}`,
      image_path: `/${label}.jpg`,
      width: 100,
      height: 100,
      source: null,
      indexed_at: null,
    },
    items: [item({ id: 'shared-crop-id', class_name: label })],
  };
}

let instance: unknown;
let target: HTMLDivElement;

beforeEach(() => {
  installDeploymentSlots([widgetTagSlot]);
  resetForProjectChange();
  vi.mocked(activeProjectKey).mockReturnValue('default');
});

afterEach(() => {
  if (instance) unmount(instance as Record<string, unknown>);
  instance = undefined;
  target?.remove();
  resetDeploymentSlots();
  resetForProjectChange();
  vi.mocked(getCropContext).mockReset();
  vi.mocked(activeProjectKey).mockReset();
});

async function render(
  props: { cropId: string } & Record<string, unknown>,
): Promise<HTMLDivElement> {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(SourceImageOverlay, { target, props: props as never });
  flushSync();
  await Promise.resolve();
  await Promise.resolve();
  flushSync();
  return target;
}

describe('SourceImageOverlay crop-context cache, keyed by project', () => {
  it('fetches twice for the same crop id across two different projects', async () => {
    vi.mocked(getCropContext).mockImplementation(() =>
      Promise.resolve(
        ctxFor(vi.mocked(activeProjectKey).mock.results.at(-1)?.value ?? 'x'),
      ),
    );

    vi.mocked(activeProjectKey).mockReturnValue('proj-a');
    await render({ cropId: 'shared-crop-id' });
    unmount(instance as Record<string, unknown>);
    instance = undefined;
    target.remove();

    vi.mocked(activeProjectKey).mockReturnValue('proj-b');
    await render({ cropId: 'shared-crop-id' });

    expect(getCropContext).toHaveBeenCalledTimes(2);
  });

  it('does NOT refetch the same crop id within the same project (baseline caching still works)', async () => {
    vi.mocked(getCropContext).mockResolvedValue(ctxFor('same'));
    vi.mocked(activeProjectKey).mockReturnValue('proj-a');

    await render({ cropId: 'shared-crop-id' });
    unmount(instance as Record<string, unknown>);
    instance = undefined;
    target.remove();

    await render({ cropId: 'shared-crop-id' });

    expect(getCropContext).toHaveBeenCalledTimes(1);
  });

  it('resetForProjectChange() clears the cache: even the SAME project refetches afterwards', async () => {
    vi.mocked(getCropContext).mockResolvedValue(ctxFor('same'));
    vi.mocked(activeProjectKey).mockReturnValue('proj-a');

    await render({ cropId: 'shared-crop-id' });
    unmount(instance as Record<string, unknown>);
    instance = undefined;
    target.remove();

    resetForProjectChange();

    await render({ cropId: 'shared-crop-id' });

    expect(getCropContext).toHaveBeenCalledTimes(2);
  });
});
