/**
 * `SourceImageOverlay`'s module-level image-detail cache must be keyed by
 * (project, image id), not image id alone — image ids are content-derived,
 * so the same image gets the same id in every project (projects contract
 * review §7). Otherwise switching projects would render another project's
 * cached detail for a colliding id.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import type { Crop } from '$lib/types';
import { imageDetail } from '$lib/test/fixtures/imageDetail';
import {
  installDeploymentSlots,
  resetDeploymentSlots,
} from '$lib/annotations/registeredSlots';
import { widgetTagSlot } from '$lib/test/fixtures/regionSlot';

vi.mock('$lib/api', () => ({ activeProjectKey: vi.fn(() => 'default') }));
vi.mock('$lib/api_cropRead', () => ({
  getImage: vi.fn(),
  getImageFileUrl: (id: string) => `/image/${id}`,
}));

const { activeProjectKey } = await import('$lib/api');
const { getImage } = await import('$lib/api_cropRead');
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

function ctxFor(label: string) {
  return imageDetail({
    image: { image_id: `img-${label}` },
    items: [item({ id: 'shared-crop-id', class_name: label })],
  });
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
  vi.mocked(getImage).mockReset();
  vi.mocked(activeProjectKey).mockReset();
});

async function render(
  props: { imageId: string } & Record<string, unknown>,
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

describe('SourceImageOverlay image-detail cache, keyed by project', () => {
  it('fetches twice for the same image id across two different projects', async () => {
    vi.mocked(getImage).mockImplementation(() =>
      Promise.resolve(
        ctxFor(vi.mocked(activeProjectKey).mock.results.at(-1)?.value ?? 'x'),
      ),
    );

    vi.mocked(activeProjectKey).mockReturnValue('proj-a');
    await render({ imageId: 'img-shared' });
    unmount(instance as Record<string, unknown>);
    instance = undefined;
    target.remove();

    vi.mocked(activeProjectKey).mockReturnValue('proj-b');
    await render({ imageId: 'img-shared' });

    expect(getImage).toHaveBeenCalledTimes(2);
  });

  it('does NOT refetch the same image id within the same project (baseline caching still works)', async () => {
    vi.mocked(getImage).mockResolvedValue(ctxFor('same'));
    vi.mocked(activeProjectKey).mockReturnValue('proj-a');

    await render({ imageId: 'img-shared' });
    unmount(instance as Record<string, unknown>);
    instance = undefined;
    target.remove();

    await render({ imageId: 'img-shared' });

    expect(getImage).toHaveBeenCalledTimes(1);
  });

  it('resetForProjectChange() clears the cache: even the SAME project refetches afterwards', async () => {
    vi.mocked(getImage).mockResolvedValue(ctxFor('same'));
    vi.mocked(activeProjectKey).mockReturnValue('proj-a');

    await render({ imageId: 'img-shared' });
    unmount(instance as Record<string, unknown>);
    instance = undefined;
    target.remove();

    resetForProjectChange();

    await render({ imageId: 'img-shared' });

    expect(getImage).toHaveBeenCalledTimes(2);
  });
});
