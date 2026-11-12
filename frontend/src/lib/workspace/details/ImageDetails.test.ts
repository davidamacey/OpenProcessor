import { describe, expect, it, vi } from 'vitest';
import { flushSync, tick } from 'svelte';
import { mountUi } from '$lib/test/mountUi';
import { imageDetail } from '$lib/test/fixtures/imageDetail';

vi.mock('$app/state', async () => await import('$lib/test/pageState.svelte'));
vi.mock('$app/paths', () => ({ resolve: (p: string) => p }));
vi.mock('$lib/api_images', () => ({ getImage: vi.fn() }));

import { getImage } from '$lib/api_images';
import ImageDetails from './ImageDetails.svelte';

describe('ImageDetails', () => {
  it('titles the panel with the file name as a normal heading and opens the lightbox', async () => {
    vi.mocked(getImage).mockResolvedValue(imageDetail());
    const ws = { openLightbox: vi.fn(), setScope: vi.fn() };
    const el = mountUi(ImageDetails, { ws, id: 'img-1' });
    await tick();
    await tick();
    flushSync();
    const h = el.querySelector('h2')!;
    expect(h.className).not.toContain('uppercase');
    (el.querySelector('[data-testid="lightbox-image"]') as HTMLElement).click();
    expect(ws.openLightbox).toHaveBeenCalledWith('img-1', h.textContent?.trim());
  });

  it('opens the group decisions only for an image that is in a near-duplicate group', async () => {
    vi.mocked(getImage).mockResolvedValue(imageDetail());
    const plain = mountUi(ImageDetails, { ws: { setScope: vi.fn() }, id: 'img-1' });
    await tick();
    await tick();
    flushSync();
    expect(plain.querySelector('[data-testid="decide-near-dup"]')).toBeNull();

    vi.mocked(getImage).mockResolvedValue(
      imageDetail({
        image: {
          near_dup: {
            group_id: 'g1',
            purpose: 'dedup',
            size: 3,
            is_representative: false,
          },
        },
      }),
    );
    const grouped = mountUi(ImageDetails, { ws: { setScope: vi.fn() }, id: 'img-2' });
    await tick();
    await tick();
    flushSync();
    expect(grouped.querySelector('[data-testid="decide-near-dup"]')).not.toBeNull();
  });
});
