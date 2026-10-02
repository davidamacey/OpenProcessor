/**
 * The W9 provenance rows: endpoint (`name@revision`), model and prompt pack
 * render verbatim and only when the item carries them; an item with none
 * renders nothing.
 */
import { afterEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { mapRawCrop } from '$lib/api';
import type { Crop } from '$lib/types';
import VlmProvenanceRows from './VlmProvenanceRows.svelte';

let instance: ReturnType<typeof mount> | null = null;
let target: HTMLDListElement;

function render(crop: Partial<Crop>) {
  target = document.createElement('dl');
  document.body.appendChild(target);
  instance = mount(VlmProvenanceRows, { target, props: { crop: crop as Crop } });
  flushSync();
}

const q = (id: string) => target.querySelector(`[data-testid="vlm-provenance-${id}"]`);

afterEach(() => {
  if (instance) unmount(instance);
  instance = null;
  target?.remove();
});

describe('VlmProvenanceRows', () => {
  it('renders each served value verbatim', () => {
    render({
      vlm_endpoint: 'cloud_vlm@2',
      vlm_model: 'vendor/vision-large',
      vlm_prompt_pack: 'widget_pack',
    });
    expect(q('endpoint')?.textContent?.trim()).toBe('cloud_vlm@2');
    expect(q('model')?.textContent?.trim()).toBe('vendor/vision-large');
    expect(q('prompt-pack')?.textContent?.trim()).toBe('widget_pack');
    expect(target.textContent).toContain('VLM endpoint');
  });

  it('renders only the rows the item carries', () => {
    render({ vlm_endpoint: 'cloud_vlm@2', vlm_model: null, vlm_prompt_pack: null });
    expect(q('endpoint')).not.toBeNull();
    expect(q('model')).toBeNull();
    expect(q('prompt-pack')).toBeNull();
  });

  it('renders nothing for an item with no VLM provenance', () => {
    render({ vlm_endpoint: null, vlm_model: null, vlm_prompt_pack: null });
    expect(target.children).toHaveLength(0);
  });

  it('works from a served item through mapRawCrop', () => {
    const crop = mapRawCrop({
      crop_id: 'c1',
      vlm_endpoint: 'local_vlm@3',
      vlm_model: 'example/vision-7b',
      vlm_prompt_pack: null,
    } as never);
    render(crop);
    expect(q('endpoint')?.textContent?.trim()).toBe('local_vlm@3');
    expect(q('prompt-pack')).toBeNull();
  });
});
