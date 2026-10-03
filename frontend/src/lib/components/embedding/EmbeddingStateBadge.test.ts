import { afterEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import EmbeddingStateBadge from './EmbeddingStateBadge.svelte';
import EmbeddingRows from '../provenance/EmbeddingRows.svelte';
import { mapRawCrop } from '$lib/api';
import { makeItem } from '$lib/test/makeItem';
import type { EmbeddingState } from '$lib/types_itemFilter';

let target: HTMLElement;
let instance: Record<string, unknown> | undefined;

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

function badge(state: EmbeddingState | null, compact = false) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(EmbeddingStateBadge, { target, props: { state, compact } });
  flushSync();
  return target.querySelector('[data-testid="embedding-state-badge"]');
}

describe('EmbeddingStateBadge', () => {
  it('names a failed encoder with the warning tone and a tooltip', () => {
    const el = badge('failed')!;
    expect(el.textContent).toBe('No vector: encoder failed');
    expect(el.getAttribute('title')).toContain('run Embed to retry');
    expect(el.className).toContain('amber');
  });

  it('names a deferred item', () => {
    expect(badge('deferred')!.textContent).toBe('No vector yet');
  });

  it('names an item the policy did not select, in a neutral tone', () => {
    const el = badge('not_selected')!;
    expect(el.textContent).toBe('Not embedded');
    expect(el.className).not.toContain('amber');
  });

  it('renders nothing for embedded or an unrecorded state', () => {
    expect(badge('embedded')).toBeNull();
    target.remove();
    expect(badge(null)).toBeNull();
  });
});

describe('EmbeddingRows', () => {
  function rows(embedding_state: EmbeddingState | null) {
    target = document.createElement('dl');
    document.body.appendChild(target);
    instance = mount(EmbeddingRows, {
      target,
      props: { crop: mapRawCrop(makeItem({ embedding_state })) },
    });
    flushSync();
    return target.querySelector('[data-testid="embedding-row"]')!.textContent!.trim();
  }

  it('reads embedded, the named state, or unknown for null', () => {
    expect(rows('embedded')).toBe('embedded');
    expect(rows('failed')).toBe('No vector: encoder failed');
    target.remove();
    unmount(instance!);
    expect(rows(null)).toBe('unknown (written before v0.4.0)');
  });
});
