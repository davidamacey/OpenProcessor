import { afterEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import EmbeddingStateBadge from './EmbeddingStateBadge.svelte';
import EmbeddingRows from '../provenance/EmbeddingRows.svelte';
import { mapRawCrop } from '$lib/api_cropMap';
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
    expect(el.className).toContain('warn');
  });

  it('has a short compact form for card chips, the full wording in the tooltip', () => {
    const el = badge('failed', true)!;
    expect(el.textContent).toBe('Embed failed');
    expect(el.getAttribute('title')).toContain('No vector: encoder failed');
    target.remove();
    unmount(instance!);
    expect(badge('deferred', true)!.textContent).toBe('No vector');
  });

  it('names a deferred item', () => {
    expect(badge('deferred')!.textContent).toBe('No vector yet');
  });

  it('names an item the policy did not select, in a neutral tone', () => {
    const el = badge('not_selected')!;
    expect(el.textContent).toBe('Not embedded');
    expect(el.className).not.toContain('warn');
  });

  it('renders nothing for embedded or an unrecorded state', () => {
    expect(badge('embedded')).toBeNull();
    target.remove();
    expect(badge(null)).toBeNull();
  });
});

describe('EmbeddingRows', () => {
  function rows(embedding_states: Record<string, EmbeddingState>) {
    target = document.createElement('dl');
    document.body.appendChild(target);
    instance = mount(EmbeddingRows, {
      target,
      props: { crop: mapRawCrop(makeItem({ embedding_states })) },
    });
    flushSync();
    const row = target.querySelector('[data-testid="embedding-row"]')!;
    return [...row.querySelectorAll('[data-testid="embedding-row-space"]')].map((e) =>
      e.textContent!.replace(/\s+/g, ' ').trim(),
    );
  }

  it('reads one line per space from the served embedding_states', () => {
    expect(rows({ primary: 'embedded' })).toEqual(['primary embedded']);
    target.remove();
    unmount(instance!);
    expect(rows({ primary: 'failed', optional_space: 'embedded' })).toEqual([
      'primary No vector: encoder failed',
      'optional_space embedded',
    ]);
  });

  it('says no state is served when the item has no key in any space', () => {
    rows({});
    expect(
      target.querySelector('[data-testid="embedding-row"]')!.textContent!.trim(),
    ).toBe('no state served');
  });
});
