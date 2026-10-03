/**
 * Open-vocabulary provenance rows: each row appears only when the served
 * item carries a value, the set links to its editor only while the
 * open-vocabulary gate is open, the region-stage skip reason reads as
 * served, and "Show matching items" carries the served set and prompt.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { openVocabAvailability } from '$lib/openVocab/openVocabAvailability.svelte';
import { listFixture } from '$lib/openVocab/fixtures';
import type { Crop } from '$lib/types';
import OpenVocabProvenanceRows from './OpenVocabProvenanceRows.svelte';

let target: HTMLDListElement;
let instance: Record<string, unknown> | undefined;

function serve(status: number) {
  vi.stubGlobal(
    'fetch',
    vi.fn(
      async () =>
        new Response(JSON.stringify(status === 200 ? listFixture() : { detail: 'x' }), {
          status,
          headers: { 'content-type': 'application/json' },
        }),
    ),
  );
}

async function render(crop: Partial<Crop>, probe = true) {
  target = document.createElement('dl');
  document.body.appendChild(target);
  instance = mount(OpenVocabProvenanceRows, { target, props: { crop: crop as Crop } });
  if (probe) await openVocabAvailability.init();
  flushSync();
}

const q = (id: string) => target.querySelector(`[data-testid="${id}"]`);

beforeEach(() => openVocabAvailability.reset());
afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
  vi.unstubAllGlobals();
  openVocabAvailability.reset();
});

describe('OpenVocabProvenanceRows', () => {
  it('renders nothing for an item with none of the served fields', async () => {
    serve(200);
    await render({ source_prompt: null, open_vocab_set: null, region_gate_skip: null });
    expect(target.textContent?.trim()).toBe('');
  });

  it('renders nothing for an item that omits them entirely', async () => {
    serve(200);
    await render({});
    expect(target.textContent?.trim()).toBe('');
  });

  it('shows the prompt and the set@revision, linked while the gate is open', async () => {
    serve(200);
    await render({
      source_prompt: 'blue widget',
      open_vocab_set: 'widgets',
      open_vocab_revision: 3,
    });
    expect(q('ov-prompt')!.textContent).toBe('blue widget');
    const set = q('ov-set')!;
    expect(set.textContent).toContain('widgets@3');
    const a = set.querySelector('a')!;
    expect(a.getAttribute('href')).toContain('/settings/open-vocab/widgets');
  });

  it('shows the set as plain text when the gate is closed', async () => {
    serve(404);
    await render({ open_vocab_set: 'widgets', open_vocab_revision: 3 });
    expect(q('ov-set')!.textContent).toContain('widgets@3');
    expect(q('ov-set')!.querySelector('a')).toBeNull();
  });

  it('shows a set with no revision as just its name', async () => {
    serve(200);
    await render({ open_vocab_set: 'widgets', open_vocab_revision: null });
    expect(q('ov-set')!.textContent?.trim()).toBe('widgets');
  });

  it('shows the region-stage skip reason verbatim', async () => {
    serve(200);
    await render({ region_gate_skip: 'tier1_parent_class_missing' });
    expect(q('ov-gate-skip')!.textContent).toBe('tier1_parent_class_missing');
  });

  it('links to matching items with the served set and prompt', async () => {
    serve(200);
    await render({
      source_prompt: 'blue widget',
      open_vocab_set: 'widgets',
      open_vocab_revision: 3,
    });
    const href = q('ov-matching-link')!.getAttribute('href')!;
    expect(href).toContain('/clusters?');
    const params = new URLSearchParams(href.split('?')[1]);
    expect(params.get('mode')).toBe('matching');
    expect(params.get('open_vocab_set')).toBe('widgets');
    expect(params.get('source_prompt')).toBe('blue widget');
  });

  it('offers no matching-items link for an item with neither a set nor a prompt', async () => {
    serve(200);
    await render({ region_gate_skip: 'x' });
    expect(q('ov-matching-link')).toBeNull();
  });
});
