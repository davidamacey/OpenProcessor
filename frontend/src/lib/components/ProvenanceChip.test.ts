/**
 * W0 naming-sweep finding m9: ProvenanceChip's label and chip color used
 * to come from a hardcoded per-model name→label/palette table
 * (`builtinDetectors.ts`). Both are now backend-driven — the label from
 * `regionVocabularyStore` (served by `GET {API_PREFIX}/regions/vocabulary`), the
 * color from that entry's `role` via `paletteForRole`. This mounts the
 * real component and asserts on rendered text + class list, the same
 * convention as SlotGallery.unclusteredPlates.test.ts.
 */
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import ProvenanceChip from './ProvenanceChip.svelte';
import { regionVocabularyStore } from '$stores/regionVocabulary.svelte';

let target: HTMLDivElement;
let instance: unknown;

function resetStore(): void {
  regionVocabularyStore.detectors = [];
  regionVocabularyStore.regionSources = [];
  regionVocabularyStore.chainActors = [];
  regionVocabularyStore.loaded = false;
}

function renderChip(props: Record<string, unknown>) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(ProvenanceChip, { target, props } as never);
  flushSync();
  return target;
}

beforeEach(() => {
  resetStore();
});

afterEach(() => {
  if (instance) {
    unmount(instance);
    instance = undefined;
  }
  target?.remove();
});

describe('ProvenanceChip — served vocabulary label + role-driven color (m9)', () => {
  it('renders the served label and a role-appropriate palette for a detector', () => {
    regionVocabularyStore.detectors = [
      { id: 'lpr_nanov11_640', label: 'LPR', role: 'detector', filterable: true },
    ];
    const el = renderChip({ detector: 'lpr_nanov11_640' });

    expect(el.textContent).toContain('LPR');
    const span = el.querySelector('span.chip') as HTMLElement;
    expect(span.className).toContain('border-blue-500/50');
  });

  it('colors a verifier-role id (e.g. the VLM) teal', () => {
    regionVocabularyStore.chainActors = [
      { id: 'gemma-4-e4b', label: 'Gemma', role: 'verifier' },
    ];
    const el = renderChip({ detector: 'gemma-4-e4b' });

    expect(el.textContent).toContain('Gemma');
    const span = el.querySelector('span.chip') as HTMLElement;
    expect(span.className).toContain('border-teal-500/50');
  });

  it('colors a human-role id emerald', () => {
    regionVocabularyStore.chainActors = [{ id: 'human', label: 'Human', role: 'human' }];
    const el = renderChip({ detector: 'human' });

    const span = el.querySelector('span.chip') as HTMLElement;
    expect(span.className).toContain('border-emerald-500/50');
  });

  it('renders an id not in any served vocabulary verbatim with the neutral chip', () => {
    const el = renderChip({ detector: 'some_unknown_future_detector' });

    expect(el.textContent).toContain('some_unknown_future_detector');
    const span = el.querySelector('span.chip') as HTMLElement;
    expect(span.className).toContain('border-zinc-700');
  });

  it('parses `raw="<id>:<tag>"`, and mutes a miss/reject tag', () => {
    regionVocabularyStore.detectors = [
      { id: 'lpr_nanov11_640', label: 'LPR', role: 'detector', filterable: true },
    ];
    const el = renderChip({ raw: 'lpr_nanov11_640:miss' });

    expect(el.textContent).toContain('LPR');
    expect(el.textContent).toContain('miss');
    const span = el.querySelector('span.chip') as HTMLElement;
    expect(span.className).toContain('opacity-60');
  });
});

describe('ProvenanceChip — DQ-p2: long unrecognized ids stay inside the chip, not overflowing it', () => {
  it('caps the detector/label span with a max-width + truncate rather than letting it render at full nowrap width', () => {
    const el = renderChip({
      raw: 'combined_verify_reject:region_visible_elsewhere',
    });
    const chip = el.querySelector('span.chip') as HTMLElement;
    const labelSpan = chip.children[0] as HTMLElement;
    const tagSpan = chip.children[1] as HTMLElement;

    expect(labelSpan.className).toContain('truncate');
    expect(labelSpan.className).toMatch(/max-w-\[/);
    expect(tagSpan.className).toContain('truncate');
    expect(tagSpan.className).toMatch(/max-w-\[/);
  });

  it('still carries the full untruncated string in title, for hover', () => {
    const el = renderChip({ detector: 'combined_verify_reject' });
    const chip = el.querySelector('span.chip') as HTMLElement;
    expect(chip.getAttribute('title')).toBe('combined_verify_reject');
  });
});
