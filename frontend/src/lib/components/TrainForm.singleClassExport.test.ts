/**
 * m23 (2026-09-24 interactive pass): with a single-class export selected
 * (`singleClassExport` prop), TrainForm still rendered the multi-class
 * `<ClassSubsetPicker>` ("Classes to train: All 84 classes · N validated
 * crops") even though `buildSpec()`/`buildCampaign()` already force
 * `include_classes: null, single_cls: true` for a single-class export
 * regardless of any picker selection — so the summary line named a
 * registry-wide count with no bearing on what would actually train.
 *
 * Real Svelte 5 mount (same pattern as TrainForm.gpuPicker.test.ts) so
 * this proves the rendered DOM, not just the source text.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import TrainForm from './TrainForm.svelte';
import { classesStore } from '$stores/classes.svelte';

let target: HTMLDivElement;
let instance: unknown;

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });
}

async function flushMicrotasks(): Promise<void> {
  await new Promise((r) => setTimeout(r, 0));
}

function render(props: Record<string, unknown>): void {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(TrainForm, {
    target,
    props: {
      datasetExportDir: '/data/export',
      profiles: [],
      presets: [],
      preflight: null,
      preflighting: false,
      starting: false,
      onPreflight: () => {},
      onStart: () => {},
      onStartCampaign: () => {},
      ...props,
    },
  } as never);
  flushSync();
}

afterEach(() => {
  if (instance) {
    unmount(instance as never);
    instance = undefined as unknown;
  }
  target?.remove();
  vi.unstubAllGlobals();
  classesStore.classes = [];
});

describe('TrainForm — class-subset summary (m23)', () => {
  it('does not render the multi-class picker/summary when singleClassExport is true', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockResolvedValue(
          jsonResponse({ options: [], allowed_ids: [], unrestricted: true }),
        ),
    );
    classesStore.classes = Array.from({ length: 84 }, (_, i) => ({
      id: i + 1,
      name: `class_${i}`,
      group: 'g',
      deprecated: false,
      validated_count: 1,
    })) as never;

    render({ singleClassExport: true });
    await flushMicrotasks();
    flushSync();

    expect(target.textContent).not.toContain('classes');
    expect(target.textContent).not.toContain('validated crops');
    expect(target.textContent).toContain('Single-class dataset');
  });

  it('still renders the interactive class-subset picker when singleClassExport is false', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockResolvedValue(
          jsonResponse({ options: [], allowed_ids: [], unrestricted: true }),
        ),
    );
    classesStore.classes = Array.from({ length: 84 }, (_, i) => ({
      id: i + 1,
      name: `class_${i}`,
      group: 'g',
      deprecated: false,
      validated_count: 1,
    })) as never;

    render({ singleClassExport: false });
    await flushMicrotasks();
    flushSync();

    expect(target.textContent).toContain('classes');
    expect(target.textContent).not.toContain('Single-class dataset');
  });
});
