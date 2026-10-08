/**
 * W10 import/lock provenance rows: each row appears only when the served
 * item carries a value; import ids link to the import page only while W10
 * is served; every value is rendered as served.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { API_PREFIX } from '$lib/api';
import { datasetsAvailability } from '$lib/datasets/datasetsAvailability.svelte';
import { formatsFixture } from '$lib/test/fixtures/datasetImport';
import type { Crop } from '$lib/types';
import ImportProvenanceRows from './ImportProvenanceRows.svelte';

let target: HTMLDListElement;
let instance: Record<string, unknown> | undefined;

function serve(status: number) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) =>
      String(url) === `${API_PREFIX}/datasets/formats`
        ? new Response(
            JSON.stringify(status === 200 ? formatsFixture() : { detail: 'x' }),
            {
              status,
              headers: { 'content-type': 'application/json' },
            },
          )
        : new Response('{}', { status: 404 }),
    ),
  );
}

async function render(crop: Partial<Crop>, probe = true) {
  target = document.createElement('dl');
  document.body.appendChild(target);
  instance = mount(ImportProvenanceRows, { target, props: { crop: crop as Crop } });
  if (probe) await datasetsAvailability.init();
  flushSync();
  return target;
}

const q = (id: string) => target.querySelector(`[data-testid="${id}"]`);

beforeEach(() => datasetsAvailability.reset());
afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
  vi.unstubAllGlobals();
  datasetsAvailability.reset();
});

describe('ImportProvenanceRows', () => {
  it('renders nothing for an item with no import facts', async () => {
    serve(200);
    await render(
      {
        label_locked: false,
        import_ids: [],
        dataset_split: null,
        imported_at: null,
        proposed_by_import: null,
        on_negative_frame: false,
        import_standalone_region: false,
        proposal_chain: [],
      },
      false,
    );
    expect(target.children).toHaveLength(0);
    // Not even the W10 probe fires for an item that names no import.
    expect(vi.mocked(fetch)).not.toHaveBeenCalled();
  });

  it('renders every served value, each only when present', async () => {
    serve(200);
    await render({
      label_locked: true,
      import_ids: ['imp_1', 'imp_2'],
      dataset_split: 'train',
      imported_at: '2026-09-27T12:00:00Z',
      proposed_by_import: 'imp_2',
      on_negative_frame: true,
      import_standalone_region: true,
      proposal_chain: ['import:imp_2', 'detector:tag_detector_v1'],
    });
    expect(q('import-label-locked')?.textContent).toBe('Locked');
    expect(q('import-split')?.textContent).toBe('train');
    expect(q('import-at')).not.toBeNull();
    expect(q('import-proposed-by')?.textContent?.trim()).toBe('imp_2');
    expect(q('import-negative-frame')).not.toBeNull();
    expect(q('import-standalone-region')).not.toBeNull();
    expect(
      [...q('import-proposal-chain')!.querySelectorAll('span')].map((s) => s.textContent),
    ).toEqual(['import:imp_2', 'detector:tag_detector_v1']);
  });

  it('shows only the rows that carry a value', async () => {
    serve(200);
    await render({ dataset_split: 'val', import_ids: [], proposal_chain: [] });
    expect(q('import-split')?.textContent).toBe('val');
    expect(q('import-label-locked')).toBeNull();
    expect(q('import-ids')).toBeNull();
    expect(q('import-at')).toBeNull();
    expect(q('import-proposed-by')).toBeNull();
    expect(q('import-negative-frame')).toBeNull();
    expect(q('import-standalone-region')).toBeNull();
    expect(q('import-proposal-chain')).toBeNull();
  });

  it('links each import id to its job page while W10 is served', async () => {
    serve(200);
    await render({ import_ids: ['imp_1', 'imp_2'] });
    const links = [...q('import-ids')!.querySelectorAll('a')];
    expect(links.map((a) => a.getAttribute('href'))).toEqual([
      '/p/default/datasets/imports/imp_1',
      '/p/default/datasets/imports/imp_2',
    ]);
    expect(links.map((a) => a.textContent)).toEqual(['imp_1', 'imp_2']);
  });

  it('shows plain text when W10 is not served', async () => {
    serve(404);
    await render({ import_ids: ['imp_1'] });
    expect(q('import-ids')?.querySelector('a')).toBeNull();
    expect(q('import-ids')?.textContent).toContain('imp_1');
  });
});
