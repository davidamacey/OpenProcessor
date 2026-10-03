/**
 * `/settings/ingest-policy` mounted against stubbed routes: the served
 * policy and detector labels load; an edit schedules one preview of the
 * draft; nothing is saved until the confirm; a 409 offers Reload / Keep my
 * edits; a served `unknown_names` list is shown as a warning.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import IngestPolicyPage from './+page.svelte';
import { servedIngestConfig } from '$lib/test/fixtures/ingestConfig';

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

const POLICY = {
  detect: { class_resolution: 'proposal', classes: null, exclude_classes: [] },
  embedding: { mode: 'all', classes: [] },
  detector: null,
  revision: 4,
};
const PREVIEW = {
  total_items: 100,
  scanned: 80,
  truncated: true,
  would_embed: 40,
  would_not_embed: 60,
  estimated_vector_mb: 1.5,
  by_class: [{ name: 'widget', would_embed: 40, would_not_embed: 60 }],
};

let requests: { method: string; url: string; body: unknown }[];
let putResponse: () => Response;
let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | null = null;

const q = (id: string) => document.querySelector<HTMLElement>(`[data-testid="${id}"]`);

beforeEach(() => {
  requests = [];
  putResponse = () => json({ ...POLICY, revision: 5, unknown_names: [] });
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init: RequestInit = {}) => {
      const u = String(url);
      const method = init.method ?? 'GET';
      requests.push({
        method,
        url: u,
        body: init.body ? JSON.parse(String(init.body)) : undefined,
      });
      if (u.endsWith('/ingest/policy/preview')) return json(PREVIEW);
      if (u.endsWith('/ingest/policy') && method === 'PUT') return putResponse();
      if (u.endsWith('/ingest/policy')) return json(POLICY);
      if (u.endsWith('/ingest/config')) {
        return json(
          servedIngestConfig({
            detector: {
              model: 'm',
              version: '1',
              input_size: 640,
              assigns_class: false,
              confidence_floor_applies: false,
              n_labels: 1,
              labels: [{ class_id: 0, name: 'widget', slug: 'widget' }],
            },
          }),
        );
      }
      return json({}, 404);
    }),
  );
  target = document.createElement('div');
  document.body.appendChild(target);
});

afterEach(() => {
  if (instance) unmount(instance);
  instance = null;
  target.remove();
  vi.unstubAllGlobals();
});

async function mountPage() {
  instance = mount(IngestPolicyPage, { target });
  await vi.waitFor(() => {
    flushSync();
    expect(q('ingest-policy-form')).not.toBeNull();
  });
}

async function chooseMode(mode: string) {
  const sel = q('policy-mode') as HTMLSelectElement;
  sel.value = mode;
  sel.dispatchEvent(new Event('change', { bubbles: true }));
  flushSync();
}

const puts = () => requests.filter((r) => r.method === 'PUT');

describe('ingest policy page', () => {
  it('loads the served policy and previews it once', async () => {
    await mountPage();
    expect((q('policy-mode') as HTMLSelectElement).value).toBe('all');
    await vi.waitFor(() => {
      flushSync();
      expect(q('policy-preview-summary')?.textContent).toContain(
        '40 of 100 stored detections would be embedded, about 1.5 MB',
      );
    });
    expect(q('policy-preview-truncated')?.textContent).toContain(
      'Estimated from 80 detections',
    );
    expect(target.textContent).toContain('already stored');
    expect(requests.filter((r) => r.url.endsWith('/ingest/policy/preview'))).toHaveLength(
      1,
    );
  });

  it('saves only after the confirm, with expected_revision', async () => {
    await mountPage();
    await chooseMode('lazy');
    q('policy-save')!.click();
    flushSync();
    expect(puts()).toHaveLength(0);
    const confirm = [...document.querySelectorAll('button')].find(
      (b) => b.textContent?.trim() === 'Save policy' && b !== q('policy-save'),
    )!;
    confirm.click();
    await vi.waitFor(() => expect(puts()).toHaveLength(1));
    expect(puts()[0]!.body).toMatchObject({
      embedding: { mode: 'lazy' },
      expected_revision: 4,
    });
  });

  it('shows a served unknown_names warning after a save', async () => {
    putResponse = () => json({ ...POLICY, revision: 5, unknown_names: ['gizmo'] });
    await mountPage();
    await chooseMode('lazy');
    q('policy-save')!.click();
    flushSync();
    [...document.querySelectorAll('button')]
      .find((b) => b.textContent?.trim() === 'Save policy' && b !== q('policy-save'))!
      .click();
    await vi.waitFor(() => {
      flushSync();
      expect(q('policy-unknown-names')?.textContent).toContain('gizmo');
    });
  });

  it('offers Reload and Keep my edits on a 409', async () => {
    putResponse = () =>
      json(
        { detail: { error: 'revision_conflict', message: 'Policy is at revision 6.' } },
        409,
      );
    await mountPage();
    await chooseMode('lazy');
    q('policy-save')!.click();
    flushSync();
    [...document.querySelectorAll('button')]
      .find((b) => b.textContent?.trim() === 'Save policy' && b !== q('policy-save'))!
      .click();
    await vi.waitFor(() => {
      flushSync();
      expect(q('policy-conflict')).not.toBeNull();
    });
    expect(q('policy-conflict')!.textContent).toContain('Policy is at revision 6.');
    expect(q('policy-reload')).not.toBeNull();
    expect(q('policy-keep')).not.toBeNull();
  });

  it('shows the served 422 reasons verbatim', async () => {
    putResponse = () =>
      json(
        {
          detail: {
            error: 'detector_not_servable',
            message: 'Not servable.',
            reasons: ['no labels file'],
          },
        },
        422,
      );
    await mountPage();
    await chooseMode('lazy');
    q('policy-save')!.click();
    flushSync();
    [...document.querySelectorAll('button')]
      .find((b) => b.textContent?.trim() === 'Save policy' && b !== q('policy-save'))!
      .click();
    await vi.waitFor(() => {
      flushSync();
      expect(q('policy-save-error')?.textContent).toContain('no labels file');
    });
  });
});
