/**
 * `/settings/open-vocab` and `/settings/open-vocab/[name]` mounted against
 * stubbed routes: absent (one line, no other open_vocab request) when the
 * list 404s; the served sets, templates and active panel; a New set that
 * posts an empty body; Delete at the served revision; the editor's targets,
 * a served issue under its cell, Save with `expected_revision`, and an
 * activation refusal that offers no override unless the server allows it.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import {
  activeFixture,
  bodyFixture,
  docFixture,
  errorReport,
  issue,
  listFixture,
  revisionsFixture,
  schemaFixture,
  summaryFixture,
} from '$lib/openVocab/fixtures';
import { openVocabAvailability } from '$lib/openVocab/openVocabAvailability.svelte';

vi.mock('$app/navigation', () => ({ goto: vi.fn() }));
vi.mock('$app/state', () => ({ page: { params: { name: 'widgets' } } }));

import ListPage from './+page.svelte';
import EditorPage from './[name]/+page.svelte';

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

class FakeEventSource {
  onopen: (() => void) | null = null;
  onerror: (() => void) | null = null;
  addEventListener(): void {}
  close(): void {}
}

type Handler = (body: unknown) => Response;
let requests: { method: string; url: string; body: unknown }[];
let routes: Record<string, Handler>;
let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | null = null;

const q = (id: string) => document.querySelector<HTMLElement>(`[data-testid="${id}"]`);
const all = (id: string) => [
  ...document.querySelectorAll<HTMLElement>(`[data-testid="${id}"]`),
];
const settle = async () => {
  await new Promise((r) => setTimeout(r, 0));
  flushSync();
};
const dialogButton = (name: string) =>
  [...document.querySelectorAll<HTMLElement>('[role="dialog"] button')].find(
    (b) => b.textContent?.trim() === name,
  );

beforeEach(() => {
  requests = [];
  routes = {
    'GET /open_vocab': () => json(listFixture()),
    'GET /open_vocab/active': () => json(activeFixture()),
    'GET /open_vocab/schema': () => json(schemaFixture()),
    'GET /open_vocab/widgets': () => json(docFixture()),
    'GET /open_vocab/widgets/revisions': () => json(revisionsFixture()),
    'POST /open_vocab/validate': () => json(errorReport()),
  };
  openVocabAvailability.reset();
  vi.stubGlobal('EventSource', FakeEventSource);
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init: RequestInit = {}) => {
      const u = String(url);
      const method = init.method ?? 'GET';
      const body = init.body ? JSON.parse(String(init.body)) : undefined;
      requests.push({ method, url: u, body });
      const path = u.replace(/^.*?(?=\/open_vocab)/, '').replace(/\?.*$/, '');
      const h = routes[`${method} ${path}`];
      return h ? h(body) : json({ detail: 'Not Found' }, 404);
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
  openVocabAvailability.reset();
});

async function mountPage(component: typeof ListPage | typeof EditorPage) {
  instance = mount(component as never, { target });
  await settle();
  await settle();
}

describe('/settings/open-vocab', () => {
  it('is one line, and the only request is the probe, when the routes are not served', async () => {
    routes['GET /open_vocab'] = () => json({ detail: 'Not Found' }, 404);
    await mountPage(ListPage);
    expect(q('open-vocab-unavailable')).not.toBeNull();
    expect(requests.every((r) => r.url.includes('/open_vocab?include_templates'))).toBe(
      true,
    );
    expect(requests).toHaveLength(1);
  });

  it('lists the served sets and templates with the active panel', async () => {
    routes['GET /open_vocab'] = () =>
      json(
        listFixture({
          sets: [summaryFixture({ active: true, active_revision: 3 })],
          active: { name: 'widgets', revision: 3 },
        }),
      );
    routes['GET /open_vocab/active'] = () =>
      json(
        activeFixture({
          active: { name: 'widgets', revision: 3 },
          source: 'stored',
        }),
      );
    await mountPage(ListPage);
    expect(q('segmenter-notice')!.getAttribute('data-state')).toBe('ready');
    expect(all('open-vocab-row')).toHaveLength(1);
    expect(q('open-vocab-active-chip')!.textContent).toContain('active r3');
    expect(all('template-row')).toHaveLength(1);
    expect(q('active-ref')!.textContent).toContain('widgets r3');
  });

  it('New set posts an empty body for the server to fill', async () => {
    routes['POST /open_vocab'] = () => json(docFixture({ name: 'fresh' }), 201);
    await mountPage(ListPage);
    q('open-vocab-new')!.click();
    flushSync();
    const input = q('open-vocab-new-name') as HTMLInputElement;
    input.value = 'fresh';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    dialogButton('Create')!.click();
    await settle();
    const post = requests.find((r) => r.method === 'POST')!;
    expect(post.body).toEqual({ name: 'fresh', body: {} });
  });

  it('Delete sends the revision the list served', async () => {
    routes['DELETE /open_vocab/widgets'] = () => new Response(null, { status: 204 });
    await mountPage(ListPage);
    [...document.querySelectorAll<HTMLElement>('button')]
      .find((b) => b.textContent?.trim() === 'Delete')!
      .click();
    flushSync();
    dialogButton('Delete')!.click();
    await settle();
    const del = requests.find((r) => r.method === 'DELETE')!;
    expect(del.url).toContain('/open_vocab/widgets?expected_revision=3');
  });
});

describe('segmenter fact', () => {
  it('is shown as served on the list and the editor, and never hides them', async () => {
    routes['GET /open_vocab'] = () =>
      json(listFixture({ segmenter: { configured: true, reachable: false } }));
    await mountPage(ListPage);
    expect(q('segmenter-notice')!.getAttribute('data-state')).toBe('unreachable');
    expect(all('open-vocab-row')).toHaveLength(1);
    unmount(instance!);
    instance = null;
    await mountPage(EditorPage);
    expect(q('segmenter-notice')!.getAttribute('data-state')).toBe('unreachable');
    expect(all('target-row')).toHaveLength(2);
  });
});

describe('/settings/open-vocab/[name]', () => {
  it('renders the served targets and set fields', async () => {
    await mountPage(EditorPage);
    expect(all('target-row')).toHaveLength(2);
    expect(document.body.textContent).toContain('Display name');
    expect(document.body.textContent).toContain('Gating');
  });

  it('puts the served validation issue under the cell its path names', async () => {
    routes['POST /open_vocab/validate'] = () =>
      json(
        errorReport(
          issue({ field: 'targets[1].prompt', message: 'Prompt is too vague.' }),
        ),
      );
    await mountPage(EditorPage);
    const input = document.querySelector<HTMLInputElement>(
      '#profile-field-targets\\[1\\]\\.prompt',
    )!;
    input.value = 'x';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    await vi.waitFor(
      () => {
        flushSync();
        expect(all('target-row')[1]!.textContent).toContain('Prompt is too vague.');
      },
      { timeout: 3000 },
    );
    expect(all('target-row')[0]!.textContent).not.toContain('Prompt is too vague.');
  });

  it('Save sends expected_revision and the edited body', async () => {
    routes['PUT /open_vocab/widgets'] = () => json(docFixture({ revision: 4 }));
    await mountPage(EditorPage);
    const input = document.querySelector<HTMLInputElement>(
      '#profile-field-targets\\[0\\]\\.prompt',
    )!;
    input.value = 'red widget';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    q('config-save')!.click();
    await settle();
    const put = requests.find((r) => r.method === 'PUT')!;
    expect((put.body as { expected_revision: number }).expected_revision).toBe(3);
    expect(
      (put.body as { body: { targets: { prompt: string }[] } }).body.targets[0]!.prompt,
    ).toBe('red widget');
    expect(bodyFixture().targets![0]!.prompt).toBe('blue widget');
  });

  it('an activation the server refuses offers no override when force is not allowed', async () => {
    routes['POST /open_vocab/widgets/activate'] = () =>
      json(
        {
          detail: {
            error: 'validation_failed',
            message: 'The segmenter is not reachable.',
            report: {
              ok: false,
              errors: [
                issue({
                  code: 'segmenter_unreachable',
                  field: null,
                  message: 'The segmenter is not reachable.',
                }),
              ],
              warnings: [],
              force_allowed: false,
            },
          },
        },
        422,
      );
    await mountPage(EditorPage);
    q('config-activate')!.click();
    flushSync();
    dialogButton('Activate')!.click();
    await settle();
    expect(q('activate-error')!.textContent).toContain('segmenter is not reachable');
    expect(q('activate-force')).toBeNull();
  });

  it('offers "Activate anyway" only when the served report allows it', async () => {
    routes['POST /open_vocab/widgets/activate'] = () =>
      json(
        {
          detail: {
            error: 'validation_failed',
            message: 'The segmenter is not reachable.',
            report: {
              ok: false,
              errors: [issue({ code: 'segmenter_unreachable', field: null })],
              warnings: [],
              force_allowed: true,
            },
          },
        },
        422,
      );
    await mountPage(EditorPage);
    q('config-activate')!.click();
    flushSync();
    dialogButton('Activate')!.click();
    await settle();
    expect(q('activate-force')).not.toBeNull();
  });
});
