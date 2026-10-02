/**
 * The test-on-crop panel, mounted: absent unless the served schema marks a
 * call testable; offers only the testable calls; sends the draft; renders
 * the served refs, prompt, raw reply, parse state, each crop's parsed
 * answer (or served skip reason) and the preview item.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { API_PREFIX } from '$lib/api';
import { packTestResponseFixture } from '$lib/test/fixtures/configTest';
import { schemaFixture } from '$lib/test/fixtures/promptPacks';
import type { PackSchemaCall } from '$lib/types_packs';
import PackTestPanel from './PackTestPanel.svelte';

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

let target: HTMLDivElement;
let instance: Record<string, unknown> | undefined;

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
  vi.unstubAllGlobals();
});

function render(calls: PackSchemaCall[], savedOnly = false) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(PackTestPanel, {
    target,
    props: {
      calls,
      name: 'widget_tag',
      revision: 2,
      savedOnly,
      draft: { class_system: 'draft' },
    },
  });
  flushSync();
}

const q = (id: string) => target.querySelector(`[data-testid="${id}"]`);

describe('PackTestPanel', () => {
  it('is absent when no served call is testable', () => {
    render(schemaFixture().calls.map((c) => ({ ...c, testable: false })));
    expect(q('pack-test-panel')).toBeNull();
  });

  it('offers only the testable calls, with served labels', () => {
    const calls = schemaFixture().calls;
    calls[1]!.testable = false;
    render(calls);
    const opts = [...(q('test-call') as HTMLSelectElement).options].map(
      (o) => o.textContent,
    );
    expect(opts).toEqual(['Classify']);
  });

  it('read-only packs only offer the saved source', () => {
    render(schemaFixture().calls, true);
    const opts = [...(q('test-source') as HTMLSelectElement).options].map((o) => o.value);
    expect(opts).toEqual(['saved']);
  });

  it('runs the draft and renders the served prompt, reply and parsed result', async () => {
    const fetchMock = vi.fn(async (url: string, _init?: RequestInit) => {
      if (String(url).endsWith('/prompt_packs/test')) {
        return json(packTestResponseFixture());
      }
      // The preview's source-image context.
      return json({ image: { image_id: 'img_1', width: 10, height: 10 }, items: [] });
    });
    vi.stubGlobal('fetch', fetchMock);
    render(schemaFixture().calls);
    const ids = q('test-crop-ids') as HTMLInputElement;
    ids.value = 'c_123';
    ids.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    (q('test-run') as HTMLButtonElement).click();
    await vi.waitFor(() => expect(q('test-result')).not.toBeNull());
    const testCall = fetchMock.mock.calls.find(([u]) =>
      String(u).endsWith('/prompt_packs/test'),
    )!;
    expect(String(testCall[0])).toBe(`${API_PREFIX}/prompt_packs/test`);
    expect(JSON.parse(String(testCall[1]!.body))).toEqual({
      draft: { class_system: 'draft' },
      call: 'classify',
      crop_ids: ['c_123'],
    });
    expect(q('test-prompt-user')?.textContent).toBe('Pick one of: widget, gadget');
    expect(q('test-raw-reply')?.textContent).toContain('"class": "widget"');
    expect(q('test-parse-status')?.textContent).toContain('parsed');
    expect(q('test-pack-ref')?.textContent).toBe('draft');
    expect(q('test-vlm-ref')?.textContent).toContain('env@abc123');
    expect(q('test-vlm-ref')?.textContent).toContain('example/vision-model');
    expect(q('test-latency')?.textContent).toBe('812.4 ms');
    expect(q('test-parsed')?.textContent).toContain('"class_name": "widget"');
    // The preview item goes through the shared overlay path.
    await vi.waitFor(() => expect(q('test-preview-item')).not.toBeNull());
  });

  it('shows a served skip reason instead of a parsed block', async () => {
    const body = packTestResponseFixture();
    body.results = [
      {
        crop_id: 'c_5',
        box_id: 'b_1',
        skipped: 'no stored box to verify',
        preview_item: null,
      },
    ];
    body.parse_ok = false;
    body.parse_error = 'reply was not JSON';
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => json(body)),
    );
    render(schemaFixture().calls);
    const ids = q('test-crop-ids') as HTMLInputElement;
    ids.value = 'c_5';
    ids.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    (q('test-run') as HTMLButtonElement).click();
    await vi.waitFor(() => expect(q('test-result')).not.toBeNull());
    expect(q('test-skipped')?.textContent).toContain('no stored box to verify');
    expect(q('test-parsed')).toBeNull();
    expect(q('test-parse-status')?.textContent).toContain('not parsed');
    expect(q('test-parse-status')?.textContent).toContain('reply was not JSON');
    expect(target.textContent).toContain('b_1');
  });

  it('crop_not_found highlights the listed ids in the input', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        json(
          {
            detail: {
              error: 'crop_not_found',
              message: 'No crop with that id.',
              crop_ids: ['c_404'],
            },
          },
          404,
        ),
      ),
    );
    render(schemaFixture().calls);
    const ids = q('test-crop-ids') as HTMLInputElement;
    ids.value = 'c_404';
    ids.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    (q('test-run') as HTMLButtonElement).click();
    await vi.waitFor(() => expect(q('test-missing-ids')).not.toBeNull());
    expect(q('test-missing-ids')?.textContent).toContain('c_404');
    expect(q('test-error')?.textContent).toContain('No crop with that id.');
    expect(ids.getAttribute('aria-invalid')).toBe('true');
  });

  it('shows a refusal verbatim', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        json(
          {
            detail: {
              error: 'test_busy',
              message: 'Another test is running. Try again.',
            },
          },
          429,
        ),
      ),
    );
    render(schemaFixture().calls);
    const ids = q('test-crop-ids') as HTMLInputElement;
    ids.value = 'c_1';
    ids.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    (q('test-run') as HTMLButtonElement).click();
    await vi.waitFor(() =>
      expect(q('test-error')?.textContent).toContain(
        'Another test is running. Try again.',
      ),
    );
  });
});
