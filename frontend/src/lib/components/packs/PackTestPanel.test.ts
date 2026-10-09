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
import { EMPTY_METHODS, parseMethodsResponse } from '$lib/strategies';
import { strategiesStore } from '$stores/strategies.svelte';
import PackTestPanel from './PackTestPanel.svelte';

const VLM_WIRE = {
  strategies: [
    {
      id: 'local_vlm',
      axis: 'vlm',
      label: 'Local VLM',
      status: 'stable',
      endpoint_status_label: 'Ready',
      per_run_ack_required: false,
    },
    {
      id: 'cloud_vlm',
      axis: 'vlm',
      label: 'Cloud VLM',
      status: 'experimental',
      warning: 'Crops leave the deployment.',
      per_run_ack_required: true,
    },
  ],
};

function pickVlm(target: HTMLElement, id: string) {
  const select = target.querySelector<HTMLSelectElement>(
    '[data-testid="vlm-run-select"]',
  )!;
  select.value = id;
  select.dispatchEvent(new Event('change', { bubbles: true }));
  flushSync();
}

function tickAck(target: HTMLElement) {
  const box = target.querySelector<HTMLInputElement>(
    '[data-testid="vlm-run-ack-checkbox"]',
  )!;
  box.checked = true;
  box.dispatchEvent(new Event('change', { bubbles: true }));
  flushSync();
}

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
  strategiesStore.methods = EMPTY_METHODS;
  strategiesStore.loaded = false;
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
    expect(q('test-latency')?.textContent).toBe('812 ms');
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

  it('a classify test with an empty registry shows the served 422 no_class_names detail', async () => {
    const message = 'This project has no classes; pass class_names or add classes first.';
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => json({ detail: { error: 'no_class_names', message } }, 422)),
    );
    render(schemaFixture().calls);
    const ids = q('test-crop-ids') as HTMLInputElement;
    ids.value = 'c_1';
    ids.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    (q('test-run') as HTMLButtonElement).click();
    await vi.waitFor(() =>
      expect(q('test-error')?.querySelector('p')?.textContent).toBe(message),
    );
    expect(q('test-missing-ids')).toBeNull();
  });

  describe('VLM picker', () => {
    async function runWith(setup: () => void) {
      strategiesStore.methods = parseMethodsResponse(VLM_WIRE);
      strategiesStore.loaded = true;
      const fetchMock = vi.fn(async (url: string, _init?: RequestInit) =>
        String(url).endsWith('/prompt_packs/test')
          ? json(packTestResponseFixture())
          : json({ image: { image_id: 'img_1', width: 10, height: 10 }, items: [] }),
      );
      vi.stubGlobal('fetch', fetchMock);
      render(schemaFixture().calls);
      const ids = q('test-crop-ids') as HTMLInputElement;
      ids.value = 'c_123';
      ids.dispatchEvent(new Event('input', { bubbles: true }));
      setup();
      flushSync();
      (q('test-run') as HTMLButtonElement).click();
      await vi.waitFor(() => expect(q('test-result')).not.toBeNull());
      const call = fetchMock.mock.calls.find(([u]) =>
        String(u).endsWith('/prompt_packs/test'),
      )!;
      return JSON.parse(String(call[1]!.body)) as Record<string, unknown>;
    }

    it('is absent when /methods serves no vlm axis', () => {
      render(schemaFixture().calls);
      expect(q('test-vlm-picker')?.querySelector('select')).toBeNull();
    });

    it('offers "Active endpoint" first and sends no vlm field when left there', async () => {
      const body = await runWith(() => {
        const opts = [
          ...target.querySelectorAll('[data-testid="vlm-run-select"] option'),
        ];
        expect(opts.map((o) => o.textContent?.trim())).toEqual([
          'Active endpoint',
          'Local VLM · Ready',
          'Cloud VLM',
        ]);
      });
      expect(body).not.toHaveProperty('vlm_name');
      expect(body).not.toHaveProperty('acknowledge_external');
    });

    it('sends the picked endpoint by name with a null revision', async () => {
      const body = await runWith(() => pickVlm(target, 'local_vlm'));
      expect(body).toMatchObject({ vlm_name: 'local_vlm', vlm_revision: null });
      expect(body).not.toHaveProperty('acknowledge_external');
    });

    it('an external endpoint shows its served warning and sends the acknowledgement once ticked', async () => {
      const body = await runWith(() => {
        pickVlm(target, 'cloud_vlm');
        expect(q('vlm-run-ack')?.textContent).toContain('Crops leave the deployment.');
        tickAck(target);
      });
      expect(body).toMatchObject({
        vlm_name: 'cloud_vlm',
        vlm_revision: null,
        acknowledge_external: true,
      });
    });

    it('does not send an acknowledgement that was not ticked', async () => {
      const body = await runWith(() => pickVlm(target, 'cloud_vlm'));
      expect(body).toMatchObject({ vlm_name: 'cloud_vlm' });
      expect(body).not.toHaveProperty('acknowledge_external');
    });
  });
});
