/**
 * The region-profile test panel, mounted: sends the chosen source and
 * options; renders the served legs (a dropped candidate greyed with its
 * reason), the candidates over the source image (box and mask polygon,
 * dropped ones dimmed) and in the crop frame, the preview heading by the
 * served `preview_basis`, the verify block, the not-eligible line, and a
 * refusal verbatim.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import {
  regionTestResponseFixture,
  packTestResponseFixture,
} from '$lib/test/fixtures/configTest';
import { EMPTY_METHODS, parseMethodsResponse } from '$lib/strategies';
import { strategiesStore } from '$stores/strategies.svelte';
import ProfileTestPanel from './ProfileTestPanel.svelte';

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
let testBodies: Array<Record<string, unknown>>;

function serve(response: () => Response) {
  testBodies = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init: RequestInit = {}) => {
      const u = String(url);
      if (u.endsWith('/region_profiles/test')) {
        testBodies.push(JSON.parse(String(init.body)));
        return response();
      }
      if (u.includes('/context')) {
        return json({
          image: { image_id: 'img_1', width: 100, height: 100 },
          items: [
            {
              crop_id: 'c_123',
              image_id: 'img_1',
              bbox_norm: [0.1, 0.1, 0.5, 0.5],
            },
          ],
        });
      }
      return json({}, 404);
    }),
  );
}

function render(over: Record<string, unknown> = {}) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(ProfileTestPanel, {
    target,
    props: {
      name: 'widget_tag',
      revision: 2,
      savedOnly: false,
      draft: { legs: 'detector' },
      ...over,
    },
  });
  flushSync();
}

const q = (id: string) => target.querySelector<HTMLElement>(`[data-testid="${id}"]`);
const qa = (id: string) => [
  ...target.querySelectorAll<HTMLElement>(`[data-testid="${id}"]`),
];

function type(id: string, value: string) {
  const el = q(id) as HTMLInputElement;
  el.value = value;
  el.dispatchEvent(new Event('input', { bubbles: true }));
  flushSync();
}

async function run(cropId = 'c_123') {
  type('profile-test-crop-id', cropId);
  (q('test-run') as HTMLButtonElement).click();
  await vi.waitFor(() => expect(q('test-result') ?? q('test-error')).not.toBeNull());
}

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
  vi.unstubAllGlobals();
  strategiesStore.methods = EMPTY_METHODS;
  strategiesStore.loaded = false;
});

describe('ProfileTestPanel', () => {
  it('cannot run without a crop id', () => {
    serve(() => json(regionTestResponseFixture()));
    render();
    expect((q('test-run') as HTMLButtonElement).disabled).toBe(true);
    type('profile-test-crop-id', 'c_1');
    expect((q('test-run') as HTMLButtonElement).disabled).toBe(false);
  });

  it('read-only profiles only offer the saved source', () => {
    serve(() => json(regionTestResponseFixture()));
    render({ savedOnly: true });
    expect(
      [...(q('test-source') as HTMLSelectElement).options].map((o) => o.value),
    ).toEqual(['saved']);
  });

  it('sends the draft by default, then the saved revision, with the options', async () => {
    serve(() => json(regionTestResponseFixture()));
    render();
    await run();
    expect(testBodies[0]).toEqual({ crop_id: 'c_123', draft: { legs: 'detector' } });
    const source = q('test-source') as HTMLSelectElement;
    source.value = 'saved';
    source.dispatchEvent(new Event('change', { bubbles: true }));
    type('test-segmenter-prompt', 'a price tag');
    const verify = q('test-verify') as HTMLInputElement;
    verify.checked = true;
    verify.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    (q('test-run') as HTMLButtonElement).click();
    await vi.waitFor(() => expect(testBodies).toHaveLength(2));
    expect(testBodies[1]).toEqual({
      crop_id: 'c_123',
      profile_name: 'widget_tag',
      profile_revision: 2,
      segmenter_text_prompt: 'a price tag',
      verify: true,
    });
  });

  it('renders each leg with its served status, time and candidates; a dropped one is greyed with its reason', async () => {
    serve(() => json(regionTestResponseFixture()));
    render();
    await run();
    const legs = qa('test-leg');
    expect(legs.map((l) => l.dataset.leg)).toEqual(['detector', 'segmenter']);
    expect(legs[0]!.querySelector('[data-testid="test-leg-status"]')?.textContent).toBe(
      'Ok',
    );
    expect(legs[0]!.textContent).toContain('41 ms');
    const rows = qa('test-candidate');
    expect(rows).toHaveLength(3);
    const dropped = rows.find((r) => r.dataset.selected === 'false')!;
    expect(dropped.className).toContain('opacity-50');
    expect(dropped.textContent).toContain('Below min score');
    expect(dropped.getAttribute('title')).toContain('Below min score');
    expect(dropped.textContent).toContain('0.12');
    const kept = rows.find(
      (r) => r.dataset.selected === 'true' && r.dataset.leg === 'detector',
    )!;
    expect(kept.className).not.toContain('opacity-50');
    const seg = rows.find((r) => r.dataset.leg === 'segmenter')!;
    expect(seg.textContent).toContain('0.83');
  });

  it("shows a skipped leg's served reason and an errored leg's status", async () => {
    serve(() =>
      json(
        regionTestResponseFixture({
          legs: [
            {
              leg: 'segmenter',
              status: 'skipped',
              reason: 'segmenter disabled',
              candidates: [],
            },
            {
              leg: 'detector',
              status: 'error',
              reason: 'detector unreachable',
              candidates: [],
            },
          ],
        }),
      ),
    );
    render();
    await run();
    const [seg, det] = qa('test-leg');
    expect(seg!.querySelector('[data-testid="test-leg-reason"]')?.textContent).toBe(
      'segmenter disabled',
    );
    expect(det!.querySelector('[data-testid="test-leg-status"]')?.textContent).toBe(
      'Error',
    );
    expect(det!.querySelector('[data-testid="test-leg-status"]')?.className).toContain(
      'text-red-300',
    );
  });

  it('draws the candidates over the source image and in the crop frame, dropped ones dimmed', async () => {
    serve(() => json(regionTestResponseFixture()));
    render();
    await run();
    await vi.waitFor(() => expect(qa('overlay-extra-box').length).toBeGreaterThan(0));
    const boxes = qa('overlay-extra-box');
    expect(boxes).toHaveLength(3);
    expect(boxes.filter((b) => b.dataset.dimmed === 'true')).toHaveLength(1);
    const polys = qa('overlay-extra-polygon');
    expect(polys).toHaveLength(1);
    expect(polys[0]!.getAttribute('points')).toBe('0.2,0.2 0.6,0.2 0.4,0.5');
    expect(qa('crop-frame-box')).toHaveLength(3);
    expect(qa('crop-frame-polygon')).toHaveLength(1);
  });

  it('heads the preview by the served preview_basis', async () => {
    serve(() => json(regionTestResponseFixture({ preview_basis: 'selection_accepted' })));
    render();
    await run();
    expect(target.textContent).toContain('Selection (not verified)');
    expect(target.textContent).not.toContain('VLM verdicts');
    unmount(instance as never);
    instance = undefined;
    target.remove();
    serve(() => json(regionTestResponseFixture({ preview_basis: 'vlm_verdicts' })));
    render();
    await run();
    expect(target.textContent).toContain('VLM verdicts');
    expect(target.textContent).not.toContain('Selection (not verified)');
  });

  it('renders the verify block only when served, with refs, prompt and reply', async () => {
    const pack = packTestResponseFixture();
    serve(() =>
      json(
        regionTestResponseFixture({
          verify: {
            latency_ms: 530,
            pack: { name: 'widget_tag', revision: 3, draft: false },
            parse_ok: false,
            parse_error: 'reply was not JSON',
            prompt: { system: 'Verify the tag box.', user_text: 'Is this a tag?' },
            raw_reply: 'maybe',
            reasoning: 'thinking aloud',
            vlm: pack.vlm,
          },
        }),
      ),
    );
    render();
    await run();
    const block = q('test-verify-block')!;
    expect(block.textContent).toContain('widget_tag@3');
    expect(block.textContent).toContain('env@abc123');
    expect(block.textContent).toContain('530 ms');
    expect(block.textContent).toContain('not parsed');
    expect(block.textContent).toContain('reply was not JSON');
    expect(block.textContent).toContain('thinking aloud');
    expect(block.querySelector('[data-testid="test-prompt-user"]')?.textContent).toBe(
      'Is this a tag?',
    );
    expect(block.querySelector('[data-testid="test-raw-reply"]')?.textContent).toBe(
      'maybe',
    );
  });

  it('has no verify block when the response carries none', async () => {
    serve(() => json(regionTestResponseFixture({ verify: null })));
    render();
    await run();
    expect(q('test-verify-block')).toBeNull();
  });

  it('says so when the item is not eligible for the profile', async () => {
    serve(() => json(regionTestResponseFixture({ item_eligible: false })));
    render();
    await run();
    expect(q('test-not-eligible')?.textContent).toContain(
      'This item is not eligible for this profile',
    );
  });

  it('shows the served profile ref', async () => {
    serve(() => json(regionTestResponseFixture()));
    render();
    await run();
    expect(q('test-profile-ref')?.textContent).toBe('widget_tag@2');
  });

  it('shows a refusal verbatim, and names the missing crop id', async () => {
    serve(() =>
      json(
        {
          detail: {
            error: 'crop_not_found',
            message: 'No crop with id c_404.',
            crop_ids: ['c_404'],
          },
        },
        404,
      ),
    );
    render();
    await run('c_404');
    expect(q('test-error')?.textContent).toContain('No crop with id c_404.');
    expect(q('test-missing-ids')?.textContent).toContain('c_404');
    expect(
      (q('profile-test-crop-id') as HTMLInputElement).getAttribute('aria-invalid'),
    ).toBe('true');
    expect(q('test-result')).toBeNull();
  });

  it('shows a busy refusal verbatim without naming ids', async () => {
    serve(() =>
      json({ detail: { error: 'test_busy', message: 'Another test is running.' } }, 429),
    );
    render();
    await run();
    expect(q('test-error')?.textContent).toContain('Another test is running.');
    expect(q('test-missing-ids')).toBeNull();
  });

  describe('VLM picker', () => {
    function check(verify: boolean) {
      const el = q('test-verify') as HTMLInputElement;
      el.checked = verify;
      el.dispatchEvent(new Event('change', { bubbles: true }));
      flushSync();
    }

    it('shows only while the verify pass is on, and only when /methods serves a vlm axis', () => {
      serve(() => json(regionTestResponseFixture()));
      strategiesStore.loaded = true;
      render();
      expect(q('test-vlm-picker')).toBeNull();
      check(true);
      expect(q('test-vlm-picker')?.querySelector('select')).toBeNull();
      unmount(instance!);
      target.remove();
      strategiesStore.methods = parseMethodsResponse(VLM_WIRE);
      render();
      check(true);
      expect(q('vlm-run-select')).not.toBeNull();
      check(false);
      expect(q('test-vlm-picker')).toBeNull();
    });

    it('sends the picked endpoint and the ticked acknowledgement with verify', async () => {
      serve(() => json(regionTestResponseFixture()));
      strategiesStore.methods = parseMethodsResponse(VLM_WIRE);
      strategiesStore.loaded = true;
      render();
      check(true);
      pickVlm(target, 'cloud_vlm');
      expect(q('vlm-run-ack')?.textContent).toContain('Crops leave the deployment.');
      tickAck(target);
      await run();
      expect(testBodies[0]).toMatchObject({
        verify: true,
        vlm_name: 'cloud_vlm',
        vlm_revision: null,
        acknowledge_external: true,
      });
    });

    it('sends no vlm field when left on the active endpoint', async () => {
      serve(() => json(regionTestResponseFixture()));
      strategiesStore.methods = parseMethodsResponse(VLM_WIRE);
      strategiesStore.loaded = true;
      render();
      check(true);
      await run();
      expect(testBodies[0]).toMatchObject({ verify: true });
      expect(testBodies[0]).not.toHaveProperty('vlm_name');
    });
  });
});
