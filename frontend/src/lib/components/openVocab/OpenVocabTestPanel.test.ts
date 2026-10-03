/**
 * The test panel, mounted: the crop's served image id is what goes on the
 * wire, the VLM pre-check is sent only when ticked, the served gate and
 * hits render (a dropped hit dimmed in the overlay, `agree_existing`
 * worded), and a segmenter failure is an error banner, not "no hits".
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { ApiError } from '$lib/api';
import {
  bodyFixture,
  testResponseFixture,
  vocabularyFixture,
} from '$lib/openVocab/fixtures';

const mocks = vi.hoisted(() => ({
  getCrop: vi.fn(),
  getCropContext: vi.fn(),
  testOpenVocab: vi.fn(),
}));

vi.mock('$lib/api', async (orig) => ({
  ...(await orig<typeof import('$lib/api')>()),
  getCrop: mocks.getCrop,
  getCropContext: mocks.getCropContext,
}));
vi.mock('$lib/api_openVocab', () => ({ testOpenVocab: mocks.testOpenVocab }));

import OpenVocabTestPanel from './OpenVocabTestPanel.svelte';

let target: HTMLDivElement;
let instance: Record<string, unknown> | undefined;

beforeEach(() => {
  mocks.getCrop.mockResolvedValue({ id: 'c1', image_id: 'img-9' });
  mocks.getCropContext.mockResolvedValue({
    image: { width: 800, height: 600 },
    items: [],
  });
  mocks.testOpenVocab.mockResolvedValue(testResponseFixture());
});
afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
  vi.clearAllMocks();
});

function render(over: Record<string, unknown> = {}) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(OpenVocabTestPanel, {
    target,
    props: {
      targets: bodyFixture().targets!,
      vocabulary: vocabularyFixture(),
      imageMaxSide: 1024,
      dedupIou: 0.5,
      ...over,
    },
  });
  flushSync();
}

const q = (id: string) => target.querySelector(`[data-testid="${id}"]`) as HTMLElement;
const settle = () => new Promise((r) => setTimeout(r, 0));

async function runOnCrop(): Promise<void> {
  const input = q('ov-test-crop-id') as HTMLInputElement;
  input.value = 'c1';
  input.dispatchEvent(new Event('input', { bubbles: true }));
  flushSync();
  q('ov-test-run').click();
  await vi.waitFor(() => expect(mocks.testOpenVocab).toHaveBeenCalled());
  await settle();
  flushSync();
}

describe('OpenVocabTestPanel', () => {
  it('cannot run without a crop id', () => {
    render();
    expect((q('ov-test-run') as HTMLButtonElement).disabled).toBe(true);
  });

  it("sends the crop's image id, the chosen target and the draft's set values", async () => {
    render();
    await runOnCrop();
    expect(mocks.testOpenVocab.mock.calls[0]![0]).toEqual({
      image_id: 'img-9',
      target: bodyFixture().targets![0],
      image_max_side: 1024,
      dedup_iou: 0.5,
    });
  });

  it('sends the pre-check only when ticked', async () => {
    render();
    const box = q('ov-test-precheck') as HTMLInputElement;
    box.checked = true;
    box.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    await runOnCrop();
    expect(mocks.testOpenVocab.mock.calls[0]![0].gating).toEqual({
      tier2_vlm_precheck: true,
    });
  });

  it('renders the served gate, hits and the dimmed dropped hit', async () => {
    render();
    await runOnCrop();
    expect(q('ov-test-class').textContent).toBe('widget');
    expect(q('ov-test-gate').textContent).toContain('let this run');
    const hits = [...target.querySelectorAll('[data-testid="ov-test-hit"]')];
    expect(hits).toHaveLength(2);
    expect(hits[1]!.textContent).toContain('Matches an item already there');
    expect(hits[1]!.getAttribute('data-selected')).toBe('false');
    await vi.waitFor(() => {
      const polys = target.querySelectorAll('[data-testid="overlay-extra-box"]');
      expect(polys).toHaveLength(2);
    });
    const boxes = [...target.querySelectorAll('[data-testid="overlay-extra-box"]')];
    expect(boxes.map((b) => b.getAttribute('data-dimmed'))).toEqual(['false', 'true']);
  });

  it('words a skipped gate with its tier and served reason', async () => {
    mocks.testOpenVocab.mockResolvedValue(
      testResponseFixture({
        hits: [],
        gate: { run: false, tier: 1, reason: 'no_parent_class' },
      }),
    );
    render();
    await runOnCrop();
    expect(q('ov-test-gate').textContent).toContain('Skipped by tier 1');
    expect(q('ov-test-gate').textContent).toContain('No parent class on the image');
    expect(q('ov-test-gate').textContent).not.toContain('no_parent_class');
    expect(q('ov-test-no-hits')).not.toBeNull();
  });

  it('shows a segmenter failure as an error, never as no hits', async () => {
    mocks.testOpenVocab.mockRejectedValue(
      new ApiError(502, '/x', {
        detail: { error: 'segmenter_error', message: 'Segmenter timed out.' },
      }),
    );
    render();
    await runOnCrop();
    expect(q('ov-test-error').textContent).toBe('Segmenter error: Segmenter timed out.');
    expect(q('ov-test-no-hits')).toBeNull();
  });

  it('labels a reason served alongside a pass that did run', async () => {
    mocks.testOpenVocab.mockResolvedValue(
      testResponseFixture({ gate: { run: true, tier: 2, reason: 'vlm_no' } }),
    );
    render();
    await runOnCrop();
    expect(q('ov-test-gate').textContent).toContain('The gate let this run');
    expect(q('ov-test-gate').textContent).toContain('The VLM pre-check said no');
  });

  it('prints a gate reason the vocabulary does not list verbatim', async () => {
    mocks.testOpenVocab.mockResolvedValue(
      testResponseFixture({
        hits: [],
        gate: { run: false, tier: 2, reason: 'new_reason' as never },
      }),
    );
    render();
    await runOnCrop();
    expect(q('ov-test-gate').textContent).toContain('new_reason');
  });

  it('asks for a target before anything can run', () => {
    render({ targets: [] });
    expect(q('test-no-targets')).not.toBeNull();
    expect(q('ov-test-run')).toBeNull();
  });
});
