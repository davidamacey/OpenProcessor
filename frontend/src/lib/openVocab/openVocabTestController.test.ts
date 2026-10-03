/**
 * Test-on-one-image for an unsaved target: a crop id is resolved to its
 * served `image_id` (never a crop id on the wire), an upload goes as
 * base64, the VLM pre-check is sent only when ticked, a new run aborts the
 * one in flight, and a 502 `segmenter_error` stays an error.
 */
import { describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api';
import type { Crop } from '$lib/types';
import { testResponseFixture } from './fixtures';
import { createOpenVocabTest } from './openVocabTestController.svelte';

const TARGET = { prompt: 'blue widget', class_name: 'widget', min_score: 0.5 };
const CTX = { target: TARGET, image_max_side: 1024, dedup_iou: 0.5 };

function setup(over: Record<string, unknown> = {}) {
  const deps = {
    getCrop: vi.fn().mockResolvedValue({ id: 'c1', image_id: 'img-9' } as Crop),
    testOpenVocab: vi.fn().mockResolvedValue(testResponseFixture()),
    ...over,
  };
  return { t: createOpenVocabTest(deps as never), deps };
}

describe('OpenVocabTest', () => {
  it('cannot run without a crop id or an upload', () => {
    const { t } = setup();
    expect(t.canRun).toBe(false);
    t.cropId = '  c1 ';
    expect(t.canRun).toBe(true);
  });

  it("sends the crop's served image_id, never the crop id", async () => {
    const { t, deps } = setup();
    t.cropId = ' c1 ';
    await t.run(CTX);
    expect(deps.getCrop).toHaveBeenCalledWith('c1', expect.anything());
    const req = deps.testOpenVocab.mock.calls[0]![0];
    expect(req).toEqual({
      image_id: 'img-9',
      target: TARGET,
      image_max_side: 1024,
      dedup_iou: 0.5,
    });
    expect(t.result?.hits).toHaveLength(2);
  });

  it('an upload goes as image_base64 with no image_id and no crop read', async () => {
    const { t, deps } = setup();
    t.source = 'upload';
    t.upload = { name: 'a.png', base64: 'AAAA' };
    await t.run(CTX);
    expect(deps.getCrop).not.toHaveBeenCalled();
    const req = deps.testOpenVocab.mock.calls[0]![0];
    expect(req.image_base64).toBe('AAAA');
    expect('image_id' in req).toBe(false);
  });

  it('sends gating only when the pre-check is ticked', async () => {
    const { t, deps } = setup();
    t.cropId = 'c1';
    await t.run(CTX);
    expect('gating' in deps.testOpenVocab.mock.calls[0]![0]).toBe(false);
    t.precheck = true;
    await t.run(CTX);
    expect(deps.testOpenVocab.mock.calls[1]![0].gating).toEqual({
      tier2_vlm_precheck: true,
    });
  });

  it('a crop the server does not know says so and sends nothing', async () => {
    const { t, deps } = setup({
      getCrop: vi.fn().mockRejectedValue(new ApiError(404, '/x', { detail: 'nope' })),
    });
    t.cropId = 'ghost';
    await t.run(CTX);
    expect(t.error).toBe('Crop ghost was not found.');
    expect(deps.testOpenVocab).not.toHaveBeenCalled();
  });

  it('a crop with no image id is refused plainly', async () => {
    const { t, deps } = setup({
      getCrop: vi.fn().mockResolvedValue({ id: 'c1', image_id: null }),
    });
    t.cropId = 'c1';
    await t.run(CTX);
    expect(t.error).toBe('Crop c1 has no source image to test on.');
    expect(deps.testOpenVocab).not.toHaveBeenCalled();
  });

  it('keeps a 502 segmenter_error as an error, never as "no hits"', async () => {
    const { t } = setup({
      testOpenVocab: vi.fn().mockRejectedValue(
        new ApiError(502, '/x', {
          detail: { error: 'segmenter_error', message: 'Segmenter timed out.' },
        }),
      ),
    });
    t.cropId = 'c1';
    t.result = testResponseFixture();
    await t.run(CTX);
    expect(t.error).toBe('Segmenter timed out.');
    expect(t.segmenterError).toBe(true);
    expect(t.result).toBeNull();
  });

  it('shows a 422 message verbatim and is not a segmenter error', async () => {
    const { t } = setup({
      testOpenVocab: vi.fn().mockRejectedValue(
        new ApiError(422, '/x', {
          detail: { error: 'invalid_request', message: 'Bad prompt.' },
        }),
      ),
    });
    t.cropId = 'c1';
    await t.run(CTX);
    expect(t.error).toBe('Bad prompt.');
    expect(t.segmenterError).toBe(false);
  });

  it('a new run aborts the one in flight and its answer is dropped', async () => {
    let firstSignal: AbortSignal | undefined;
    const testOpenVocab = vi
      .fn()
      .mockImplementationOnce(
        (_r: unknown, signal: AbortSignal) =>
          new Promise((_res, rej) => {
            firstSignal = signal;
            signal.addEventListener('abort', () =>
              rej(Object.assign(new Error('aborted'), { name: 'AbortError' })),
            );
          }),
      )
      .mockResolvedValue(testResponseFixture({ prompt: 'second' }));
    const { t } = setup({ testOpenVocab });
    t.cropId = 'c1';
    const first = t.run(CTX);
    await vi.waitFor(() => expect(firstSignal).toBeDefined());
    await t.run(CTX);
    await first;
    expect(firstSignal?.aborted).toBe(true);
    expect(t.result?.prompt).toBe('second');
    expect(t.error).toBeNull();
  });
});
