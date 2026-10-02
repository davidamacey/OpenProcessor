/**
 * Region-profile test-on-crop: one crop id, the chosen source (draft or
 * saved), the segmenter prompt only when non-empty, `verify` only when
 * ticked, the VLM selection only when set; a new run aborts the old one;
 * refusals show the served message and name a `crop_not_found` id.
 */
import { describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api';
import { regionTestResponseFixture } from '$lib/test/fixtures/configTest';
import { createProfileTest } from './profileTestController.svelte';

const ctx = { name: 'widget_tag', revision: 2, draft: { legs: 'detector' } };
const response = () => ({
  ...regionTestResponseFixture(),
  preview: {} as never,
});

describe('ProfileTest', () => {
  it('cannot run without a crop id; trims what was typed', async () => {
    const test = vi.fn().mockResolvedValue(response());
    const t = createProfileTest(test);
    expect(t.canRun).toBe(false);
    t.cropId = '   ';
    expect(t.canRun).toBe(false);
    t.cropId = '  c_1  ';
    expect(t.canRun).toBe(true);
    await t.run(ctx);
    expect(test.mock.calls[0]![0].crop_id).toBe('c_1');
  });

  it('draft source sends the draft and nothing else', async () => {
    const test = vi.fn().mockResolvedValue(response());
    const t = createProfileTest(test);
    t.cropId = 'c_1';
    await t.run(ctx);
    expect(test.mock.calls[0]![0]).toEqual({
      crop_id: 'c_1',
      draft: { legs: 'detector' },
    });
    expect(t.result?.crop_id).toBe('c_123');
    expect(t.running).toBe(false);
  });

  it('saved source sends the profile name and revision, never the draft', async () => {
    const test = vi.fn().mockResolvedValue(response());
    const t = createProfileTest(test);
    t.cropId = 'c_1';
    t.source = 'saved';
    await t.run(ctx);
    expect(test.mock.calls[0]![0]).toEqual({
      crop_id: 'c_1',
      profile_name: 'widget_tag',
      profile_revision: 2,
    });
  });

  it('sends the segmenter prompt only when non-empty, and verify only when ticked', async () => {
    const test = vi.fn().mockResolvedValue(response());
    const t = createProfileTest(test);
    t.cropId = 'c_1';
    t.segmenterPrompt = '   ';
    await t.run(ctx);
    expect(test.mock.calls[0]![0]).not.toHaveProperty('segmenter_text_prompt');
    expect(test.mock.calls[0]![0]).not.toHaveProperty('verify');
    t.segmenterPrompt = 'a price tag';
    t.verify = true;
    await t.run(ctx);
    expect(test.mock.calls[1]![0]).toMatchObject({
      segmenter_text_prompt: 'a price tag',
      verify: true,
    });
    t.verify = false;
    await t.run(ctx);
    expect(test.mock.calls[2]![0]).not.toHaveProperty('verify');
  });

  it('merges the VLM selection only when one is set', async () => {
    const test = vi.fn().mockResolvedValue(response());
    const t = createProfileTest(test);
    t.cropId = 'c_1';
    t.verify = true;
    t.vlmSelection = {
      vlm_name: 'remote_a',
      vlm_revision: null,
      acknowledge_external: true,
    };
    await t.run(ctx);
    expect(test.mock.calls[0]![0]).toMatchObject({
      vlm_name: 'remote_a',
      vlm_revision: null,
      acknowledge_external: true,
    });
    t.vlmSelection = null;
    await t.run(ctx);
    expect(test.mock.calls[1]![0]).not.toHaveProperty('vlm_name');
  });

  it('savedOnly forces the saved source and gives back the operator choice after', () => {
    const t = createProfileTest(vi.fn());
    t.setSavedOnly(true);
    expect(t.source).toBe('saved');
    t.setSavedOnly(true);
    t.setSavedOnly(false);
    expect(t.source).toBe('draft');
    t.source = 'saved';
    t.setSavedOnly(true);
    t.setSavedOnly(false);
    expect(t.source).toBe('saved');
  });

  it('a new run aborts the one in flight and only the new one lands', async () => {
    const signals: AbortSignal[] = [];
    const test = vi
      .fn()
      .mockImplementationOnce((_b: unknown, s: AbortSignal) => {
        signals.push(s);
        return new Promise((_res, rej) => {
          s.addEventListener('abort', () =>
            rej(Object.assign(new Error('aborted'), { name: 'AbortError' })),
          );
        });
      })
      .mockImplementationOnce((_b: unknown, s: AbortSignal) => {
        signals.push(s);
        return Promise.resolve(response());
      });
    const t = createProfileTest(test);
    t.cropId = 'c_1';
    const first = t.run(ctx);
    const second = t.run(ctx);
    await Promise.all([first, second]);
    expect(signals[0]!.aborted).toBe(true);
    expect(signals[1]!.aborted).toBe(false);
    expect(t.result?.crop_id).toBe('c_123');
    expect(t.error).toBeNull();
    expect(t.running).toBe(false);
  });

  it('stop() aborts the run in flight', async () => {
    let signal!: AbortSignal;
    const test = vi.fn().mockImplementation((_b: unknown, s: AbortSignal) => {
      signal = s;
      return new Promise((_res, rej) =>
        s.addEventListener('abort', () =>
          rej(Object.assign(new Error('aborted'), { name: 'AbortError' })),
        ),
      );
    });
    const t = createProfileTest(test);
    t.cropId = 'c_1';
    const run = t.run(ctx);
    t.stop();
    await run;
    expect(signal.aborted).toBe(true);
    expect(t.error).toBeNull();
  });

  it('shows the served message of a refusal; profile_invalid keeps its report', async () => {
    const report = {
      ok: false,
      errors: [
        {
          code: 'x',
          severity: 'error',
          field: 'f',
          message: 'bad',
          detail: {},
          bypassable: false,
        },
      ],
      warnings: [],
      force_allowed: false,
    };
    const test = vi
      .fn()
      .mockRejectedValueOnce(
        new ApiError(422, '/t', {
          detail: { error: 'profile_invalid', message: 'The draft has errors.', report },
        }),
      )
      .mockRejectedValueOnce(
        new ApiError(429, '/t', {
          detail: { error: 'test_busy', message: 'Another test is running.' },
        }),
      );
    const t = createProfileTest(test);
    t.cropId = 'c_1';
    await t.run(ctx);
    expect(t.error).toBe('The draft has errors.');
    expect(t.errorReport).toEqual(report);
    expect(t.result).toBeNull();
    await t.run(ctx);
    expect(t.error).toBe('Another test is running.');
    expect(t.errorReport).toBeNull();
  });

  it('crop_not_found names the missing id; a later run clears it', async () => {
    const test = vi
      .fn()
      .mockRejectedValueOnce(
        new ApiError(404, '/t', {
          detail: {
            error: 'crop_not_found',
            message: 'No such crop.',
            crop_ids: ['c_9'],
          },
        }),
      )
      .mockResolvedValueOnce(response());
    const t = createProfileTest(test);
    t.cropId = 'c_9';
    await t.run(ctx);
    expect(t.missingCropIds).toEqual(['c_9']);
    await t.run(ctx);
    expect(t.missingCropIds).toEqual([]);
  });
});
