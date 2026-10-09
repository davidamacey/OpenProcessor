/**
 * Test-on-crop: the request carries exactly the chosen source, call and
 * crop ids (plus `use_region_box` only when picked, and the VLM selection
 * only when set); refusals show the served message and, for
 * `pack_invalid`, the served report; `crop_not_found` names the ids.
 */
import { describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api';
import { packTestResponseFixture } from '$lib/test/fixtures/configTest';
import { issue } from '$lib/test/fixtures/promptPacks';
import { createPackTest, parseCropIds } from './packTestController.svelte';

const ctx = { name: 'widget_tag', revision: 2, draft: { class_system: 'draft text' } };

describe('parseCropIds', () => {
  it('splits on commas and whitespace and drops blanks', () => {
    expect(parseCropIds(' c_1, c_2\nc_3  ,, ')).toEqual(['c_1', 'c_2', 'c_3']);
    expect(parseCropIds('   ')).toEqual([]);
  });
});

describe('PackTest', () => {
  it('cannot run without a call and a crop id', () => {
    const t = createPackTest(vi.fn());
    expect(t.canRun).toBe(false);
    t.call = 'classify';
    expect(t.canRun).toBe(false);
    t.cropIdsText = 'c_1';
    expect(t.canRun).toBe(true);
  });

  it('savedOnly forces the saved source and gives back the operator choice after', () => {
    const t = createPackTest(vi.fn());
    expect(t.source).toBe('draft');
    t.setSavedOnly(true);
    expect(t.source).toBe('saved');
    t.setSavedOnly(true);
    t.setSavedOnly(false);
    expect(t.source).toBe('draft');
    t.source = 'saved';
    t.setSavedOnly(true);
    t.setSavedOnly(false);
    expect(t.source).toBe('saved');
    t.setSavedOnly(false);
    expect(t.source).toBe('saved');
  });

  it('a new run aborts the one in flight and only the new one lands', async () => {
    const signals: AbortSignal[] = [];
    const test = vi
      .fn()
      .mockImplementationOnce((_b: unknown, s: AbortSignal) => {
        signals.push(s);
        return new Promise((_res, rej) =>
          s.addEventListener('abort', () =>
            rej(Object.assign(new Error('aborted'), { name: 'AbortError' })),
          ),
        );
      })
      .mockImplementationOnce((_b: unknown, s: AbortSignal) => {
        signals.push(s);
        return Promise.resolve(packTestResponseFixture());
      });
    const t = createPackTest(test);
    t.call = 'classify';
    t.cropIdsText = 'c_1';
    const first = t.run(ctx);
    expect(t.canRun).toBe(false);
    const second = t.run(ctx);
    await Promise.all([first, second]);
    expect(signals[0]!.aborted).toBe(true);
    expect(signals[1]!.aborted).toBe(false);
    expect(t.result?.call).toBe('classify');
    expect(t.error).toBeNull();
    expect(t.running).toBe(false);
  });

  it('draft source sends the draft body and nothing else', async () => {
    const test = vi.fn().mockResolvedValue(packTestResponseFixture());
    const t = createPackTest(test);
    t.call = 'classify';
    t.cropIdsText = 'c_1 c_2';
    await t.run(ctx);
    expect(test.mock.calls[0]![0]).toEqual({
      draft: { class_system: 'draft text' },
      call: 'classify',
      crop_ids: ['c_1', 'c_2'],
    });
    expect(t.result?.raw_reply).toContain('widget');
    expect(t.running).toBe(false);
    // Not chosen, so not sent.
    expect(test.mock.calls[0]![0]).not.toHaveProperty('use_region_box');
    expect(test.mock.calls[0]![0]).not.toHaveProperty('vlm_name');
    expect(test.mock.calls[0]![0]).not.toHaveProperty('acknowledge_external');
  });

  it('merges the VLM selection into the request only when one is set', async () => {
    const test = vi.fn().mockResolvedValue(packTestResponseFixture());
    const t = createPackTest(test);
    t.call = 'classify';
    t.cropIdsText = 'c_1';
    t.vlmSelection = {
      vlm_name: 'remote_a',
      vlm_revision: null,
      acknowledge_external: true,
    };
    await t.run(ctx);
    expect(test.mock.calls[0]![0]).toEqual({
      draft: { class_system: 'draft text' },
      call: 'classify',
      crop_ids: ['c_1'],
      vlm_name: 'remote_a',
      vlm_revision: null,
      acknowledge_external: true,
    });
    t.vlmSelection = null;
    await t.run(ctx);
    expect(test.mock.calls[1]![0]).not.toHaveProperty('vlm_name');
  });

  it('crop_not_found exposes the missing ids; a later run clears them', async () => {
    const test = vi
      .fn()
      .mockRejectedValueOnce(
        new ApiError(404, '/t', {
          detail: {
            error: 'crop_not_found',
            message: 'No crop with that id.',
            crop_ids: ['c_9'],
          },
        }),
      )
      .mockResolvedValueOnce(packTestResponseFixture());
    const t = createPackTest(test);
    t.call = 'classify';
    t.cropIdsText = 'c_1 c_9';
    await t.run(ctx);
    expect(t.error).toBe('No crop with that id.');
    expect(t.missingCropIds).toEqual(['c_9']);
    await t.run(ctx);
    expect(t.missingCropIds).toEqual([]);
  });

  it('only crop_not_found fills missingCropIds', async () => {
    const test = vi.fn().mockRejectedValue(
      new ApiError(422, '/t', {
        detail: {
          error: 'too_many_crops',
          message: 'Too many.',
          limit: 4,
          crop_ids: ['c_1'],
        },
      }),
    );
    const t = createPackTest(test);
    t.call = 'classify';
    t.cropIdsText = 'c_1';
    await t.run(ctx);
    expect(t.error).toBe('Too many.');
    expect(t.missingCropIds).toEqual([]);
  });

  it('saved source sends the pack name and revision; use_region_box only when picked', async () => {
    const test = vi.fn().mockResolvedValue(packTestResponseFixture());
    const t = createPackTest(test);
    t.call = 'combined';
    t.cropIdsText = 'c_1';
    t.source = 'saved';
    t.useRegionBox = 'none';
    await t.run(ctx);
    expect(test.mock.calls[0]![0]).toEqual({
      pack_name: 'widget_tag',
      pack_revision: 2,
      call: 'combined',
      crop_ids: ['c_1'],
      use_region_box: 'none',
    });
  });

  it('pack_invalid shows the message and the served report; a plain error its detail', async () => {
    const report = { ok: false, errors: [issue()], warnings: [], force_allowed: false };
    const test = vi
      .fn()
      .mockRejectedValueOnce(
        new ApiError(422, '/t', {
          detail: { error: 'pack_invalid', message: 'The draft has errors.', report },
        }),
      )
      .mockRejectedValueOnce(
        new ApiError(409, '/t', {
          detail: { error: 'vlm_not_configured', message: 'No VLM is configured.' },
        }),
      );
    const t = createPackTest(test);
    t.call = 'classify';
    t.cropIdsText = 'c_1';
    await t.run(ctx);
    expect(t.error).toBe('The draft has errors.');
    expect(t.errorReport).toEqual(report);
    expect(t.result).toBeNull();
    await t.run(ctx);
    expect(t.error).toBe('No VLM is configured.');
    expect(t.errorReport).toBeNull();
  });
});
