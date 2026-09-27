/**
 * Test-on-crop: the request carries exactly the chosen source, call and
 * crop ids (plus `use_region_box` only when picked); refusals show the
 * served message and, for `pack_invalid`, the served report.
 */
import { describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api';
import { issue, testResponseFixture } from '$lib/test/fixtures/promptPacks';
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

  it('draft source sends the draft body and nothing else', async () => {
    const test = vi.fn().mockResolvedValue(testResponseFixture());
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
  });

  it('saved source sends the pack name and revision; use_region_box only when picked', async () => {
    const test = vi.fn().mockResolvedValue(testResponseFixture());
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
