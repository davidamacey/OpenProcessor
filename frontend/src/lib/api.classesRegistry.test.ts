/**
 * W4 (docs/design/logic-moves-adoption-plan-2026-09-24.md §1.7-1.8, §2 W4):
 * `GET {API_PREFIX}/classes` now serves `thresholds`, `reserved_hotkeys` and
 * a per-class `adequacy` string alongside the class list, and
 * `POST {API_PREFIX}/classes/merge?dry_run=true` reports the blast radius
 * of a merge before any write happens. getClasses()/previewClassMerge()
 * must pass all of this through verbatim.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX, getClasses, previewClassMerge } from './api';

const ok = (body: unknown) =>
  new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('getClasses', () => {
  it('maps thresholds, reserved_hotkeys and per-class adequacy through untouched', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        ok({
          classes: [
            {
              class_id: 8,
              class_name: 'widget_a',
              group: 'single',
              sample_count: 8,
              validated_count: 8,
              deprecated: false,
              hotkey_letter: 'b',
              adequacy: 'block',
              added_at: '2026-04-29',
            },
          ],
          thresholds: {
            block_below: 20,
            warn_below: 500,
            min_test_per_class: 5,
            aug_target_min: 500,
            aug_target_max: 3000,
          },
          reserved_hotkeys: ['/', 'a', 'b', 'd', 'e', 'f', 'g', 'm', 'n', 'u', 'x', 'z'],
        }),
      ),
    );

    const res = await getClasses();

    expect(res.classes[0]).toMatchObject({
      id: 8,
      name: 'widget_a',
      hotkey_letter: 'b',
      adequacy: 'block',
    });
    expect(res.thresholds).toEqual({
      block_below: 20,
      warn_below: 500,
      min_test_per_class: 5,
      aug_target_min: 500,
      aug_target_max: 3000,
    });
    expect(res.reserved_hotkeys).toContain('b');
    expect(res.reserved_hotkeys).toHaveLength(12);
  });
});

describe('previewClassMerge', () => {
  it('sends dry_run=true and returns the blast-radius counts verbatim', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      ok({
        dry_run: true,
        source_id: 1,
        target_id: 2,
        would_relabel: 22,
        would_unvalidate: 0,
        holdout_blocking: 0,
        blocked: false,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await previewClassMerge({ source_id: 1, target_id: 2 });

    expect(res).toEqual({
      dry_run: true,
      source_id: 1,
      target_id: 2,
      would_relabel: 22,
      would_unvalidate: 0,
      holdout_blocking: 0,
      blocked: false,
    });
    const [calledUrl, calledInit] = fetchMock.mock.calls[0]!;
    expect(String(calledUrl)).toBe(`${API_PREFIX}/classes/merge?dry_run=true`);
    expect(calledInit.method).toBe('POST');
  });
});
