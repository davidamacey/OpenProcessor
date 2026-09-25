/**
 * W0 naming-sweep finding m9 — the deployment-configured detector/
 * segmenter/verifier vocabulary, served whole by
 * `GET {API_PREFIX}/regions/vocabulary`, and every review tab's served
 * label/description, served whole by `GET {API_PREFIX}/review/tabs`. Consumed
 * by `$stores/regionVocabulary.svelte` and
 * `$stores/reviewTabsVocabulary.svelte` respectively.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { getRegionVocabulary, getReviewTabsVocabulary, API_PREFIX } from './api';

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('getRegionVocabulary', () => {
  it('GETs {API_PREFIX}/regions/vocabulary and returns detectors/region_sources/chain_actors/text_choices/text_rules/rejection_reasons', async () => {
    const payload = {
      detectors: [
        {
          id: 'tag_detector_v1',
          label: 'Tag detector',
          role: 'detector',
          filterable: true,
        },
      ],
      region_sources: [
        { id: 'tag_holdout_sample', label: 'Tag holdout sample', role: 'human' },
      ],
      chain_actors: [{ id: 'gemma-4-e4b', label: 'Gemma', role: 'verifier' }],
      // dq-region (2026-09-24): region_text_choice values + the active
      // profile's text-validity rules.
      text_choices: ['readers_agree', 'vlm_preferred', 'vlm_invalid'],
      text_rules: {
        uppercase: true,
        charset: 'alnum',
        len_min: 4,
        len_max: 8,
        format: '',
        reject_sequences: true,
        placeholders: ['ABC1234'],
        no_reading_words: ['NULL'],
        invalid_reasons: ['placeholder', 'sequence'],
      },
      // openprocessor fix #29 / 840beb8 adoption.
      rejection_reasons: [
        {
          id: 'region_visible_elsewhere',
          label: 'Verifier: the box is wrong (region is elsewhere)',
          kind: 'model_verdict',
          match: 'exact',
          label_template: null,
        },
        {
          id: 'sanity_reject:',
          label: 'Box failed the geometry check',
          kind: 'automatic',
          match: 'prefix',
          label_template: 'Box failed the geometry check ({detail})',
        },
        {
          id: 'verifier_no_verdict',
          label: 'Verifier gave no verdict — needs human review',
          kind: 'needs_human',
          match: 'exact',
          label_template: null,
        },
      ],
      region_profile: {
        name: 'widget_tag',
        display_name: 'Widget tags',
        region_class_name: 'widget_tag',
        text_reader: 'ocr',
      },
    };
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(payload));
    vi.stubGlobal('fetch', fetchMock);

    const res = await getRegionVocabulary();

    const [url] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_PREFIX}/regions/vocabulary`);
    expect(res).toEqual(payload);
  });

  it('defaults each list to empty, text_rules to null, and rejection_reasons to [] when the response omits them', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({})));

    const res = await getRegionVocabulary();

    expect(res).toEqual({
      detectors: [],
      region_sources: [],
      chain_actors: [],
      text_choices: [],
      text_rules: null,
      rejection_reasons: [],
      region_profile: null,
    });
  });
});

describe('getReviewTabsVocabulary', () => {
  it('GETs {API_PREFIX}/review/tabs and returns the tabs array, including filter_specs', async () => {
    const payload = {
      tabs: [
        {
          id: 'all',
          label: 'All crops',
          description: 'Every crop in the pool',
          filter_specs: [],
        },
        {
          id: 'regions',
          label: 'Widget tags',
          filters: ['text', 'region_status'],
          filter_defaults: { region_status: 'all' },
          filter_specs: [
            {
              param: 'region_status',
              kind: 'enum',
              label: 'Status',
              options: [
                { value: 'all', label: 'All (accepted + rejected candidates)' },
                { value: 'detected', label: 'Detected only' },
                { value: 'verify_rejected', label: 'Verifier-rejected candidates only' },
              ],
            },
          ],
        },
      ],
    };
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(payload));
    vi.stubGlobal('fetch', fetchMock);

    const res = await getReviewTabsVocabulary();

    const [url] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_PREFIX}/review/tabs`);
    expect(res).toEqual(payload.tabs);
  });

  it('filters out entries missing a valid id/label, and defaults to [] (filters/filter_defaults/filter_specs) when absent', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        jsonResponse({
          tabs: [
            { id: 'all', label: 'All' },
            { id: '', label: 'Bad id' },
            { id: 'no_label' },
          ],
        }),
      ),
    );

    const res = await getReviewTabsVocabulary();
    expect(res).toEqual([{ id: 'all', label: 'All', filter_specs: [] }]);

    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({})));
    expect(await getReviewTabsVocabulary()).toEqual([]);
  });

  it('drops a malformed filter_specs entry (missing kind/options) rather than throwing', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        jsonResponse({
          tabs: [
            {
              id: 'regions',
              label: 'Widget tags',
              filter_specs: [
                { param: 'region_status', kind: 'enum', label: 'Status', options: [] },
                { param: 'bad', label: 'Bad — no kind' },
              ],
            },
          ],
        }),
      ),
    );

    const res = await getReviewTabsVocabulary();
    expect(res).toEqual([
      {
        id: 'regions',
        label: 'Widget tags',
        filter_specs: [
          { param: 'region_status', kind: 'enum', label: 'Status', options: [] },
        ],
      },
    ]);
  });
});
