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
  it('GETs {API_PREFIX}/regions/vocabulary and returns detectors/region_sources/chain_actors/text_choices/text_rules', async () => {
    const payload = {
      detectors: [
        { id: 'lpr_nanov11_640', label: 'LPR', role: 'detector', filterable: true },
      ],
      region_sources: [
        { id: 'lpr_frozen_test_sample', label: 'LPR frozen test', role: 'human' },
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
    };
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(payload));
    vi.stubGlobal('fetch', fetchMock);

    const res = await getRegionVocabulary();

    const [url] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_PREFIX}/regions/vocabulary`);
    expect(res).toEqual(payload);
  });

  it('defaults each list to empty and text_rules to null when the response omits them', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({})));

    const res = await getRegionVocabulary();

    expect(res).toEqual({
      detectors: [],
      region_sources: [],
      chain_actors: [],
      text_choices: [],
      text_rules: null,
    });
  });
});

describe('getReviewTabsVocabulary', () => {
  it('GETs {API_PREFIX}/review/tabs and returns the tabs array', async () => {
    const payload = {
      tabs: [
        { id: 'all', label: 'All crops', description: 'Every crop in the pool' },
        { id: 'regions', label: 'License plates' },
      ],
    };
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(payload));
    vi.stubGlobal('fetch', fetchMock);

    const res = await getReviewTabsVocabulary();

    const [url] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_PREFIX}/review/tabs`);
    expect(res).toEqual(payload.tabs);
  });

  it('filters out entries missing a valid id/label, and defaults to [] when tabs is absent', async () => {
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
    expect(res).toEqual([{ id: 'all', label: 'All' }]);

    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({})));
    expect(await getReviewTabsVocabulary()).toEqual([]);
  });
});
