import { describe, expect, it } from 'vitest';
import {
  buildRunRequest,
  failureWhere,
  formatMetric,
  groupEvalDatasets,
  hasOverlap,
  isBest,
  isTerminal,
  parseClassMap,
} from './view';
import {
  DS_CURRENT,
  DS_EXTERNAL,
  DS_OLDER,
  EVAL_DATASETS,
  MATRIX,
} from '$lib/test/fixtures/bakeoff';

describe('formatMetric', () => {
  it('renders a missing value as "—", never 0', () => {
    expect(formatMetric(null, 'map_50')).toBe('—');
    expect(formatMetric(undefined, 'latency_ms')).toBe('—');
    expect(formatMetric(Number.NaN, 'f1')).toBe('—');
  });
  it('renders a real 0 as 0', () => {
    expect(formatMetric(0, 'map_50')).toBe('0.0');
  });
  it('shows fractions as percent and latency/size as-is', () => {
    expect(formatMetric(0.6234, 'map_50_95')).toBe('62.3');
    expect(formatMetric(1, 'coverage')).toBe('100.0');
    expect(formatMetric(4.25, 'latency_ms')).toBe('4.3');
    expect(formatMetric(5.4, 'size_mb')).toBe('5.4');
    expect(formatMetric(3, 'rank')).toBe('3');
  });
});

describe('groupEvalDatasets', () => {
  it('puts exports first in served order, externals grouped by served group', () => {
    const extra = { ...EVAL_DATASETS[2], id: 'external:public/other', group: 'public' };
    const g = groupEvalDatasets([
      EVAL_DATASETS[2],
      extra,
      EVAL_DATASETS[0],
      EVAL_DATASETS[1],
    ]);
    expect(g.exports.map((d) => d.id)).toEqual([DS_CURRENT, DS_OLDER]);
    expect(g.external.map((x) => [x.group, x.datasets.map((d) => d.id)])).toEqual([
      ['curated', [DS_EXTERNAL]],
      ['public', ['external:public/other']],
    ]);
  });
});

describe('hasOverlap', () => {
  it('warns only when the served overlap names at least one image', () => {
    expect(hasOverlap(null)).toBe(false);
    expect(hasOverlap(undefined)).toBe(false);
    expect(hasOverlap({ n_images: 0, fraction: 0 })).toBe(false);
    expect(hasOverlap({ n_images: 1, fraction: 0.01 })).toBe(true);
  });
});

describe('buildRunRequest', () => {
  it('sends only served keys, refs in order, profile when set', () => {
    const body = buildRunRequest({
      datasetIds: [DS_CURRENT, DS_EXTERNAL],
      runIds: ['r1'],
      baselineNames: ['b1'],
      customRefs: [
        { source: 'custom', name: 'c1', backend: 'triton', triton_model: 'm' },
      ],
      profile: 'generic',
    });
    expect(body).toEqual({
      datasets: [{ id: DS_CURRENT }, { id: DS_EXTERNAL }],
      models: [
        { source: 'run', run_id: 'r1' },
        { source: 'baseline', name: 'b1' },
        { source: 'custom', name: 'c1', backend: 'triton', triton_model: 'm' },
      ],
      profile: 'generic',
    });
  });
  it('omits profile when unset so the server default applies', () => {
    const body = buildRunRequest({
      datasetIds: [DS_CURRENT],
      runIds: ['r1'],
      baselineNames: [],
      customRefs: [],
      profile: '',
    });
    expect('profile' in body).toBe(false);
    expect(Object.keys(body).sort()).toEqual(['datasets', 'models']);
  });
});

describe('parseClassMap', () => {
  it('accepts an empty field and a string map', () => {
    expect(parseClassMap('  ')).toBeUndefined();
    expect(parseClassMap('{"0": "gear"}')).toEqual({ '0': 'gear' });
  });
  it('rejects bad JSON, non-objects and non-string names', () => {
    expect(() => parseClassMap('{')).toThrow(/valid JSON/);
    expect(() => parseClassMap('[1]')).toThrow(/object/);
    expect(() => parseClassMap('{"0": 1}')).toThrow(/"0"/);
  });
});

describe('failureWhere', () => {
  it('names whatever the failure names', () => {
    expect(failureWhere({ stage: 'throughput', dataset: null, model: null })).toBe(
      'throughput',
    );
    expect(failureWhere({ stage: null, dataset: 'd', model: 'm' })).toBe('d · m');
    expect(failureWhere({ stage: null, dataset: null, model: null })).toBe('job');
  });
});

describe('isBest / isTerminal', () => {
  it('treats every served tied winner as best', () => {
    expect(isBest(MATRIX, DS_CURRENT, 'map_50_95', 'run:run-a')).toBe(true);
    expect(isBest(MATRIX, DS_CURRENT, 'map_50_95', 'run:run-b')).toBe(true);
    expect(isBest(MATRIX, DS_CURRENT, 'map_50_95', 'baseline:ref')).toBe(false);
    expect(isBest(MATRIX, DS_CURRENT, 'recall', 'run:run-a')).toBe(false);
    expect(isBest(null, DS_CURRENT, 'map_50', 'run:run-a')).toBe(false);
  });
  it('knows the terminal states', () => {
    expect(isTerminal('done')).toBe(true);
    expect(isTerminal('error')).toBe(true);
    expect(isTerminal('queued')).toBe(false);
    expect(isTerminal('running')).toBe(false);
    expect(isTerminal(null)).toBe(false);
  });
});
