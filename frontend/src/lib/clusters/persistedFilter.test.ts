import { describe, expect, it } from 'vitest';
import { EMPTY_CLUSTER_FILTER } from './clusterListFilter';
import { filterPersistKey, parsePersistedFilter } from './persistedFilter';

const NONE = { sort: null, filter: EMPTY_CLUSTER_FILTER };

describe('/clusters persisted filter', () => {
  it('is keyed by project slug', () => {
    expect(filterPersistKey('a')).not.toBe(filterPersistKey('b'));
    expect(filterPersistKey('a')).toContain('a');
  });

  it('keeps a valid sort and the filter fields', () => {
    const stored = {
      sort: 'size_asc',
      filter: { ...EMPTY_CLUSTER_FILTER, flag: ['unvalidated'], min_size: 3 },
    };
    expect(parsePersistedFilter(JSON.stringify(stored))).toEqual(stored);
  });

  it('drops an unknown sort (stale or hand-edited storage)', () => {
    expect(parsePersistedFilter(JSON.stringify({ sort: 'bogus' }))).toEqual(NONE);
  });

  it('drops fields of the wrong type', () => {
    const raw = JSON.stringify({
      sort: 'size_desc',
      filter: {
        flag: 'x',
        min_size: 'many',
        validation_state: 'half',
        needs_review: 'yes',
      },
    });
    expect(parsePersistedFilter(raw)).toEqual({
      sort: 'size_desc',
      filter: EMPTY_CLUSTER_FILTER,
    });
  });

  it('returns defaults for malformed or absent storage', () => {
    expect(parsePersistedFilter(null)).toEqual(NONE);
    expect(parsePersistedFilter('{nope')).toEqual(NONE);
    expect(parsePersistedFilter('"x"')).toEqual(NONE);
  });
});
