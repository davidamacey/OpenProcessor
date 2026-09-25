/**
 * BA-7 (OpenProcessor #36, c676d2b): every failed ingest result now
 * carries a stable `error_kind`. These cover the grouping/filtering
 * surface the Failed tab (`IngestRunPanel.svelte`) reads.
 */
import { describe, expect, it } from 'vitest';
import { createIngestResults } from './ingestResults.svelte';

describe('createIngestResults — error_kind grouping', () => {
  it('errorKindCounts groups failed entries by error_kind, unknown for null/absent, sorted desc', () => {
    const results = createIngestResults();
    results.set('a', { identifier: 'a', kind: 'failed', error_kind: 'decode_failed' });
    results.set('b', { identifier: 'b', kind: 'failed', error_kind: 'decode_failed' });
    results.set('c', { identifier: 'c', kind: 'failed', error_kind: 'too_large' });
    results.set('d', { identifier: 'd', kind: 'failed', error_kind: null });
    results.set('e', { identifier: 'e', kind: 'failed' }); // no error_kind key at all
    results.set('f', { identifier: 'f', kind: 'ingested' }); // not failed — excluded

    expect(results.errorKindCounts()).toEqual([
      ['decode_failed', 2],
      ['unknown', 2],
      ['too_large', 1],
    ]);
  });

  it('countOfErrorKind counts only failed entries matching the kind', () => {
    const results = createIngestResults();
    results.set('a', { identifier: 'a', kind: 'failed', error_kind: 'too_large' });
    results.set('b', { identifier: 'b', kind: 'failed', error_kind: 'too_large' });
    results.set('c', { identifier: 'c', kind: 'not_sent', error_kind: 'too_large' });

    expect(results.countOfErrorKind('too_large')).toBe(2);
    expect(results.countOfErrorKind('unknown')).toBe(0);
  });

  it('page(kind, offset, limit, errorKind) narrows to the matching error_kind', () => {
    const results = createIngestResults();
    results.set('a', { identifier: 'a', kind: 'failed', error_kind: 'decode_failed' });
    results.set('b', { identifier: 'b', kind: 'failed', error_kind: 'too_large' });
    results.set('c', { identifier: 'c', kind: 'failed', error_kind: 'decode_failed' });

    const filtered = results.page('failed', 0, 100, 'decode_failed');
    expect(filtered.map(([id]) => id).sort()).toEqual(['a', 'c']);

    const unfiltered = results.page('failed', 0, 100);
    expect(unfiltered.length).toBe(3);
  });

  it('toCsv includes the error_kind column', () => {
    const results = createIngestResults();
    results.set('a', {
      identifier: 'a.jpg',
      kind: 'failed',
      error: 'boom',
      error_kind: 'decode_failed',
    });
    const csv = results.toCsv();
    const [header, row] = csv.split('\n');
    expect(header).toBe('identifier,status,error,error_kind,image_id,n_crops');
    expect(row).toBe('a.jpg,failed,boom,decode_failed,,');
  });
});
