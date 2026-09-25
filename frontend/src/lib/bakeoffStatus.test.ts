import { describe, expect, it } from 'vitest';
import { bakeoffFailureWhere } from './bakeoffStatus';

describe('bakeoffFailureWhere', () => {
  it('names a whole failed stage', () => {
    expect(bakeoffFailureWhere({ stage: 'throughput', error: 'x' })).toBe('throughput');
  });

  it('names a failed dataset x model cell', () => {
    expect(bakeoffFailureWhere({ dataset: 'tags', model: 'yolo11s', error: 'x' })).toBe(
      'tags · yolo11s',
    );
  });

  it('names a dataset-level failure (frozen verify)', () => {
    expect(bakeoffFailureWhere({ dataset: 'tags', error: 'frozen verify failed' })).toBe(
      'tags',
    );
  });

  it('falls back to job when nothing is named', () => {
    expect(bakeoffFailureWhere({ error: 'x' })).toBe('job');
  });
});
