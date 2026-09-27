/**
 * `$lib/modelSharing`: who may toggle sharing (the served owner, and only
 * with the served sharing revision), the served mapping counts, and copy
 * that never calls unsharing safe.
 */
import { describe, expect, it } from 'vitest';
import {
  canToggleSharing,
  mappingText,
  shareConfirmText,
  sharingRole,
  unmappedText,
  unshareConfirmText,
} from './modelSharing';
import type { ModelInfo } from './types';

function model(over: Partial<ModelInfo>): ModelInfo {
  return {
    name: 'm',
    friendly_name: 'M',
    role: 'r',
    kind: 'triton',
    model_type: 't',
    status: 'ready',
    version: null,
    inference_count: 0,
    exec_count: 0,
    inference_failed: 0,
    avg_latency_ms: null,
    last_error: null,
    endpoint: null,
    unloadable: true,
    optional: false,
    project: null,
    shared: false,
    class_mapping: null,
    ...over,
  };
}

describe('sharingRole', () => {
  it('reads the served project against the active slug', () => {
    expect(sharingRole(model({ project: 'alpha' }), 'alpha')).toBe('owner');
    expect(sharingRole(model({ project: 'beta' }), 'alpha')).toBe('foreign');
    expect(sharingRole(model({ project: null }), 'alpha')).toBe('none');
    expect(sharingRole(model({ project: 'alpha' }), null)).toBe('foreign');
  });
});

describe('canToggleSharing', () => {
  it('needs the served owner AND the served sharing revision', () => {
    expect(
      canToggleSharing(model({ project: 'alpha', sharing_revision: 3 }), 'alpha'),
    ).toBe(true);
    expect(canToggleSharing(model({ project: 'alpha' }), 'alpha')).toBe(false);
    expect(
      canToggleSharing(model({ project: 'beta', sharing_revision: 3 }), 'alpha'),
    ).toBe(false);
    expect(canToggleSharing(model({ project: null, sharing_revision: 3 }), 'alpha')).toBe(
      false,
    );
  });
});

describe('copy', () => {
  it('counts come from the served summary', () => {
    expect(mappingText({ mapped_count: 1, unmapped: [] })).toBe('1 class maps');
    expect(mappingText({ mapped_count: 7, unmapped: ['x'] })).toBe('7 classes map');
    expect(unmappedText({ mapped_count: 7, unmapped: ['x', 'y'] })).toBe(
      '2 not in this project',
    );
  });

  it('unsharing is never described as safe, and says another project may use it', () => {
    const t = unshareConfirmText('alpha__det');
    expect(t).toContain('alpha__det');
    expect(t).toMatch(/another project may be using it/i);
    expect(t).not.toMatch(/\bsafe/i);
    expect(shareConfirmText('alpha__det')).toMatch(/by name/);
  });
});
