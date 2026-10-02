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
  forceUnshareText,
  forceUnshareUnreadableText,
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
    owned: false,
    sharing_revision: null,
    class_mapping: null,
    ...over,
  };
}

describe('sharingRole', () => {
  it('reads the served ownership verdict, never the project slug', () => {
    expect(sharingRole(model({ owned: true, project: 'alpha' }))).toBe('owner');
    // Owned even when the legacy promote.json names no project.
    expect(sharingRole(model({ owned: true, project: null }))).toBe('owner');
    expect(sharingRole(model({ owned: false, project: 'beta' }))).toBe('foreign');
    expect(sharingRole(model({ owned: false, project: null }))).toBe('none');
  });
});

describe('canToggleSharing', () => {
  it('needs the served owner AND the served sharing revision', () => {
    expect(canToggleSharing(model({ owned: true, sharing_revision: 3 }))).toBe(true);
    expect(canToggleSharing(model({ owned: true, sharing_revision: null }))).toBe(false);
    expect(
      canToggleSharing(model({ owned: false, project: 'beta', sharing_revision: 3 })),
    ).toBe(false);
    expect(canToggleSharing(model({ owned: false, sharing_revision: 3 }))).toBe(false);
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

  it('unsharing is never described as safe, and says the server names any project using it', () => {
    const t = unshareConfirmText('alpha__det');
    expect(t).toContain('alpha__det');
    expect(t).toMatch(/active detection profile/i);
    expect(t).not.toMatch(/may be using it/i);
    expect(t).not.toMatch(/\bsafe/i);
    expect(shareConfirmText('alpha__det')).toMatch(/by name/);
  });

  it('the forced-unshare warning names the served projects', () => {
    expect(forceUnshareText(['beta', 'gamma'])).toContain('beta, gamma');
    expect(forceUnshareText([])).toMatch(/Projects using this model/);
    expect(forceUnshareUnreadableText()).toMatch(/could not check every project/);
  });
});
