import { describe, expect, it } from 'vitest';
import {
  defaultTarget,
  issuePath,
  unplacedOpenVocabIssues,
  issuesForTargetField,
  openVocabFieldAsProfileField,
  rowsByScope,
} from './openVocabFields';
import { errorReport, issue, schemaFixture } from './fixtures';

describe('openVocabFieldAsProfileField', () => {
  it('copies the served row and uses the scope as the group', () => {
    const row = schemaFixture().fields.find((f) => f.field === 'min_score')!;
    expect(openVocabFieldAsProfileField(row)).toMatchObject({
      field: 'min_score',
      label: 'Minimum score',
      group: 'target',
      type: 'float',
      default: 0.5,
      min: 0,
      max: 1,
      advanced: false,
      enum: null,
      applies_when: null,
    });
  });
});

describe('rowsByScope', () => {
  it('groups in served order', () => {
    const by = rowsByScope(schemaFixture());
    expect(by.target.map((r) => r.field)).toEqual([
      'prompt',
      'class_name',
      'min_score',
      'enabled',
      'max_instances',
      'parent_classes',
    ]);
    expect(by.set.map((r) => r.field)).toContain('display_name');
    expect(by.tier3_hit_rate.map((r) => r.field)).toEqual(['enabled', 'window']);
  });
});

describe('issuePath', () => {
  it('builds the served dotted path for each scope', () => {
    expect(issuePath('set', 'display_name')).toBe('display_name');
    expect(issuePath('target', 'prompt', 2)).toBe('targets[2].prompt');
    expect(issuePath('gating', 'tier2_vlm_precheck')).toBe('gating.tier2_vlm_precheck');
    expect(issuePath('tier3_hit_rate', 'window')).toBe('gating.tier3_hit_rate.window');
  });
});

describe('issuesForTargetField', () => {
  it('returns only the issues naming that row and cell', () => {
    const report = errorReport(
      issue({ field: 'targets[0].prompt' }),
      issue({ field: 'targets[1].prompt', code: 'b' }),
    );
    expect(issuesForTargetField(report, 1, 'prompt').map((i) => i.code)).toEqual(['b']);
    expect(issuesForTargetField(report, 1, 'min_score')).toEqual([]);
  });
});

describe('defaultTarget', () => {
  it("is built from the served target rows' defaults, nothing else", () => {
    expect(defaultTarget(schemaFixture())).toEqual({
      prompt: '',
      class_name: '',
      min_score: 0.5,
      enabled: true,
      max_instances: 20,
      parent_classes: [],
    });
  });
});

describe('unplacedOpenVocabIssues', () => {
  const schema = schemaFixture();

  it('leaves out the issues a set, gating or target cell owns', () => {
    const report = errorReport(
      issue({ field: 'display_name', code: 'a' }),
      issue({ field: 'targets[3].prompt', code: 'b' }),
      issue({ field: 'gating.tier2_vlm_precheck', code: 'c' }),
      issue({ field: 'gating.tier3_hit_rate.window', code: 'd' }),
    );
    expect(unplacedOpenVocabIssues(report, schema)).toEqual([]);
  });

  it('keeps whole-body issues and paths no served row names', () => {
    const report = errorReport(
      issue({ field: null, code: 'open_vocab_no_enabled_targets' }),
      issue({ field: 'targets', code: 'open_vocab_too_many_targets' }),
      issue({ field: 'targets[0].mystery', code: 'unknown_cell' }),
    );
    expect(unplacedOpenVocabIssues(report, schema).map((i) => i.code)).toEqual([
      'open_vocab_no_enabled_targets',
      'open_vocab_too_many_targets',
      'unknown_cell',
    ]);
  });
});
