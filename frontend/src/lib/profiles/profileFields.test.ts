/**
 * The profile form's lookups: each `choices_from` resolves to its served
 * list (the spec's CHOICE_SOURCES table), the empty choice comes first,
 * a stored value the list doesn't carry is kept and marked, `applies_when`
 * reads the saved revision's served `effective`, and rows group by the
 * served `groups[]` order.
 */
import { describe, expect, it } from 'vitest';
import {
  gatingSchemaFixture,
  profileSchemaFixture,
  vocabularyFixture,
} from '$lib/test/fixtures/regionProfiles';
import type { ProfileSchemaField } from '$lib/types_profiles';
import {
  appliesInSaved,
  choiceList,
  groupFields,
  numberFromInput,
  selectOptions,
  valueText,
} from './profileFields';

const field = (name: string): ProfileSchemaField =>
  profileSchemaFixture().fields.find((f) => f.field === name)!;

describe('choiceList', () => {
  const v = vocabularyFixture();
  it.each([
    ['detectors', ['tag_detector_v1', 'item_detector_base']],
    ['segmenters', ['segmenter_v1']],
    ['registry_classes', ['widget', 'gadget', 'gizmo']],
    ['text_reader_modes', ['none', 'ocr']],
    ['ocr_pipeline_models', ['ocr_pipeline']],
    ['ocr_rec_models', ['ocr_rec_v1']],
  ])('%s -> the served choice ids', (from, ids) => {
    expect(choiceList(v, from)!.map((c) => c.id)).toEqual(ids);
  });

  it('uses the served choice label, never a per-list key', () => {
    expect(choiceList(v, 'detectors')![0]).toEqual({
      id: 'tag_detector_v1',
      label: 'tag_detector_v1 (promoted)',
    });
  });

  it('lists outside the vocabulary, no source, or no vocabulary -> null', () => {
    expect(choiceList(v, 'vlm_catalog')).toBeNull();
    expect(choiceList(v, 'secret_refs')).toBeNull();
    expect(choiceList(v, null)).toBeNull();
    expect(choiceList(null, 'detectors')).toBeNull();
  });
});

describe('selectOptions', () => {
  const choices = [
    { id: 'a', label: 'A' },
    { id: 'b', label: 'B' },
  ];
  it('the served empty choice first, then the served choices', () => {
    expect(selectOptions(field('detector_model'), choices, 'a')).toEqual([
      { id: '', label: 'No detector leg', listed: true },
      { id: 'a', label: 'A', listed: true },
      { id: 'b', label: 'B', listed: true },
    ]);
  });

  it('keeps a stored value the list lacks, marked', () => {
    const opts = selectOptions(field('display_name'), choices, 'gone');
    expect(opts.at(-1)).toEqual({
      id: 'gone',
      label: 'gone (not in the list)',
      listed: false,
    });
    expect(opts.some((o) => o.id === '')).toBe(false);
  });

  it('does not duplicate the empty choice when it is stored', () => {
    const opts = selectOptions(field('detector_model'), choices, '');
    expect(opts.filter((o) => o.id === '')).toHaveLength(1);
  });
});

describe('appliesInSaved', () => {
  const eff = {
    reads_text: false,
    text_hint_active: true,
    legs: ['segmenter'],
    segmenter_enabled: true,
  };
  it('reads each condition off the served effective block', () => {
    expect(appliesInSaved('detector', eff)).toBe(false);
    expect(appliesInSaved('segmenter', eff)).toBe(true);
    expect(appliesInSaved('reads_text', eff)).toBe(false);
    expect(appliesInSaved('text_hint', eff)).toBe(true);
  });
  it('null for no condition, an unknown one, or no effective', () => {
    expect(appliesInSaved(null, eff)).toBeNull();
    expect(appliesInSaved('someday', eff)).toBeNull();
    expect(appliesInSaved('detector', null)).toBeNull();
  });
});

describe('the v0.4.0 gating group', () => {
  it('is a served group like any other: the four gate_hit_* rows in served order', () => {
    const g = groupFields(gatingSchemaFixture());
    expect(g.map((x) => x.id)).toEqual(['gating']);
    expect(g[0]!.label).toBe('Gating');
    expect(g[0]!.fields.map((f) => f.field)).toEqual([
      'gate_hit_rate',
      'gate_hit_window',
      'gate_hit_miss_threshold',
      'gate_hit_sample_floor',
    ]);
  });
});

describe('groupFields', () => {
  it('follows the served group order and drops empty groups', () => {
    const g = groupFields(profileSchemaFixture());
    expect(g.map((x) => x.id)).toEqual([
      'identity',
      'items',
      'detector',
      'segmenter',
      'text',
      'advanced',
    ]);
    expect(g.find((x) => x.id === 'segmenter')!.label).toBe('Segmenter');
  });

  it('drops a served group no row names', () => {
    const s = profileSchemaFixture();
    s.groups = [...s.groups, { id: 'unused', label: 'Unused' }];
    expect(groupFields(s).map((x) => x.id)).not.toContain('unused');
  });

  it('puts a row whose group is not served at the end under its id', () => {
    const s = profileSchemaFixture();
    s.fields.push({ ...s.fields[0]!, field: 'x', group: 'later' });
    s.groups = s.groups.filter((x) => x.id !== 'verify');
    const g = groupFields(s);
    expect(g.at(-1)).toMatchObject({ id: 'later', label: 'later' });
  });
});

describe('small formatters', () => {
  it('valueText', () => {
    expect(valueText(null)).toBe('—');
    expect(valueText('')).toBe('—');
    expect(valueText([])).toBe('—');
    expect(valueText([1, 2])).toBe('1, 2');
    expect(valueText(0)).toBe('0');
  });
  it('numberFromInput: empty is null', () => {
    expect(numberFromInput('')).toBeNull();
    expect(numberFromInput(' 4 ')).toBe(4);
    expect(numberFromInput('0.5')).toBe(0.5);
  });
});
