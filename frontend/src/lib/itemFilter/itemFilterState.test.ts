import { describe, expect, it } from 'vitest';
import { ItemFilterState, withoutOpenVocab } from './itemFilterState.svelte';

function filled(): ItemFilterState {
  const s = new ItemFilterState();
  s.classNames = ['widget', 'gadget x'];
  s.excludeClassNames = ['junk'];
  s.confMin = 0.2;
  s.confMax = 0.9;
  s.minArea = 0.01;
  s.maxArea = 0.5;
  s.maxRank = 2;
  s.origin = ['sam3', 'human'];
  s.embeddingState = ['failed'];
  s.reviewStatus = ['pending'];
  s.openVocabSet = 'tags';
  s.sourcePrompt = 'blue widget';
  return s;
}

describe('ItemFilterState', () => {
  it('starts empty and omits every empty value from the query and the body', () => {
    const s = new ItemFilterState();
    expect(s.isEmpty).toBe(true);
    expect(s.toQuery()).toEqual({});
    expect(s.toBody()).toEqual({});
  });

  it('query uses the query names, body uses the schema names', () => {
    const s = filled();
    expect(s.toQuery()).toEqual({
      class_name: ['widget', 'gadget x'],
      exclude_class_name: ['junk'],
      conf_min: 0.2,
      conf_max: 0.9,
      min_area: 0.01,
      max_area: 0.5,
      max_rank: 2,
      origin: ['sam3', 'human'],
      embedding_state: ['failed'],
      review_status: ['pending'],
      open_vocab_set: 'tags',
      source_prompt: 'blue widget',
    });
    expect(s.toBody()).toEqual({
      class_names: ['widget', 'gadget x'],
      exclude_class_names: ['junk'],
      conf_min: 0.2,
      conf_max: 0.9,
      min_area: 0.01,
      max_area: 0.5,
      max_rank: 2,
      origin: ['sam3', 'human'],
      embedding_state: ['failed'],
      review_status: ['pending'],
      open_vocab_set: 'tags',
      source_prompt: 'blue widget',
    });
  });

  it('an allow predicate drops params a route does not declare', () => {
    const q = filled().toQuery((p) => p !== 'origin' && p !== 'max_rank');
    expect(q).not.toHaveProperty('origin');
    expect(q).not.toHaveProperty('max_rank');
    expect(q.class_name).toEqual(['widget', 'gadget x']);
    const noVocab = filled().toQuery(withoutOpenVocab);
    expect(noVocab).not.toHaveProperty('open_vocab_set');
    expect(noVocab).not.toHaveProperty('source_prompt');
  });

  it('round-trips through URL params, repeats included', () => {
    const params = new URLSearchParams('crop_id=abc');
    filled().toUrl(params);
    expect(params.getAll('class_name')).toEqual(['widget', 'gadget x']);
    expect(params.get('open_vocab_set')).toBe('tags');
    expect(params.get('crop_id')).toBe('abc');
    const back = new ItemFilterState();
    back.fromUrl(params);
    expect(back.toQuery()).toEqual(filled().toQuery());
  });

  it('toUrl removes keys that are now empty', () => {
    const params = new URLSearchParams('class_name=a&origin=sam3&tab=all');
    new ItemFilterState().toUrl(params);
    expect(params.toString()).toBe('tab=all');
  });

  it('fromUrl drops an origin the contract does not know', () => {
    const s = new ItemFilterState();
    s.fromUrl(new URLSearchParams('origin=sam3&origin=bogus'));
    expect(s.origin).toEqual(['sam3']);
  });

  it('chips name class names verbatim and humanize enums; clear() removes one', () => {
    const s = filled();
    const chips = s.chips();
    const labels = chips.map((c) => c.label);
    expect(labels).toContain('widget');
    expect(labels).toContain('Origin: Sam3');
    expect(labels).toContain('Open-vocabulary set: tags');
    chips.find((c) => c.label === 'widget')!.clear();
    expect(s.classNames).toEqual(['gadget x']);
    chips.find((c) => c.param === 'open_vocab_set')!.clear();
    expect(s.openVocabSet).toBeNull();
  });

  it('clear() empties everything', () => {
    const s = filled();
    s.clear();
    expect(s.isEmpty).toBe(true);
  });

  it('valueOf / setValue address each field by its query param name', () => {
    const s = new ItemFilterState();
    s.setValue('class_name', ['a', 'b']);
    s.setValue('conf_min', '0.4');
    s.setValue('max_rank', '3');
    s.setValue('origin', ['human', 'bogus']);
    s.setValue('open_vocab_set', 'tags');
    expect(s.valueOf('class_name')).toEqual(['a', 'b']);
    expect(s.valueOf('conf_min')).toBe('0.4');
    expect(s.valueOf('origin')).toEqual(['human']);
    expect(s.toQuery()).toMatchObject({ conf_min: 0.4, max_rank: 3 });
    s.setValue('conf_min', '');
    s.setValue('open_vocab_set', '');
    expect(s.confMin).toBeNull();
    expect(s.openVocabSet).toBeNull();
    expect(s.valueOf('nope')).toBeUndefined();
  });
});
