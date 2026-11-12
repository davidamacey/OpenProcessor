import { describe, expect, it } from 'vitest';
import { SETTINGS_SECTIONS, sectionOfPath, searchSections } from './settingsNav';

describe('settingsNav', () => {
  it('lists every section once with a unique id and path', () => {
    const ids = SETTINGS_SECTIONS.map((s) => s.id);
    expect(new Set(ids).size).toBe(ids.length);
    expect(new Set(SETTINGS_SECTIONS.map((s) => s.path)).size).toBe(ids.length);
  });

  it('covers the sections the owner named, in their groups', () => {
    const by = (g: string) =>
      SETTINGS_SECTIONS.filter((s) => s.group === g).map((s) => s.label);
    expect(by('project')).toEqual([
      'General',
      'Ingest policy',
      'Open vocabulary',
      'Embedding spaces',
      'Cluster profiles',
      'Assist-loop presets',
      'Near-duplicate profiles',
    ]);
    expect(by('labelling')).toEqual([
      'Prompt packs',
      'Region profiles',
      'Models and VLM',
      'Curation scores',
    ]);
    expect(by('deployment')).toEqual(['Serving profiles']);
    expect(by('personal')).toEqual(['Keyboard shortcuts']);
  });

  it('a sub-page belongs to its section, the longest path wins', () => {
    expect(sectionOfPath('/p/x/settings')?.id).toBe('general');
    expect(sectionOfPath('/p/x/settings/models/vlm/a')?.id).toBe('models');
    expect(sectionOfPath('/p/x/settings/presets/cluster-profiles/p1')?.id).toBe(
      'cluster-profiles',
    );
    expect(sectionOfPath('/p/x/dashboard')).toBeNull();
  });

  it('search matches every term over the label, keywords and scope', () => {
    expect(searchSections(SETTINGS_SECTIONS, 'hotkeys').map((s) => s.id)).toEqual([
      'keyboard',
    ]);
    expect(
      searchSections(SETTINGS_SECTIONS, 'deployment serving').map((s) => s.id),
    ).toEqual(['serving-profiles']);
    expect(searchSections(SETTINGS_SECTIONS, '')).toHaveLength(SETTINGS_SECTIONS.length);
    expect(searchSections(SETTINGS_SECTIONS, 'zzzz')).toEqual([]);
  });
});
