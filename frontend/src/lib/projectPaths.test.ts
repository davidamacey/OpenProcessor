/**
 * The one place in-app URLs are built (owner decision: the active
 * project lives in the URL path only, `/p/<slug>/...`).
 */
import { afterEach, describe, expect, it } from 'vitest';
import { API_PREFIX, ProjectNotSelectedError } from '$lib/api';
import {
  legacyRedirectTarget,
  NAV_GROUPS,
  PRIMARY_NAV,
  breadcrumbOf,
  navGroupOf,
  projectHref,
  sectionLabel,
  similarHref,
  sectionOf,
  switchProjectHref,
} from '$lib/projectPaths';
import { projectsStore } from '$stores/projects.svelte';
import { testProject } from '$lib/test/fixtures/projects';

const DEFAULT = testProject({ slug: 'default', prefix: API_PREFIX, is_default: true });

afterEach(() => {
  projectsStore.current = DEFAULT;
});

describe('projectHref', () => {
  it('builds on the active project', () => {
    projectsStore.current = testProject({ slug: 'alpha' });
    expect(projectHref('/classes?x=1')).toBe('/p/alpha/classes?x=1');
    expect(projectHref('/workspace?view=clusters')).toBe(
      '/p/alpha/workspace?view=clusters',
    );
  });

  it('uses an explicit slug over the active one, URL-encoded', () => {
    expect(projectHref('/dashboard', 'beta')).toBe('/p/beta/dashboard');
    expect(projectHref('/dashboard', 'a b')).toBe('/p/a%20b/dashboard');
  });

  it('fails closed with no active project', () => {
    projectsStore.current = null;
    expect(() => projectHref('/classes')).toThrow(ProjectNotSelectedError);
  });
});

describe('switchProjectHref', () => {
  const url = (s: string) => new URL(s, 'http://x');

  it('keeps the section and ordinary query params', () => {
    expect(switchProjectHref(url('/p/alpha/workspace?view=detections'), 'beta')).toBe(
      '/p/beta/workspace?view=detections',
    );
  });

  it("drops ids that don't carry across projects", () => {
    expect(
      switchProjectHref(url('/p/alpha/workspace?view=review&crop_id=c1&sort=x'), 'beta'),
    ).toBe('/p/beta/workspace?view=review&sort=x');
    expect(
      switchProjectHref(url('/p/alpha/workspace?view=clusters&class=7'), 'beta'),
    ).toBe('/p/beta/workspace?view=clusters');
    expect(
      switchProjectHref(
        url(
          '/p/alpha/workspace?view=review&queue=to_label&focus=item:c1&scope=cluster:u1',
        ),
        'beta',
      ),
    ).toBe('/p/beta/workspace?view=review&queue=to_label');
  });

  it('lands on the default section outside a project', () => {
    expect(switchProjectHref(url('/projects'), 'beta')).toBe('/p/beta/dashboard');
    expect(switchProjectHref(url('/p/alpha'), 'beta')).toBe('/p/beta/dashboard');
  });
});

describe('legacyRedirectTarget', () => {
  it('sends / to the default section, query kept', () => {
    expect(legacyRedirectTarget('/', '', 'default')).toBe('/p/default/dashboard');
    expect(legacyRedirectTarget('/', '?x=1', 'd')).toBe('/p/d/dashboard?x=1');
  });

  it('moves a bare section path under the default project, query kept', () => {
    expect(legacyRedirectTarget('/classes', '?x=1&crop_id=c1', 'default')).toBe(
      '/p/default/classes?x=1&crop_id=c1',
    );
    expect(legacyRedirectTarget('/workspace', '?view=clusters', 'default')).toBe(
      '/p/default/workspace?view=clusters',
    );
  });

  it('answers null for anything that is not a project section', () => {
    expect(legacyRedirectTarget('/nope', '', 'default')).toBeNull();
    expect(legacyRedirectTarget('/p', '', 'default')).toBeNull();
  });
});

describe('sectionOf', () => {
  it('reads the section under /p/<slug>/', () => {
    expect(sectionOf('/p/alpha/workspace')).toBe('workspace');
    expect(sectionOf('/p/alpha')).toBeNull();
    expect(sectionOf('/workspace')).toBeNull();
    expect(sectionOf('/p/alpha/unknown')).toBeNull();
  });
});

describe('similarHref', () => {
  it('encodes the crop id into the workspace query', () => {
    expect(similarHref('a/b c', 'alpha')).toBe(
      '/p/alpha/workspace?view=clusters&similar_to=a%2Fb%20c',
    );
  });
});

describe('the primary navigation', () => {
  it('lists each section once, in the order the header shows them, with a label', () => {
    const sections = PRIMARY_NAV.map((n) => n.section);
    expect(new Set(sections).size).toBe(sections.length);
    expect(sections).toEqual([
      'dashboard',
      'ingest',
      'classes',
      'workspace',
      'quality',
      'export',
      'train',
      'bakeoff',
      'models',
      'settings',
    ]);
    expect(PRIMARY_NAV.every((n) => n.label.length > 0)).toBe(true);
  });

  it('gives the breadcrumb the label of the nav link, and the id for a section with no link', () => {
    expect(sectionLabel('bakeoff')).toBe('Bake-off');
    expect(sectionLabel('quality')).toBe('Data quality');
    expect(sectionLabel('images')).toBe('Images');
  });

  it('groups the links in the order of the loop, hidden sections in no menu', () => {
    expect(NAV_GROUPS.map((g) => [g.id, g.sections.map((s) => s.section)])).toEqual([
      ['home', ['dashboard']],
      ['data', ['ingest', 'classes']],
      ['work', ['workspace']],
      ['quality', ['quality']],
      ['model', ['export', 'train', 'bakeoff', 'models']],
      ['settings', ['settings']],
    ]);
    expect(navGroupOf('images')).toBe('data');
    expect(NAV_GROUPS.flatMap((g) => g.sections).map((s) => s.section)).toEqual(
      PRIMARY_NAV.map((n) => n.section),
    );
  });
});

describe('breadcrumbOf', () => {
  const trail = (path: string, name?: string) =>
    breadcrumbOf(path, 'alpha', name).map((c) => [c.label, c.href ?? null]);

  it('has no trail on a section or a tab page: the nav and the tab strip mark them', () => {
    expect(trail('/p/alpha/dashboard')).toEqual([]);
    expect(trail('/p/alpha/quality')).toEqual([]);
    expect(trail('/p/alpha/quality/audit')).toEqual([]);
    expect(trail('/p/alpha/settings')).toEqual([]);
    expect(trail('/p/alpha/settings/prompt-packs')).toEqual([]);
    expect(trail('/p/alpha/classes')).toEqual([]);
    expect(trail('/p/alpha/export')).toEqual([]);
    expect(trail('/projects')).toEqual([]);
  });

  it('shows the trail of the import job page, back to the Import tab', () => {
    expect(trail('/p/alpha/ingest/imports/imp_1', 'COCO val')).toEqual([
      ['Data', null],
      ['Ingest', '/p/alpha/ingest?tab=import'],
      ['COCO val', null],
    ]);
  });

  it('uses the served name of an object, falling back to the segment', () => {
    expect(trail('/p/alpha/images/img_1', 'street.jpg')).toEqual([
      ['Data', null],
      ['Images', '/p/alpha/images'],
      ['street.jpg', null],
    ]);
    expect(trail('/p/alpha/images/img_1')[2]).toEqual(['img_1', null]);
    expect(trail('/p/alpha/settings/prompt-packs/my%20pack')).toEqual([
      ['Settings', '/p/alpha/settings'],
      ['Prompt packs', '/p/alpha/settings/prompt-packs'],
      ['my pack', null],
    ]);
  });
});
