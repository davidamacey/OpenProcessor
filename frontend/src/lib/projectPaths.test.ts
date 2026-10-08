/**
 * The one place in-app URLs are built (owner decision: the active
 * project lives in the URL path only, `/p/<slug>/...`).
 */
import { afterEach, describe, expect, it } from 'vitest';
import { API_PREFIX, ProjectNotSelectedError } from '$lib/api';
import {
  legacyRedirectTarget,
  projectHref,
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
    expect(projectHref('/review?tab=all')).toBe('/p/alpha/review?tab=all');
    expect(projectHref('/clusters/12')).toBe('/p/alpha/clusters/12');
  });

  it('uses an explicit slug over the active one, URL-encoded', () => {
    expect(projectHref('/dashboard', 'beta')).toBe('/p/beta/dashboard');
    expect(projectHref('/dashboard', 'a b')).toBe('/p/a%20b/dashboard');
  });

  it('fails closed with no active project', () => {
    projectsStore.current = null;
    expect(() => projectHref('/review')).toThrow(ProjectNotSelectedError);
  });
});

describe('switchProjectHref', () => {
  const url = (s: string) => new URL(s, 'http://x');

  it('keeps the section and ordinary query params', () => {
    expect(switchProjectHref(url('/p/alpha/review?tab=regions'), 'beta')).toBe(
      '/p/beta/review?tab=regions',
    );
  });

  it("drops ids that don't carry across projects", () => {
    expect(
      switchProjectHref(url('/p/alpha/review?tab=all&crop_id=c1&preset=x'), 'beta'),
    ).toBe('/p/beta/review?tab=all&preset=x');
    expect(switchProjectHref(url('/p/alpha/clusters?class=7'), 'beta')).toBe(
      '/p/beta/clusters',
    );
    expect(switchProjectHref(url('/p/alpha/clusters/42'), 'beta')).toBe(
      '/p/beta/clusters',
    );
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
    expect(legacyRedirectTarget('/review', '?tab=regions&crop_id=c1', 'default')).toBe(
      '/p/default/review?tab=regions&crop_id=c1',
    );
    expect(legacyRedirectTarget('/clusters/12', '', 'default')).toBe(
      '/p/default/clusters/12',
    );
  });

  it('answers null for anything that is not a project section', () => {
    expect(legacyRedirectTarget('/nope', '', 'default')).toBeNull();
    expect(legacyRedirectTarget('/p', '', 'default')).toBeNull();
  });
});

describe('sectionOf', () => {
  it('reads the section under /p/<slug>/', () => {
    expect(sectionOf('/p/alpha/clusters/3')).toBe('clusters');
    expect(sectionOf('/p/alpha')).toBeNull();
    expect(sectionOf('/review')).toBeNull();
    expect(sectionOf('/p/alpha/unknown')).toBeNull();
  });
});
