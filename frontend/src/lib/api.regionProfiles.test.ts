/**
 * W4 region-profile and vocabulary wrappers (any_domain_plan.md §4.2,
 * §7.3, §7.4): every route is scoped, every body is sent as given, the
 * list always asks for templates, `for_activation` and
 * `include_other_projects` travel as query params, and OCC values are
 * passed through untouched.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  API_PREFIX,
  activateRegionProfile,
  cloneRegionProfile,
  deactivateRegionProfile,
  deleteRegionProfile,
  getActiveRegionProfile,
  getConfigVocabulary,
  getRegionProfile,
  getRegionProfileImpact,
  getRegionProfileRevision,
  getRegionProfileRevisions,
  getRegionProfileSchema,
  listRegionProfiles,
  rollbackRegionProfile,
  updateRegionProfile,
  validateRegionProfile,
} from './api';

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

function capture(body: unknown = {}, status = 200) {
  const fetchMock = vi
    .fn()
    .mockResolvedValue(
      status === 204 ? new Response(null, { status }) : json(body, status),
    );
  vi.stubGlobal('fetch', fetchMock);
  return () => {
    const [url, init] = fetchMock.mock.calls[0]! as [string, RequestInit];
    return {
      url: String(url),
      method: init.method ?? 'GET',
      body: init.body ? JSON.parse(String(init.body)) : undefined,
    };
  };
}

const P = `${API_PREFIX}/region_profiles`;
const REF = { name: 'widget_tag', revision: 2 };

describe('W4 wrappers hit the scoped routes', () => {
  it('the list always asks for the templates (the new-from-template picker)', async () => {
    const sent = capture();
    await listRegionProfiles();
    expect(sent()).toMatchObject({ url: `${P}?include_templates=true`, method: 'GET' });
  });

  it('reads: schema, active, impact, one profile, revisions, one revision', async () => {
    let sent = capture();
    await getRegionProfileSchema();
    expect(sent().url).toBe(`${P}/schema`);

    sent = capture();
    await getActiveRegionProfile();
    expect(sent().url).toBe(`${P}/active`);

    sent = capture();
    await getRegionProfileImpact();
    expect(sent().url).toBe(`${P}/active/impact`);

    sent = capture();
    await getRegionProfile('a/b');
    expect(sent().url).toBe(`${P}/a%2Fb`);

    sent = capture();
    await getRegionProfileRevisions('widget_tag');
    expect(sent().url).toBe(`${P}/widget_tag/revisions`);

    sent = capture();
    await getRegionProfileRevision('widget_tag', 2);
    expect(sent().url).toBe(`${P}/widget_tag/revisions/2`);
  });

  it('validate sends {name, body} and the for_activation flag', async () => {
    let sent = capture({ ok: true, errors: [], warnings: [], force_allowed: false });
    await validateRegionProfile({ name: null, body: { detector_model: '' } }, false);
    expect(sent()).toEqual({
      url: `${P}/validate?for_activation=false`,
      method: 'POST',
      body: { name: null, body: { detector_model: '' } },
    });

    sent = capture({ ok: true, errors: [], warnings: [], force_allowed: false });
    await validateRegionProfile({ name: 'x', body: {} }, true);
    expect(sent().url).toBe(`${P}/validate?for_activation=true`);
  });

  it('writes: clone, update, delete with the served revision', async () => {
    let sent = capture();
    await cloneRegionProfile('widget_tag', {
      new_name: 'widget_tag_v2',
      revision: null,
      source: 'template',
      description: null,
    });
    expect(sent()).toEqual({
      url: `${P}/widget_tag/clone`,
      method: 'POST',
      body: {
        new_name: 'widget_tag_v2',
        revision: null,
        source: 'template',
        description: null,
      },
    });

    sent = capture();
    await updateRegionProfile('widget_tag', {
      expected_revision: 3,
      description: 'd',
      body: { max_regions_per_item: 4 },
    });
    expect(sent()).toEqual({
      url: `${P}/widget_tag`,
      method: 'PUT',
      body: { expected_revision: 3, description: 'd', body: { max_regions_per_item: 4 } },
    });

    sent = capture(undefined, 204);
    await deleteRegionProfile('widget_tag', 3);
    expect(sent()).toMatchObject({
      url: `${P}/widget_tag?expected_revision=3`,
      method: 'DELETE',
    });
  });

  it('activation writes carry expected_active exactly as given', async () => {
    let sent = capture();
    await activateRegionProfile('widget_tag', {
      revision: 3,
      expected_active: REF,
      force: false,
    });
    expect(sent()).toEqual({
      url: `${P}/widget_tag/activate`,
      method: 'POST',
      body: { revision: 3, expected_active: REF, force: false },
    });

    sent = capture();
    await rollbackRegionProfile({ expected_active: REF });
    expect(sent()).toEqual({
      url: `${P}/active/rollback`,
      method: 'POST',
      body: { expected_active: REF },
    });

    sent = capture();
    await deactivateRegionProfile({ expected_active: REF });
    expect(sent()).toEqual({
      url: `${P}/deactivate`,
      method: 'POST',
      body: { expected_active: REF },
    });
  });

  it('the vocabulary asks for other projects only when asked to', async () => {
    let sent = capture();
    await getConfigVocabulary(false);
    expect(sent().url).toBe(`${API_PREFIX}/config/vocabulary`);

    sent = capture();
    await getConfigVocabulary(true);
    expect(sent().url).toBe(
      `${API_PREFIX}/config/vocabulary?include_other_projects=true`,
    );
  });
});
