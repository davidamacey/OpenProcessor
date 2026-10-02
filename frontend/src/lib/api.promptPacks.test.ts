/**
 * W3 prompt-pack wrappers (any_domain_plan.md §7.2, §7.5): every route is
 * scoped, every body is sent as given, and the structured `ConfigErrorDetail` refusal surfaces its
 * served `message`.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  API_PREFIX,
  ApiError,
  activatePromptPack,
  clonePromptPack,
  deletePromptPack,
  getActivePromptPack,
  getPromptPack,
  getPromptPackRevision,
  getPromptPackRevisions,
  getPromptPackSchema,
  listPromptPacks,
  configErrorDetail,
  configErrorText,
  rollbackPromptPack,
  updatePromptPack,
  validatePromptPack,
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

const P = `${API_PREFIX}/prompt_packs`;
const REF = { name: 'widget_tag', revision: 1 };

describe('W3 wrappers hit the scoped routes', () => {
  it('reads: list, schema, active, one pack, revisions, one revision', async () => {
    let sent = capture();
    await listPromptPacks();
    expect(sent()).toMatchObject({ url: P, method: 'GET' });

    sent = capture();
    await getPromptPackSchema();
    expect(sent().url).toBe(`${P}/schema`);

    sent = capture();
    await getActivePromptPack();
    expect(sent().url).toBe(`${P}/active`);

    sent = capture();
    await getPromptPack('a/b');
    expect(sent().url).toBe(`${P}/a%2Fb`);

    sent = capture();
    await getPromptPackRevisions('widget_tag');
    expect(sent().url).toBe(`${P}/widget_tag/revisions`);

    sent = capture();
    await getPromptPackRevision('widget_tag', 3);
    expect(sent().url).toBe(`${P}/widget_tag/revisions/3`);
  });

  it('validate posts {name, body} as given', async () => {
    const sent = capture({ ok: true, errors: [], warnings: [], force_allowed: false });
    await validatePromptPack({ name: null, body: { class_system: 'x' } });
    expect(sent()).toEqual({
      url: `${P}/validate`,
      method: 'POST',
      body: { name: null, body: { class_system: 'x' } },
    });
  });

  it('writes: clone, update, delete, activate, rollback', async () => {
    let sent = capture();
    await clonePromptPack('widget_tag', {
      new_name: 'widget_tag_2',
      revision: null,
      source: 'template',
      description: null,
    });
    expect(sent()).toEqual({
      url: `${P}/widget_tag/clone`,
      method: 'POST',
      body: {
        new_name: 'widget_tag_2',
        revision: null,
        source: 'template',
        description: null,
      },
    });

    sent = capture();
    await updatePromptPack('widget_tag', {
      expected_revision: 2,
      description: 'd',
      body: { class_system: 'y' },
    });
    expect(sent()).toEqual({
      url: `${P}/widget_tag`,
      method: 'PUT',
      body: { expected_revision: 2, description: 'd', body: { class_system: 'y' } },
    });

    sent = capture(null, 204);
    await expect(deletePromptPack('widget_tag', 2)).resolves.toBeUndefined();
    expect(sent()).toMatchObject({
      url: `${P}/widget_tag?expected_revision=2`,
      method: 'DELETE',
    });

    sent = capture();
    await activatePromptPack('widget_tag', {
      revision: 2,
      expected_active: REF,
      force: false,
    });
    expect(sent()).toEqual({
      url: `${P}/widget_tag/activate`,
      method: 'POST',
      body: { revision: 2, expected_active: REF, force: false },
    });

    sent = capture();
    await rollbackPromptPack({ expected_active: REF });
    expect(sent()).toEqual({
      url: `${P}/active/rollback`,
      method: 'POST',
      body: { expected_active: REF },
    });
  });
});

describe('configErrorDetail / configErrorText', () => {
  const err = (status: number, body: unknown) => new ApiError(status, `${P}/x`, body);

  it('reads a structured refusal and shows its served message', () => {
    const e = err(409, {
      detail: {
        error: 'revision_conflict',
        message: 'Someone saved revision 3 first.',
        current_revision: 3,
      },
    });
    expect(configErrorDetail(e)).toMatchObject({
      error: 'revision_conflict',
      current_revision: 3,
    });
    expect(configErrorText(e)).toBe('Someone saved revision 3 first.');
  });

  it('falls back to the generic detail for a plain or pydantic error', () => {
    const plain = err(400, { detail: 'bad' });
    expect(configErrorDetail(plain)).toBeNull();
    expect(configErrorText(plain)).toBe('bad');
    const list = err(422, { detail: [{ loc: ['body', 'call'], msg: 'Field required' }] });
    expect(configErrorDetail(list)).toBeNull();
    expect(configErrorText(list)).toBe('call: Field required');
    expect(configErrorText(new Error('network down'))).toBe('network down');
  });
});
