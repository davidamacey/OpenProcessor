/**
 * The open-vocabulary list binding: the served list and templates, the
 * active ref with Rollback and Turn off (`expected_active` as read, a
 * nameless ref when nothing is active), create-from-nothing, Delete with
 * the served revision, Clone, and the `open_vocab` wake-up.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api';
import type { CurationEvent } from '$lib/sse';
import { activeFixture, docFixture, listFixture, summaryFixture } from './fixtures';
import {
  OPEN_VOCAB_AXIS,
  createOpenVocabList,
  isOpenVocabConfigEvent,
  type OpenVocabListDeps,
} from './openVocabListController.svelte';

afterEach(() => vi.restoreAllMocks());

function setup(over: Record<string, unknown> = {}) {
  let emit: (e: CurationEvent) => void = () => {};
  const deps = {
    listOpenVocab: vi.fn().mockResolvedValue(listFixture()),
    getActiveOpenVocab: vi.fn().mockResolvedValue(activeFixture()),
    createOpenVocab: vi.fn().mockResolvedValue(docFixture({ name: 'fresh' })),
    cloneOpenVocab: vi.fn().mockResolvedValue(docFixture({ name: 'copy' })),
    deleteOpenVocab: vi.fn().mockResolvedValue(undefined),
    activateOpenVocab: vi.fn(),
    rollbackOpenVocab: vi.fn().mockResolvedValue(activeFixture()),
    deactivateOpenVocab: vi
      .fn()
      .mockResolvedValue(activeFixture({ active: { name: null, revision: null } })),
    subscribe: vi.fn((cb: (e: CurationEvent) => void) => {
      emit = cb;
      return { close: vi.fn() };
    }),
    ...over,
  };
  const list = createOpenVocabList(deps as Partial<OpenVocabListDeps>);
  return { list, deps, emit: (e: CurationEvent) => emit(e) };
}

describe('OpenVocabList', () => {
  it('loads the served sets, templates and active ref', async () => {
    const { list, deps } = setup();
    await list.load();
    expect(list.list?.sets.map((s) => s.name)).toEqual(['widgets']);
    expect(list.list?.templates.map((t) => t.name)).toEqual(['starter_widgets']);
    expect(list.active.active?.axis).toBe('open_vocab');
    expect(deps.listOpenVocab).toHaveBeenCalledTimes(1);
  });

  it('rolls back with expected_active = the ref as read, then re-reads', async () => {
    const active = activeFixture({
      active: { name: 'widgets', revision: 3 },
      previous: { name: 'widgets', revision: 2 },
      source: 'stored',
    });
    const { list, deps } = setup({
      getActiveOpenVocab: vi.fn().mockResolvedValue(active),
    });
    await list.load();
    expect(await list.rollback()).toBe(true);
    expect(deps.rollbackOpenVocab).toHaveBeenCalledWith({
      expected_active: { name: 'widgets', revision: 3 },
    });
    expect(deps.listOpenVocab).toHaveBeenCalledTimes(2);
  });

  it('turns off with the active ref as read and drops nothing else', async () => {
    const active = activeFixture({
      active: { name: 'widgets', revision: 3 },
      source: 'stored',
    });
    const { list, deps } = setup({
      getActiveOpenVocab: vi.fn().mockResolvedValue(active),
    });
    await list.load();
    expect(list.active.deactivatable).toBe(true);
    expect(await list.deactivate()).toBe(true);
    expect(deps.deactivateOpenVocab).toHaveBeenCalledWith({
      expected_active: { name: 'widgets', revision: 3 },
    });
  });

  it('create-from-nothing posts an empty body for the server to fill', async () => {
    const { list, deps } = setup();
    await list.load();
    const doc = await list.create('fresh');
    expect(doc?.name).toBe('fresh');
    expect(deps.createOpenVocab).toHaveBeenCalledWith({ name: 'fresh', body: {} });
  });

  it('a refused create keeps the served message and returns null', async () => {
    const { list } = setup({
      createOpenVocab: vi.fn().mockRejectedValue(
        new ApiError(409, '/x', {
          detail: { error: 'name_taken', message: 'That name is taken.' },
        }),
      ),
    });
    expect(await list.create('widgets')).toBeNull();
    expect(list.createError).toBe('That name is taken.');
  });

  it('deletes at the revision the list served', async () => {
    const { list, deps } = setup();
    await list.load();
    await list.remove(summaryFixture({ revision: 3 }));
    expect(deps.deleteOpenVocab).toHaveBeenCalledWith('widgets', 3);
  });

  it('clones a template by naming its source', async () => {
    const { list, deps } = setup();
    const doc = await list.clone(
      { name: 'starter_widgets', source: 'template' },
      'copy',
      '',
    );
    expect(doc?.name).toBe('copy');
    expect(deps.cloneOpenVocab).toHaveBeenCalledWith('starter_widgets', {
      new_name: 'copy',
      revision: null,
      source: 'template',
      description: null,
    });
  });

  it('re-reads on a config.changed event for its axis only', async () => {
    const { list, deps, emit } = setup();
    list.start();
    await Promise.resolve();
    await Promise.resolve();
    const n = deps.listOpenVocab.mock.calls.length;
    emit({ type: 'config.changed', axis: 'prompt_pack' } as never);
    expect(deps.listOpenVocab.mock.calls.length).toBe(n);
    emit({ type: 'config.changed', axis: OPEN_VOCAB_AXIS } as never);
    expect(deps.listOpenVocab.mock.calls.length).toBe(n + 1);
    list.stop();
    expect(
      isOpenVocabConfigEvent({ type: 'config.changed', axis: 'open_vocab' } as never),
    ).toBe(true);
  });
});
