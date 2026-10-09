/**
 * The pack list page's state: the served list and active pack, Rollback
 * with `expected_active` from the last read, Delete with the row's
 * revision, Clone, and the `config.changed` wake-up.
 */
import { describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api';
import type { CurationEvent } from '$lib/sse';
import {
  activeFixture,
  docFixture,
  issue,
  listFixture,
} from '$lib/test/fixtures/promptPacks';
import { createPackList, type PackListDeps } from './packListController.svelte';

function refusal(status: number, detail: Record<string, unknown>): ApiError {
  return new ApiError(status, '/x', { detail });
}

function setup(over: Partial<PackListDeps> = {}) {
  let emit: (e: CurationEvent) => void = () => {};
  const close = vi.fn();
  const deps = {
    listPromptPacks: vi.fn().mockResolvedValue(listFixture()),
    getActivePromptPack: vi.fn().mockResolvedValue(activeFixture()),
    rollbackPromptPack: vi.fn().mockResolvedValue(
      activeFixture({
        active: { name: 'generic_item_v1', revision: null },
        previous: { name: 'widget_tag', revision: 1 },
      }),
    ),
    activatePromptPack: vi.fn(),
    deletePromptPack: vi.fn().mockResolvedValue(undefined),
    clonePromptPack: vi.fn().mockResolvedValue(docFixture({ name: 'widget_tag_2' })),
    subscribe: vi.fn((cb: (e: CurationEvent) => void) => {
      emit = cb;
      return { close };
    }),
  };
  const list = createPackList({ ...deps, ...over });
  return { list, deps, emit: (e: CurationEvent) => emit(e), close };
}

describe('PackList', () => {
  it('loads the served list and the active pack', async () => {
    const { list } = setup();
    await list.load();
    expect(list.list?.packs.map((p) => p.name)).toEqual([
      'generic_item_v1',
      'widget_tag',
    ]);
    expect(list.active.active?.active).toEqual({ name: 'widget_tag', revision: 1 });
  });

  it('shows a failed list read verbatim', async () => {
    const { list } = setup({
      listPromptPacks: vi.fn().mockRejectedValue(
        refusal(503, {
          error: 'config_store_unavailable',
          message: 'Config store is down.',
        }),
      ),
    });
    await list.load();
    expect(list.loadError).toBe('Config store is down.');
  });

  it('rollback sends expected_active from the last read, adopts the answer, re-reads', async () => {
    const { list, deps } = setup();
    await list.load();
    deps.getActivePromptPack.mockImplementation(
      () => deps.rollbackPromptPack.mock.results[0]!.value,
    );
    expect(await list.rollback()).toBe(true);
    expect(deps.rollbackPromptPack).toHaveBeenCalledWith({
      expected_active: { name: 'widget_tag', revision: 1 },
    });
    expect(list.active.active?.active.name).toBe('generic_item_v1');
    expect(deps.listPromptPacks).toHaveBeenCalledTimes(2);
  });

  it('rollback: active_conflict shows the served message and re-reads the active pack', async () => {
    const { list, deps } = setup({
      rollbackPromptPack: vi.fn().mockRejectedValue(
        refusal(409, {
          error: 'active_conflict',
          message: 'The active pack changed.',
          current: { name: 'other', revision: 4 },
        }),
      ),
    });
    await list.load();
    deps.getActivePromptPack.mockResolvedValue(
      activeFixture({ active: { name: 'other', revision: 4 } }),
    );
    expect(await list.rollback()).toBe(false);
    expect(list.active.actionError).toBe('The active pack changed.');
    expect(list.active.active?.active).toEqual({ name: 'other', revision: 4 });
  });

  it('rollback: no_previous is shown and nothing is re-read', async () => {
    const { list, deps } = setup({
      rollbackPromptPack: vi
        .fn()
        .mockRejectedValue(
          refusal(409, { error: 'no_previous', message: 'Nothing to roll back to.' }),
        ),
    });
    await list.load();
    expect(await list.rollback()).toBe(false);
    expect(list.active.actionError).toBe('Nothing to roll back to.');
    expect(deps.getActivePromptPack).toHaveBeenCalledTimes(1);
  });

  it('delete sends the row revision; in_use is shown verbatim', async () => {
    const { list, deps } = setup();
    await list.load();
    const row = list.list!.packs[1]!;
    expect(await list.remove(row)).toBe(true);
    expect(deps.deletePromptPack).toHaveBeenCalledWith('widget_tag', 2);

    deps.deletePromptPack.mockRejectedValue(
      refusal(409, { error: 'in_use', message: 'widget_tag is the active pack.' }),
    );
    expect(await list.remove(row)).toBe(false);
    expect(list.deleteError).toBe('widget_tag is the active pack.');
  });

  it('delete: revision_conflict re-reads the list', async () => {
    const { list, deps } = setup();
    await list.load();
    deps.deletePromptPack.mockRejectedValue(
      refusal(409, {
        error: 'revision_conflict',
        message: 'Stale.',
        current_revision: 3,
      }),
    );
    await list.remove(list.list!.packs[1]!);
    expect(list.deleteError).toBe('Stale.');
    expect(deps.listPromptPacks).toHaveBeenCalledTimes(2);
  });

  it('clone sends the source, trimmed name and optional description', async () => {
    const { list, deps } = setup();
    const doc = await list.clone(
      { name: 'widget_tag', source: 'template' },
      ' widget_tag_2 ',
      '',
    );
    expect(doc?.name).toBe('widget_tag_2');
    expect(deps.clonePromptPack).toHaveBeenCalledWith('widget_tag', {
      new_name: 'widget_tag_2',
      revision: null,
      source: 'template',
      description: null,
    });
  });

  it('clone from another project sends from_project only when one is chosen', async () => {
    const { list, deps } = setup();
    await list.clone({ name: 'widget_tag', source: null }, 'copy', '', 'alpha');
    expect(deps.clonePromptPack).toHaveBeenLastCalledWith('widget_tag', {
      new_name: 'copy',
      revision: null,
      source: null,
      description: null,
      from_project: 'alpha',
    });
    await list.clone({ name: 'widget_tag', source: null }, 'copy2', '', null);
    const body = vi.mocked(deps.clonePromptPack).mock.calls.at(-1)![1];
    expect('from_project' in body).toBe(false);
  });

  it('clone: name_conflict and a validation report surface as served', async () => {
    const report = { ok: false, errors: [issue()], warnings: [], force_allowed: false };
    const { list } = setup({
      clonePromptPack: vi
        .fn()
        .mockRejectedValueOnce(
          refusal(409, { error: 'name_conflict', message: 'That name is taken.' }),
        )
        .mockRejectedValueOnce(
          refusal(422, { error: 'validation_failed', message: 'Fix the pack.', report }),
        ),
    });
    expect(await list.clone({ name: 'a', source: null }, 'b', '')).toBeNull();
    expect(list.cloneError).toBe('That name is taken.');
    expect(list.cloneReport).toBeNull();
    expect(await list.clone({ name: 'a', source: null }, 'b', '')).toBeNull();
    expect(list.cloneError).toBe('Fix the pack.');
    expect(list.cloneReport).toEqual(report);
  });

  it('re-reads on config.changed for the prompt_pack axis only; stop closes', async () => {
    const { list, deps, emit, close } = setup();
    list.start();
    await vi.waitFor(() => expect(deps.listPromptPacks).toHaveBeenCalledTimes(1));
    emit({ type: 'config.changed', axis: 'keymap' } as CurationEvent);
    emit({ type: 'crop.created' } as CurationEvent);
    expect(deps.listPromptPacks).toHaveBeenCalledTimes(1);
    emit({ type: 'config.changed', axis: 'prompt_pack', name: 'x' } as CurationEvent);
    await vi.waitFor(() => expect(deps.listPromptPacks).toHaveBeenCalledTimes(2));
    list.stop();
    expect(close).toHaveBeenCalled();
  });
});
