/**
 * One pack's editor state: the draft, the server's live validation,
 * Save with `expected_revision` and its conflict paths, revisions and
 * restore, pinned activation with `force` only after a served
 * `force_allowed`, and the `config.changed` wake-up.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api';
import type { CurationEvent } from '$lib/sse';
import {
  activeFixture,
  builtinDocFixture,
  cleanReport,
  docFixture,
  issue,
  revisionsFixture,
  schemaFixture,
} from '$lib/test/fixtures/promptPacks';
import type { ValidationReport } from '$lib/types_packs';
import {
  createPackEditor,
  issuesForField,
  unplacedIssues,
  VALIDATE_DEBOUNCE_MS,
  type PackEditorDeps,
} from './packEditorController.svelte';

function refusal(status: number, detail: Record<string, unknown>): ApiError {
  return new ApiError(status, '/x', { detail });
}

const errorReport = (): ValidationReport => ({
  ok: false,
  errors: [issue()],
  warnings: [issue({ code: 'pack_example_values', severity: 'info', field: null })],
  force_allowed: false,
});

function setup(over: Partial<PackEditorDeps> = {}, doc = docFixture()) {
  let emit: (e: CurationEvent) => void = () => {};
  const close = vi.fn();
  const deps = {
    getPromptPackSchema: vi.fn().mockResolvedValue(schemaFixture()),
    getPromptPack: vi.fn().mockResolvedValue(doc),
    getPromptPackRevisions: vi.fn().mockResolvedValue(revisionsFixture()),
    getPromptPackRevision: vi.fn().mockResolvedValue(
      docFixture({
        revision: 1,
        read_only: true,
        description: 'First cut',
        body: { class_system: 'old' },
      }),
    ),
    updatePromptPack: vi
      .fn()
      .mockImplementation(
        async (
          _n: string,
          b: { body: Record<string, unknown>; description: string | null },
        ) =>
          docFixture({ revision: 3, body: b.body as never, description: b.description }),
      ),
    validatePromptPack: vi.fn().mockResolvedValue(errorReport()),
    getActivePromptPack: vi.fn().mockResolvedValue(activeFixture()),
    activatePromptPack: vi
      .fn()
      .mockResolvedValue(activeFixture({ active: { name: 'widget_tag', revision: 2 } })),
    rollbackPromptPack: vi.fn(),
    subscribe: vi.fn((cb: (e: CurationEvent) => void) => {
      emit = cb;
      return { close };
    }),
  };
  const ed = createPackEditor('widget_tag', { ...deps, ...over });
  return { ed, deps, emit: (e: CurationEvent) => emit(e), close };
}

afterEach(() => {
  vi.useRealTimers();
});

describe('PackEditor load and draft', () => {
  it('loads schema, doc, active and revisions; the draft starts clean', async () => {
    const { ed, deps } = setup();
    await ed.load();
    expect(ed.schema?.fields).toHaveLength(4);
    expect(ed.draftBody).toEqual(docFixture().body);
    expect(ed.expectedRevision).toBe(2);
    expect(ed.report).toEqual(cleanReport());
    expect(ed.revisions?.map((r) => r.revision)).toEqual([2, 1]);
    expect(ed.active.active?.active.revision).toBe(1);
    expect(ed.dirty).toBe(false);
    expect(ed.editable).toBe(true);
    expect(deps.getPromptPack).toHaveBeenCalledWith('widget_tag');
  });

  it('a read-only pack is not editable and has no revision list', async () => {
    const { ed, deps } = setup({}, builtinDocFixture());
    await ed.load();
    expect(ed.editable).toBe(false);
    expect(ed.revisions).toBeNull();
    expect(deps.getPromptPackRevisions).not.toHaveBeenCalled();
    ed.setField('class_system', 'x');
    expect(ed.dirty).toBe(false);
  });

  it('a missing pack shows the served message', async () => {
    const { ed } = setup({
      getPromptPack: vi
        .fn()
        .mockRejectedValue(
          refusal(404, { error: 'not_found', message: 'No pack named x.' }),
        ),
    });
    await ed.load();
    expect(ed.loadError).toBe('No pack named x.');
  });

  it('an edit makes the draft dirty and never touches the served doc', async () => {
    const { ed } = setup();
    await ed.load();
    ed.setField('synonyms', { thingamajig: 'widget' });
    expect(ed.dirty).toBe(true);
    expect(ed.doc?.body.synonyms).toEqual({ doohickey: 'gadget' });
    ed.setField('synonyms', { doohickey: 'gadget' });
    expect(ed.dirty).toBe(false);
    ed.setDescription('changed');
    expect(ed.dirty).toBe(true);
  });
});

describe('PackEditor validation', () => {
  it('posts one validate after a burst of edits, with name null, and shows the served report', async () => {
    vi.useFakeTimers();
    const { ed, deps } = setup();
    await ed.load();
    ed.setField('class_user_template', 'a');
    ed.setField('class_user_template', 'ab');
    await vi.advanceTimersByTimeAsync(VALIDATE_DEBOUNCE_MS - 1);
    expect(deps.validatePromptPack).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1);
    expect(deps.validatePromptPack).toHaveBeenCalledTimes(1);
    expect(deps.validatePromptPack.mock.calls[0]![0]).toEqual({
      name: null,
      body: { ...docFixture().body, class_user_template: 'ab' },
    });
    await vi.waitFor(() => expect(ed.report).toEqual(errorReport()));
    expect(issuesForField(ed.report, 'class_user_template').map((i) => i.code)).toEqual([
      'pack_placeholder_missing',
    ]);
    expect(
      unplacedIssues(
        ed.report,
        ed.schema!.fields.map((f) => f.field),
      ).map((i) => i.code),
    ).toEqual(['pack_example_values']);
  });

  it('attaches a dotted map-entry path to its map field only', () => {
    const report: ValidationReport = {
      ok: false,
      errors: [],
      warnings: [
        issue({
          code: 'pack_synonym_target_unknown',
          severity: 'warning',
          field: 'synonyms.foo',
        }),
        issue({ code: 'x', severity: 'warning', field: 'synonymsx' }),
      ],
      force_allowed: false,
    };
    expect(issuesForField(report, 'synonyms').map((i) => i.code)).toEqual([
      'pack_synonym_target_unknown',
    ]);
    expect(unplacedIssues(report, ['synonyms']).map((i) => i.code)).toEqual(['x']);
  });

  it('a failed validate shows its error and keeps the last report', async () => {
    const { ed } = setup({
      validatePromptPack: vi
        .fn()
        .mockRejectedValue(new ApiError(500, '/v', { detail: 'boom' })),
    });
    await ed.load();
    await ed.validateNow();
    expect(ed.validateError).toBe('boom');
    expect(ed.report).toEqual(cleanReport());
  });
});

describe('PackEditor save', () => {
  it('sends expected_revision, the draft body and description; adopts the served doc', async () => {
    const { ed, deps } = setup();
    await ed.load();
    expect(ed.canSave).toBe(false);
    ed.setField('class_system', 'new text');
    ed.setDescription('  Tags  ');
    expect(await ed.save()).toBe(true);
    expect(deps.updatePromptPack).toHaveBeenCalledWith('widget_tag', {
      expected_revision: 2,
      description: 'Tags',
      body: { ...docFixture().body, class_system: 'new text' },
    });
    expect(ed.doc?.revision).toBe(3);
    expect(ed.expectedRevision).toBe(3);
    expect(ed.dirty).toBe(false);
    expect(deps.getPromptPackRevisions).toHaveBeenCalledTimes(2);
  });

  it('revision_conflict: reload drops the edits; keep-mine adopts the served revision', async () => {
    const { ed, deps } = setup();
    await ed.load();
    deps.updatePromptPack.mockRejectedValueOnce(
      refusal(409, {
        error: 'revision_conflict',
        message: 'Revision 4 was saved after you opened this pack.',
        current_revision: 4,
      }),
    );
    ed.setField('class_system', 'mine');
    expect(await ed.save()).toBe(false);
    expect(ed.conflict).toEqual({
      message: 'Revision 4 was saved after you opened this pack.',
      currentRevision: 4,
    });
    ed.keepMine();
    expect(ed.expectedRevision).toBe(4);
    expect(ed.conflict).toBeNull();
    expect(ed.draftBody.class_system).toBe('mine');
    await ed.save();
    expect(deps.updatePromptPack.mock.calls[1]![1].expected_revision).toBe(4);

    deps.updatePromptPack.mockRejectedValueOnce(
      refusal(409, {
        error: 'revision_conflict',
        message: 'Stale again.',
        current_revision: 5,
      }),
    );
    ed.setField('class_system', 'mine again');
    await ed.save();
    deps.getPromptPack.mockResolvedValue(
      docFixture({ revision: 5, body: { class_system: 'theirs' } }),
    );
    await ed.reloadLatest();
    expect(ed.draftBody).toEqual({ class_system: 'theirs' });
    expect(ed.expectedRevision).toBe(5);
    expect(ed.conflict).toBeNull();
  });

  it('422 validation_failed shows the message and the served report', async () => {
    const report = errorReport();
    const { ed } = setup({
      updatePromptPack: vi.fn().mockRejectedValue(
        refusal(422, {
          error: 'validation_failed',
          message: 'The pack has errors.',
          report,
        }),
      ),
    });
    await ed.load();
    ed.setField('class_system', 'x');
    expect(await ed.save()).toBe(false);
    expect(ed.saveError).toBe('The pack has errors.');
    expect(ed.report).toEqual(report);
    expect(ed.conflict).toBeNull();
  });
});

describe('PackEditor revisions', () => {
  it('views a revision read-only and restores it as a new revision', async () => {
    const { ed, deps } = setup();
    await ed.load();
    await ed.viewRevision(1);
    expect(deps.getPromptPackRevision).toHaveBeenCalledWith('widget_tag', 1);
    expect(ed.viewing?.revision).toBe(1);
    expect(ed.editable).toBe(false);
    expect(await ed.restoreViewed()).toBe(true);
    expect(deps.updatePromptPack).toHaveBeenCalledWith('widget_tag', {
      expected_revision: 2,
      description: 'First cut',
      body: { class_system: 'old' },
    });
    expect(ed.viewing).toBeNull();
    expect(ed.doc?.revision).toBe(3);
  });

  it('a missing revision shows the served message', async () => {
    const { ed } = setup({
      getPromptPackRevision: vi
        .fn()
        .mockRejectedValue(
          refusal(404, { error: 'unknown_revision', message: 'No revision 9.' }),
        ),
    });
    await ed.load();
    await ed.viewRevision(9);
    expect(ed.revisionError).toBe('No revision 9.');
    expect(ed.viewing).toBeNull();
  });
});

describe('PackEditor activation', () => {
  it('pins the revision and sends expected_active from the last read', async () => {
    const { ed, deps } = setup();
    await ed.load();
    expect(await ed.active.activate('widget_tag', 2, false)).toBe(true);
    expect(deps.activatePromptPack).toHaveBeenCalledWith('widget_tag', {
      revision: 2,
      expected_active: { name: 'widget_tag', revision: 1 },
      force: false,
    });
    expect(ed.active.active?.active).toEqual({ name: 'widget_tag', revision: 2 });
    await ed.active.activate('generic_item_v1', null, true);
    expect(deps.activatePromptPack).toHaveBeenLastCalledWith('generic_item_v1', {
      revision: null,
      expected_active: { name: 'widget_tag', revision: 2 },
      force: true,
    });
  });

  it('422 keeps the served report so the page can offer force only when allowed', async () => {
    const report: ValidationReport = {
      ok: false,
      errors: [issue({ code: 'pack_description_class_unknown', bypassable: true })],
      warnings: [],
      force_allowed: true,
    };
    const { ed } = setup({
      activatePromptPack: vi.fn().mockRejectedValue(
        refusal(422, {
          error: 'validation_failed',
          message: 'Cannot activate.',
          report,
        }),
      ),
    });
    await ed.load();
    expect(await ed.active.activate('widget_tag', 2, false)).toBe(false);
    expect(ed.active.actionError).toBe('Cannot activate.');
    expect(ed.active.activateReport?.force_allowed).toBe(true);
  });

  it('active_conflict re-reads the active pack', async () => {
    const { ed, deps } = setup({
      activatePromptPack: vi.fn().mockRejectedValue(
        refusal(409, {
          error: 'active_conflict',
          message: 'Someone activated another pack.',
        }),
      ),
    });
    await ed.load();
    deps.getActivePromptPack.mockResolvedValue(
      activeFixture({ active: { name: 'other', revision: 7 } }),
    );
    await ed.active.activate('widget_tag', 2, false);
    expect(ed.active.actionError).toBe('Someone activated another pack.');
    expect(ed.active.active?.active).toEqual({ name: 'other', revision: 7 });
  });
});

describe('PackEditor events', () => {
  it('re-reads a clean pack on its own event; flags a dirty one instead', async () => {
    const { ed, deps, emit, close } = setup();
    ed.start();
    await vi.waitFor(() => expect(ed.doc).not.toBeNull());
    deps.getPromptPack.mockResolvedValue(docFixture({ revision: 4 }));

    emit({ type: 'config.changed', axis: 'prompt_pack', name: 'other' } as CurationEvent);
    await vi.waitFor(() => expect(deps.getActivePromptPack).toHaveBeenCalledTimes(2));
    expect(deps.getPromptPack).toHaveBeenCalledTimes(1);

    emit({
      type: 'config.changed',
      axis: 'prompt_pack',
      name: 'widget_tag',
    } as CurationEvent);
    await vi.waitFor(() => expect(ed.doc?.revision).toBe(4));

    ed.setField('class_system', 'mine');
    emit({
      type: 'config.changed',
      axis: 'prompt_pack',
      name: 'widget_tag',
    } as CurationEvent);
    await vi.waitFor(() => expect(ed.remoteChanged).toBe(true));
    expect(ed.draftBody.class_system).toBe('mine');
    expect(deps.getPromptPack).toHaveBeenCalledTimes(2);

    emit({ type: 'config.changed', axis: 'keymap' } as CurationEvent);
    ed.stop();
    expect(close).toHaveBeenCalled();
    expect(deps.getActivePromptPack).toHaveBeenCalledTimes(4);
  });
});
