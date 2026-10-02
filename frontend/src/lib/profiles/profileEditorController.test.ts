/**
 * One region profile's editor: schema, doc, active and the vocabulary load
 * together; the debounced validate posts `{name: null, body}` without
 * `for_activation`; "Check for activation" posts the draft with it and
 * keeps that report apart (cleared by the next edit); Save and its
 * conflict; activation pins the revision, keeps the served impact and
 * polls `/health` only on success; the vocabulary re-reads with other
 * projects on request; the `detection_profile` wake-up.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api';
import { VALIDATE_DEBOUNCE_MS } from '$lib/config/configEditor.svelte';
import type { CurationEvent } from '$lib/sse';
import { cleanReport, issue } from '$lib/test/fixtures/promptPacks';
import {
  activateResponseFixture,
  profileActiveFixture,
  profileDocFixture,
  profileRevisionsFixture,
  profileSchemaFixture,
  vocabularyFixture,
} from '$lib/test/fixtures/regionProfiles';
import {
  createProfileEditor,
  type ProfileEditorDeps,
} from './profileEditorController.svelte';

function refusal(status: number, detail: Record<string, unknown>): ApiError {
  return new ApiError(status, '/x', { detail });
}

afterEach(() => vi.useRealTimers());

function setup(over: Partial<ProfileEditorDeps> = {}) {
  let emit: (e: CurationEvent) => void = () => {};
  const base = {
    getRegionProfileSchema: vi.fn().mockResolvedValue(profileSchemaFixture()),
    getRegionProfile: vi.fn().mockResolvedValue(profileDocFixture()),
    getRegionProfileRevisions: vi.fn().mockResolvedValue(profileRevisionsFixture()),
    getRegionProfileRevision: vi
      .fn()
      .mockResolvedValue(profileDocFixture({ revision: 1, read_only: true })),
    updateRegionProfile: vi
      .fn()
      .mockImplementation(async (_n: string, b: { body: Record<string, never> }) =>
        profileDocFixture({ revision: 4, body: b.body }),
      ),
    validateRegionProfile: vi.fn().mockResolvedValue(cleanReport()),
    getConfigVocabulary: vi.fn().mockResolvedValue(vocabularyFixture()),
    getActiveRegionProfile: vi.fn().mockResolvedValue(profileActiveFixture()),
    activateRegionProfile: vi.fn().mockResolvedValue(activateResponseFixture()),
    rollbackRegionProfile: vi.fn(),
    deactivateRegionProfile: vi.fn(),
    onchanged: vi.fn(),
    subscribe: vi.fn((cb: (e: CurationEvent) => void) => {
      emit = cb;
      return { close: vi.fn() };
    }),
  };
  const deps = { ...base, ...over } as typeof base;
  const ed = createProfileEditor('widget_tag', deps);
  return { ed, deps, emit: (e: CurationEvent) => emit(e) };
}

describe('ProfileEditor', () => {
  it('loads the schema, doc, active profile, vocabulary and revisions', async () => {
    const { ed, deps } = setup();
    await ed.load();
    expect(deps.getRegionProfile).toHaveBeenCalledWith('widget_tag');
    expect(deps.getConfigVocabulary).toHaveBeenCalledWith(false);
    expect(ed.schema?.groups[0]?.id).toBe('identity');
    expect(ed.doc?.effective?.legs).toEqual(['detector', 'segmenter']);
    expect(ed.vocabulary?.detectors).toHaveLength(2);
    expect(ed.revisions).toHaveLength(3);
    expect(ed.active.active?.active).toEqual({ name: 'widget_tag', revision: 2 });
  });

  it('a failed vocabulary read is shown, not fatal', async () => {
    const { ed } = setup({
      getConfigVocabulary: vi
        .fn()
        .mockRejectedValue(
          refusal(503, { error: 'config_store_unavailable', message: 'Down.' }),
        ),
    });
    await ed.load();
    expect(ed.doc).not.toBeNull();
    expect(ed.loadError).toBeNull();
    expect(ed.vocabularyError).toBe('Down.');
  });

  it('live validation posts {name: null, body} without for_activation, once per burst', async () => {
    vi.useFakeTimers();
    const { ed, deps } = setup();
    await ed.load();
    ed.setField('max_regions_per_item', 8);
    ed.setField('max_regions_per_item', 9);
    await vi.advanceTimersByTimeAsync(VALIDATE_DEBOUNCE_MS);
    expect(deps.validateRegionProfile).toHaveBeenCalledTimes(1);
    const [body, forActivation] = deps.validateRegionProfile.mock.calls[0]!;
    expect(body).toEqual({
      name: null,
      body: { ...profileDocFixture().body, max_regions_per_item: 9 },
    });
    expect(forActivation).toBe(false);
  });

  it('check for activation posts the draft with for_activation and keeps that report apart', async () => {
    const activationReport = {
      ok: false,
      errors: [
        issue({
          code: 'detector_model_not_ready',
          field: 'detector_model',
          bypassable: true,
        }),
      ],
      warnings: [],
      force_allowed: true,
    };
    const { ed, deps } = setup({
      validateRegionProfile: vi
        .fn()
        .mockImplementation(async (_b: unknown, forActivation: boolean) =>
          forActivation ? activationReport : cleanReport(),
        ),
    });
    await ed.load();
    await ed.checkForActivation();
    expect(deps.validateRegionProfile).toHaveBeenCalledWith(
      { name: null, body: profileDocFixture().body },
      true,
    );
    expect(ed.activationReport).toEqual(activationReport);
    expect(ed.report).toEqual(cleanReport());
    ed.setField('detector_model', '');
    expect(ed.activationReport).toBeNull();
  });

  it('save sends expected_revision; a conflict can keep my edits', async () => {
    const { ed, deps } = setup();
    await ed.load();
    ed.setField('segmenter_text_prompt', 'label');
    deps.updateRegionProfile.mockRejectedValueOnce(
      refusal(409, {
        error: 'revision_conflict',
        message: 'Saved elsewhere.',
        current_revision: 5,
      }),
    );
    expect(await ed.save()).toBe(false);
    expect(deps.updateRegionProfile.mock.calls[0]![1]).toMatchObject({
      expected_revision: 3,
    });
    expect(ed.conflict?.message).toBe('Saved elsewhere.');
    ed.keepMine();
    expect(await ed.save()).toBe(true);
    expect(deps.updateRegionProfile.mock.calls[1]![1]).toMatchObject({
      expected_revision: 5,
      body: { segmenter_text_prompt: 'label' },
    });
    expect(ed.doc?.revision).toBe(4);
  });

  it('activation pins the revision, keeps the served impact and polls health once', async () => {
    const { ed, deps } = setup();
    await ed.load();
    expect(await ed.active.activate('widget_tag', 3, false)).toBe(true);
    expect(deps.activateRegionProfile).toHaveBeenCalledWith('widget_tag', {
      revision: 3,
      expected_active: { name: 'widget_tag', revision: 2 },
      force: false,
    });
    expect(ed.active.lastActivation?.impact?.suggested_reprocess?.scopes).toEqual([
      'region',
    ]);
    expect(deps.onchanged).toHaveBeenCalledTimes(1);
  });

  it("a refused activation with force_allowed keeps the report; force is the caller's", async () => {
    const report = {
      ok: false,
      errors: [issue({ code: 'segmenter_unreachable', bypassable: true })],
      warnings: [],
      force_allowed: true,
    };
    const { ed, deps } = setup({
      activateRegionProfile: vi
        .fn()
        .mockRejectedValueOnce(
          refusal(422, { error: 'validation_failed', message: 'Not ready.', report }),
        )
        .mockResolvedValueOnce(activateResponseFixture()),
    });
    await ed.load();
    expect(await ed.active.activate('widget_tag', 3, false)).toBe(false);
    expect(ed.active.activateReport?.force_allowed).toBe(true);
    expect(deps.onchanged).not.toHaveBeenCalled();
    expect(await ed.active.activate('widget_tag', 3, true)).toBe(true);
    expect(deps.activateRegionProfile.mock.calls[1]![1]).toMatchObject({ force: true });
  });

  it('including other projects re-reads the vocabulary with the flag', async () => {
    const { ed, deps } = setup();
    await ed.load();
    await ed.setIncludeOtherProjects(true);
    expect(deps.getConfigVocabulary).toHaveBeenLastCalledWith(true);
    await ed.setIncludeOtherProjects(true);
    expect(deps.getConfigVocabulary).toHaveBeenCalledTimes(2);
  });

  it('follows detection_profile events: a clean draft re-reads, a dirty one is told', async () => {
    const { ed, deps, emit } = setup();
    ed.start();
    await vi.waitFor(() => expect(ed.doc).not.toBeNull());
    emit({ type: 'config.changed', axis: 'prompt_pack', name: 'widget_tag' } as never);
    emit({
      type: 'config.changed',
      axis: 'detection_profile',
      name: 'widget_tag',
    } as never);
    await vi.waitFor(() => expect(deps.getRegionProfile).toHaveBeenCalledTimes(2));
    ed.setField('segmenter_text_prompt', 'mine');
    emit({
      type: 'config.changed',
      axis: 'detection_profile',
      name: 'widget_tag',
    } as never);
    await vi.waitFor(() => expect(ed.remoteChanged).toBe(true));
    expect(deps.getRegionProfile).toHaveBeenCalledTimes(2);
    ed.stop();
  });
});
