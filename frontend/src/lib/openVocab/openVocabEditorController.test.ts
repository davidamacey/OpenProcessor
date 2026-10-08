/**
 * The open-vocabulary editor binding: target edits that mark the draft
 * dirty and schedule the served validation, defaults that come from the
 * served schema, Save with `expected_revision`, the 409 conflict path and
 * the activation check.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api';
import type { CurationEvent } from '$lib/sse';
import {
  activeFixture,
  bodyFixture,
  cleanReport,
  docFixture,
  errorReport,
  issue,
  listFixture,
  revisionsFixture,
  schemaFixture,
} from './fixtures';
import {
  createOpenVocabEditor,
  type OpenVocabEditorDeps,
} from './openVocabEditorController.svelte';

beforeEach(() => vi.useFakeTimers());
afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

function refusal(status: number, detail: Record<string, unknown>): ApiError {
  return new ApiError(status, '/x', { detail });
}

function setup(over: Record<string, unknown> = {}) {
  let emit: (e: CurationEvent) => void = () => {};
  const deps = {
    getOpenVocabSchema: vi.fn().mockResolvedValue(schemaFixture()),
    listOpenVocab: vi.fn().mockResolvedValue(listFixture()),
    getOpenVocab: vi.fn().mockResolvedValue(docFixture()),
    getOpenVocabRevisions: vi.fn().mockResolvedValue(revisionsFixture()),
    getOpenVocabRevision: vi.fn().mockResolvedValue(docFixture({ revision: 2 })),
    updateOpenVocab: vi.fn().mockResolvedValue(docFixture({ revision: 4 })),
    validateOpenVocab: vi.fn().mockResolvedValue(cleanReport()),
    getActiveOpenVocab: vi.fn().mockResolvedValue(activeFixture()),
    activateOpenVocab: vi.fn(),
    rollbackOpenVocab: vi.fn(),
    deactivateOpenVocab: vi.fn(),
    subscribe: vi.fn((cb: (e: CurationEvent) => void) => {
      emit = cb;
      return { close: vi.fn() };
    }),
    ...over,
  };
  const ed = createOpenVocabEditor('widgets', deps as Partial<OpenVocabEditorDeps>);
  return { ed, deps, emit: (e: CurationEvent) => emit(e) };
}

describe('OpenVocabEditor', () => {
  it('loads the doc, schema and revisions', async () => {
    const { ed } = setup();
    await ed.load();
    expect(ed.doc?.revision).toBe(3);
    expect(ed.schema?.fields.length).toBeGreaterThan(0);
    expect(ed.revisions?.map((r) => r.revision)).toEqual([3, 2]);
    expect(ed.dirty).toBe(false);
  });

  it('reads the served segmenter fact alongside the doc', async () => {
    const { ed } = setup({
      listOpenVocab: vi
        .fn()
        .mockResolvedValue(
          listFixture({ segmenter: { configured: true, reachable: false } }),
        ),
    });
    await ed.load();
    expect(ed.segmenter).toEqual({ configured: true, reachable: false });
  });

  it('a failed segmenter read leaves the fact unread and does not fail the load', async () => {
    const { ed } = setup({ listOpenVocab: vi.fn().mockRejectedValue(new Error('boom')) });
    await ed.load();
    expect(ed.segmenter).toBeNull();
    expect(ed.loadError).toBeNull();
    expect(ed.doc?.revision).toBe(3);
  });

  it("addTarget appends a target built from the served rows' defaults", async () => {
    const { ed } = setup();
    await ed.load();
    ed.addTarget();
    expect(ed.draftBody.targets).toHaveLength(3);
    expect(ed.draftBody.targets![2]).toEqual({
      prompt: '',
      class_name: '',
      min_score: 0.5,
      enabled: true,
      max_instances: 20,
      parent_classes: [],
    });
    expect(ed.dirty).toBe(true);
  });

  it('a target edit marks the draft dirty and posts it to validate after the debounce', async () => {
    const { ed, deps } = setup();
    await ed.load();
    ed.setTargetField(1, 'prompt', 'chipped widget');
    expect(ed.draftBody.targets![1]!.prompt).toBe('chipped widget');
    expect(ed.dirty).toBe(true);
    expect(deps.validateOpenVocab).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(400);
    expect(deps.validateOpenVocab).toHaveBeenCalledTimes(1);
    const [req, forActivation] = deps.validateOpenVocab.mock.calls[0]!;
    expect(req).toEqual({ name: null, body: ed.draftBody });
    expect(forActivation).toBe(false);
  });

  it('removes and moves targets, keeping the order the operator chose', async () => {
    const { ed } = setup();
    await ed.load();
    ed.moveTarget(1, -1);
    expect(ed.draftBody.targets!.map((t) => t.prompt)).toEqual([
      'cracked widget',
      'blue widget',
    ]);
    ed.moveTarget(0, -1);
    expect(ed.draftBody.targets!.map((t) => t.prompt)).toEqual([
      'cracked widget',
      'blue widget',
    ]);
    ed.removeTarget(0);
    expect(ed.draftBody.targets!.map((t) => t.prompt)).toEqual(['blue widget']);
  });

  it('sets and gating fields, including the hit-rate sub-object', async () => {
    const { ed } = setup();
    await ed.load();
    ed.setGatingField('tier2_vlm_precheck', true);
    ed.setHitRateField('window', 30);
    expect(ed.draftBody.gating).toEqual({
      tier2_vlm_precheck: true,
      tier3_hit_rate: { enabled: false, window: 30 },
    });
  });

  it('refuses every edit on a read-only doc', async () => {
    const { ed } = setup({
      getOpenVocab: vi.fn().mockResolvedValue(docFixture({ read_only: true })),
    });
    await ed.load();
    ed.addTarget();
    ed.setTargetField(0, 'prompt', 'x');
    ed.removeTarget(0);
    expect(ed.dirty).toBe(false);
  });

  it('keeps the served live-validation issues for the page to place', async () => {
    const report = errorReport(issue({ field: 'targets[0].prompt' }));
    const { ed } = setup({ validateOpenVocab: vi.fn().mockResolvedValue(report) });
    await ed.load();
    ed.setTargetField(0, 'prompt', '');
    await vi.advanceTimersByTimeAsync(400);
    expect(ed.report?.errors[0]?.field).toBe('targets[0].prompt');
  });

  it('Save sends expected_revision and adopts the served doc', async () => {
    const { ed, deps } = setup();
    await ed.load();
    ed.setTargetField(0, 'min_score', 0.7);
    expect(await ed.save()).toBe(true);
    const [name, req] = deps.updateOpenVocab.mock.calls[0]!;
    expect(name).toBe('widgets');
    expect(req.expected_revision).toBe(3);
    expect(req.body.targets[0].min_score).toBe(0.7);
    expect(ed.doc?.revision).toBe(4);
    expect(ed.dirty).toBe(false);
  });

  it('a 409 revision_conflict offers reload or keep-mine', async () => {
    const { ed, deps } = setup({
      updateOpenVocab: vi.fn().mockRejectedValue(
        refusal(409, {
          error: 'revision_conflict',
          message: 'Changed elsewhere.',
          current_revision: 5,
        }),
      ),
    });
    await ed.load();
    ed.setTargetField(0, 'min_score', 0.7);
    expect(await ed.save()).toBe(false);
    expect(ed.conflict).toEqual({ message: 'Changed elsewhere.', currentRevision: 5 });
    ed.keepMine();
    expect(ed.expectedRevision).toBe(5);
    await ed.save();
    expect(deps.updateOpenVocab.mock.calls[1]![1].expected_revision).toBe(5);
  });

  it('checkActivation posts the draft with for_activation and keeps the report apart', async () => {
    const act = errorReport(
      issue({ code: 'open_vocab_no_enabled_targets', field: null }),
    );
    const validate = vi.fn().mockResolvedValue(act);
    const { ed } = setup({ validateOpenVocab: validate });
    await ed.load();
    await ed.checkActivation();
    expect(validate).toHaveBeenCalledWith({ name: null, body: bodyFixture() }, true);
    expect(ed.activationReport?.errors[0]?.code).toBe('open_vocab_no_enabled_targets');
    expect(ed.report).not.toBe(ed.activationReport);
  });

  it('wakes on config.changed for the open_vocab axis only', async () => {
    const { ed, deps, emit } = setup();
    ed.start();
    await vi.advanceTimersByTimeAsync(0);
    const before = deps.getOpenVocab.mock.calls.length;
    emit({ type: 'config.changed', axis: 'detection_profile', name: 'widgets' } as never);
    await vi.advanceTimersByTimeAsync(0);
    expect(deps.getOpenVocab.mock.calls.length).toBe(before);
    emit({ type: 'config.changed', axis: 'open_vocab', name: 'widgets' } as never);
    await vi.advanceTimersByTimeAsync(0);
    expect(deps.getOpenVocab.mock.calls.length).toBe(before + 1);
    ed.stop();
  });
});
