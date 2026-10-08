/**
 * The combine wizard's state: a debounced preview (one request per burst,
 * the previous aborted), untouched options omitted from the body,
 * served suggestions filling untouched mapping rows only, `map` choices
 * following the form's own `create` rows, and Start gated on a fresh,
 * ok preview with `expected_preview_sha`.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api';
import { combinePreview } from '$lib/test/fixtures/combine';
import type { CombinePreview } from '$lib/types_combine';
import {
  COMBINE_PREVIEW_DEBOUNCE_MS,
  createCombineWizard,
  type CombineWizardDeps,
} from './combineWizardController.svelte';

type PreviewFn = CombineWizardDeps['previewCombine'];
type StartFn = CombineWizardDeps['startCombine'];

function setup(
  preview: PreviewFn = async () => combinePreview(),
  start: StartFn = async () => ({ job_id: 'cmb_1', target: 'merged' }),
) {
  const previewCombine = vi.fn(preview);
  const startCombine = vi.fn(start);
  const getDatasetFormatsFor = vi.fn(async (_project: { prefix: string }) => ({
    mapping_actions: [
      { value: 'map', label: 'Map to class', description: 'Point at a created class' },
      { value: 'create', label: 'Create class', description: '' },
    ],
  }));
  const wizard = createCombineWizard({
    previewCombine,
    startCombine,
    getDatasetFormatsFor,
    projectOf: (slug) => ({ prefix: `/curation/projects/${slug}` }),
  });
  return { wizard, previewCombine, startCombine, getDatasetFormatsFor };
}

function fill(w: ReturnType<typeof setup>['wizard']): void {
  w.addSource('widgets-a');
  w.addSource('widgets-b');
  w.setTarget({ slug: 'merged', displayName: 'Merged' });
}

const settle = () => vi.advanceTimersByTimeAsync(COMBINE_PREVIEW_DEBOUNCE_MS + 1);

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

describe('preview', () => {
  it('one request per burst of changes', async () => {
    const { wizard, previewCombine } = setup();
    wizard.addSource('widgets-a');
    wizard.setTarget({ slug: 'm' });
    wizard.setTarget({ slug: 'me' });
    wizard.setTarget({ slug: 'merged', displayName: 'Merged' });
    expect(previewCombine).not.toHaveBeenCalled();
    await settle();
    expect(previewCombine).toHaveBeenCalledTimes(1);
    expect(previewCombine.mock.calls[0]![0].target.slug).toBe('merged');
  });

  it('a new run aborts the previous in-flight request and drops its answer', async () => {
    const signals: AbortSignal[] = [];
    let resolveFirst: (p: CombinePreview) => void = () => {};
    const { wizard, previewCombine } = setup((_body, signal) => {
      signals.push(signal!);
      if (signals.length === 1)
        return new Promise<CombinePreview>((r) => (resolveFirst = r));
      return Promise.resolve(combinePreview({ preview_sha: 'second' }));
    });
    fill(wizard);
    await settle();
    expect(previewCombine).toHaveBeenCalledTimes(1);
    wizard.setTarget({ description: 'changed' });
    await settle();
    expect(signals[0]!.aborted).toBe(true);
    resolveFirst(combinePreview({ preview_sha: 'first' }));
    await vi.advanceTimersByTimeAsync(0);
    expect(wizard.preview?.preview_sha).toBe('second');
  });

  it('does not preview without a source, a slug and a display name', async () => {
    const { wizard, previewCombine } = setup();
    wizard.addSource('widgets-a');
    wizard.setTarget({ slug: 'merged' });
    await settle();
    expect(previewCombine).not.toHaveBeenCalled();
    expect(wizard.canPreview).toBe(false);
  });

  it('a failed preview shows the served message and clears the stale preview', async () => {
    const { wizard } = setup(async () => {
      throw new ApiError(422, '/u', {
        detail: { error: 'invalid', message: 'served: nope' },
      });
    });
    fill(wizard);
    await settle();
    expect(wizard.previewError).toBe('served: nope');
    expect(wizard.preview).toBeNull();
    expect(wizard.canStart).toBe(false);
  });
});

describe('request body', () => {
  it('omits every option and label_states the operator never touched', async () => {
    const { wizard } = setup();
    fill(wizard);
    expect(wizard.requestBody()).toEqual({
      target: { slug: 'merged', display_name: 'Merged' },
      sources: [{ project: 'widgets-a' }, { project: 'widgets-b' }],
    });
  });

  it('sends touched options, label_states, description and a priority order', async () => {
    const { wizard } = setup();
    fill(wizard);
    wizard.setTarget({ description: ' tags ' });
    wizard.setLabelStates('widgets-b', 'validated_only');
    wizard.setOption('dedup', 'none');
    wizard.setOption('dedup_iou', 0.5);
    wizard.setOption('holdout', 'recompute');
    wizard.setOption('settings_from', 'widgets-b');
    wizard.moveSource('widgets-b', -1);
    expect(wizard.requestBody()).toEqual({
      target: { slug: 'merged', display_name: 'Merged', description: 'tags' },
      sources: [
        { project: 'widgets-b', include: { label_states: 'validated_only' } },
        { project: 'widgets-a' },
      ],
      dedup: 'none',
      dedup_iou: 0.5,
      holdout: 'recompute',
      settings_from: 'widgets-b',
    });
    wizard.setOption('dedup', undefined);
    expect(wizard.requestBody()).not.toHaveProperty('dedup');
  });

  it('removing a source drops its rows and a settings_from naming it', () => {
    const { wizard } = setup();
    fill(wizard);
    wizard.setChoice('widgets-b', 'widget', { action: 'skip' });
    wizard.setOption('settings_from', 'widgets-b');
    wizard.removeSource('widgets-b');
    expect(wizard.choices['widgets-b']).toBeUndefined();
    expect(wizard.options.settings_from).toBeUndefined();
    expect(wizard.requestBody().class_mapping).toBeUndefined();
  });

  it('mapping rows: create/map carry new_class_name, skip/region do not', () => {
    const { wizard } = setup();
    fill(wizard);
    wizard.setChoice('widgets-a', 'widget', { action: 'create' });
    wizard.setChoice('widgets-a', 'gadget', { action: 'skip' });
    wizard.setChoice('widgets-b', 'widget', { action: 'map', new_class_name: 'widget' });
    wizard.setChoice('widgets-b', 'tag', { action: 'region' });
    expect(wizard.requestBody().class_mapping).toEqual({
      'widgets-a': [
        { dataset_class: 'widget', action: 'create', new_class_name: 'widget' },
        { dataset_class: 'gadget', action: 'skip' },
      ],
      'widgets-b': [
        { dataset_class: 'widget', action: 'map', new_class_name: 'widget' },
        { dataset_class: 'tag', action: 'region' },
      ],
    });
  });
});

describe('served suggestions', () => {
  it('fill untouched rows after a preview, then re-preview once with them', async () => {
    const { wizard, previewCombine } = setup();
    fill(wizard);
    await settle();
    expect(previewCombine).toHaveBeenCalledTimes(1);
    expect(wizard.choiceFor('widgets-a', 'widget')).toMatchObject({
      action: 'create',
      new_class_name: 'widget',
      touched: false,
    });
    expect(wizard.choiceFor('widgets-b', 'widget').action).toBe('map');
    await settle();
    expect(previewCombine).toHaveBeenCalledTimes(2);
    expect(previewCombine.mock.calls[1]![0].class_mapping?.['widgets-b']).toEqual([
      { dataset_class: 'widget', action: 'map', new_class_name: 'widget' },
    ]);
    // The second answer's suggestions change nothing: no third request.
    await settle();
    expect(previewCombine).toHaveBeenCalledTimes(2);
  });

  it('never overwrite a touched row', async () => {
    const { wizard, previewCombine } = setup();
    fill(wizard);
    wizard.setChoice('widgets-a', 'gadget', { action: 'skip' });
    await settle();
    await settle();
    expect(wizard.choiceFor('widgets-a', 'gadget')).toMatchObject({
      action: 'skip',
      touched: true,
    });
    // But the untouched sibling did take its suggestion.
    expect(wizard.choiceFor('widgets-a', 'widget').action).toBe('create');
    expect(
      previewCombine.mock.calls.at(-1)![0].class_mapping?.['widgets-a'],
    ).toContainEqual({
      dataset_class: 'gadget',
      action: 'skip',
    });
  });

  it('"Reset to suggestions" re-copies them over touched rows', async () => {
    const { wizard } = setup();
    fill(wizard);
    await settle();
    await settle();
    wizard.setChoice('widgets-a', 'gadget', { action: 'skip' });
    wizard.resetToSuggestions();
    expect(wizard.choiceFor('widgets-a', 'gadget')).toMatchObject({
      action: 'create',
      new_class_name: 'gadget',
      touched: false,
    });
  });

  it('map choices are the names the create rows define, across sources', () => {
    const { wizard } = setup();
    fill(wizard);
    expect(wizard.createdNames).toEqual([]);
    wizard.setChoice('widgets-a', 'widget', { action: 'create' });
    wizard.setChoice('widgets-b', 'sprocket', {
      action: 'create',
      new_class_name: 'cog',
    });
    wizard.setChoice('widgets-b', 'widget', { action: 'map', new_class_name: 'widget' });
    expect(wizard.createdNames).toEqual(['widget', 'cog']);
    wizard.setChoice('widgets-b', 'sprocket', { action: 'skip' });
    expect(wizard.createdNames).toEqual(['widget']);
  });
});

describe('start', () => {
  it('is blocked until a fresh, ok preview is on screen', async () => {
    const { wizard } = setup(async () => combinePreview({ ok: false }));
    fill(wizard);
    expect(wizard.canStart).toBe(false);
    await settle();
    await settle();
    expect(wizard.preview?.ok).toBe(false);
    expect(wizard.canStart).toBe(false);
  });

  it('is blocked while stale and sends expected_preview_sha once fresh', async () => {
    const { wizard, startCombine } = setup();
    fill(wizard);
    await settle();
    await settle();
    expect(wizard.canStart).toBe(true);
    wizard.setTarget({ description: 'edited' });
    expect(wizard.stale).toBe(true);
    expect(wizard.canStart).toBe(false);
    await settle();
    await settle();
    expect(wizard.canStart).toBe(true);
    expect(await wizard.start()).toEqual({ job_id: 'cmb_1', target: 'merged' });
    const sent = startCombine.mock.calls[0]![0];
    expect(sent.expected_preview_sha).toBe('sha-1');
    expect(sent.target.description).toBe('edited');
  });

  it('preview_stale shows the served message and re-previews', async () => {
    const { wizard, previewCombine } = setup(undefined, async () => {
      throw new ApiError(409, '/u', {
        detail: { error: 'preview_stale', message: 'a source changed; preview again' },
      });
    });
    fill(wizard);
    await settle();
    await settle();
    const before = previewCombine.mock.calls.length;
    expect(await wizard.start()).toBeNull();
    expect(wizard.refusal).toMatchObject({
      code: 'preview_stale',
      message: 'a source changed; preview again',
    });
    await vi.advanceTimersByTimeAsync(0);
    expect(previewCombine.mock.calls.length).toBe(before + 1);
  });

  it('combine_invalid carries the served report; project_busy carries the served jobs', async () => {
    const report = {
      ok: false,
      errors: [
        {
          code: 'unmapped_class',
          severity: 'error',
          field: 'widgets-a',
          message: 'no mapping',
          detail: {},
          bypassable: false,
        },
      ],
      warnings: [],
      force_allowed: false,
    };
    const jobs = [{ kind: 'train', kind_label: 'Training', id: 'j1', label: 'a run' }];
    let n = 0;
    const { wizard } = setup(undefined, async () => {
      n += 1;
      throw new ApiError(n === 1 ? 422 : 409, '/u', {
        detail:
          n === 1
            ? {
                error: 'combine_invalid',
                message: 'the combine request has errors',
                report,
              }
            : { error: 'project_busy', message: 'a source is busy', jobs },
      });
    });
    fill(wizard);
    await settle();
    await settle();
    await wizard.start();
    expect(wizard.refusal?.code).toBe('combine_invalid');
    expect(wizard.refusal?.report?.errors[0]?.code).toBe('unmapped_class');
    await wizard.start();
    expect(wizard.refusal?.code).toBe('project_busy');
    expect(wizard.refusal?.jobs).toEqual(jobs);
    expect(wizard.refusal?.report).toBeNull();
  });
});

describe('mapping action labels', () => {
  it('are read once from the first source and fall back to the raw id', async () => {
    const { wizard, getDatasetFormatsFor } = setup();
    expect(wizard.actionLabel('map')).toBe('map');
    wizard.addSource('widgets-a');
    wizard.addSource('widgets-b');
    await vi.advanceTimersByTimeAsync(0);
    expect(getDatasetFormatsFor).toHaveBeenCalledTimes(1);
    expect(getDatasetFormatsFor.mock.calls[0]![0]).toEqual({
      prefix: '/curation/projects/widgets-a',
    });
    expect(wizard.actionLabel('map')).toBe('Map to class');
    expect(wizard.actionDescription('map')).toBe('Point at a created class');
    expect(wizard.actionLabel('skip')).toBe('skip');
    wizard.moveSource('widgets-b', -1);
    await vi.advanceTimersByTimeAsync(0);
    expect(getDatasetFormatsFor).toHaveBeenCalledTimes(2);
    expect(getDatasetFormatsFor.mock.calls[1]![0]).toEqual({
      prefix: '/curation/projects/widgets-b',
    });
  });

  it('a failed read keeps the raw ids', async () => {
    const wizard = createCombineWizard({
      getDatasetFormatsFor: async () => {
        throw new ApiError(404, '/u', { detail: 'Not Found' });
      },
      projectOf: () => ({ prefix: '/p' }),
    });
    wizard.addSource('widgets-a');
    await vi.advanceTimersByTimeAsync(0);
    expect(wizard.actionLabel('create')).toBe('create');
  });
});
