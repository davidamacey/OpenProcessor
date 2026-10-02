/**
 * The import wizard's controller: debounced preview of exactly what the
 * operator chose, Start tied to the preview's `import_key`, and every
 * served Start outcome (§7.12 item 1, W10.11).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api';
import { jobFixture, previewFixture } from '$lib/test/fixtures/datasetImport';
import type { DatasetPreview } from '$lib/types_import';
import { ImportWizard, PREVIEW_DEBOUNCE_MS } from './importWizardController.svelte';

function refusal(status: number, detail: Record<string, unknown>): ApiError {
  return new ApiError(status, '/x', { detail });
}

function wizard(preview: DatasetPreview = previewFixture()) {
  const deps = {
    previewDataset: vi.fn().mockResolvedValue(preview),
    startDatasetImport: vi.fn().mockResolvedValue(jobFixture()),
    resumeDatasetImport: vi.fn().mockResolvedValue(jobFixture({ status: 'running' })),
    uploadDatasetArchive: vi
      .fn()
      .mockResolvedValue({ upload_id: 'u', dataset_path: '/up/abc', bytes: 3, files: 2 }),
  };
  return { w: new ImportWizard(deps), deps };
}

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

async function settle(): Promise<void> {
  await vi.advanceTimersByTimeAsync(PREVIEW_DEBOUNCE_MS + 1);
}

describe('preview', () => {
  it('debounces and previews exactly the chosen source, mapping and touched options', async () => {
    const { w, deps } = wizard();
    w.setSource('/data/widgets');
    w.setFormat('yolo');
    w.setChoice('Widget', { action: 'map', class_id: 2 });
    w.setChoice('sprocket', { action: 'create', new_class_name: 'sprocket' });
    w.setChoice('bolt', { action: '' });
    w.setOption('processing', 'propose');
    expect(deps.previewDataset).not.toHaveBeenCalled();
    await settle();
    expect(deps.previewDataset).toHaveBeenCalledTimes(1);
    const body = deps.previewDataset.mock.calls[0]![0];
    expect(body).toEqual({
      source: { path: '/data/widgets', format: 'yolo' },
      mapping: [
        { dataset_class: 'Widget', action: 'map', class_id: 2 },
        { dataset_class: 'sprocket', action: 'create', new_class_name: 'sprocket' },
      ],
      accept_suggestions: false,
      options: { processing: 'propose' },
    });
    expect(body).not.toHaveProperty('project');
    expect(w.preview?.import_key).toBe('k'.repeat(64));
    expect(w.stale).toBe(false);
  });

  it('does not preview without a source path', async () => {
    const { w, deps } = wizard();
    w.setFormat('coco');
    await settle();
    expect(deps.previewDataset).not.toHaveBeenCalled();
  });

  it('shows a refused preview by its served message', async () => {
    const { w, deps } = wizard();
    deps.previewDataset.mockRejectedValue(
      refusal(422, {
        error: 'dataset_path_not_allowed',
        message: 'That path is outside the allowed roots.',
      }),
    );
    w.setSource('/etc');
    await settle();
    expect(w.previewError).toBe('That path is outside the allowed roots.');
    expect(w.preview).toBeNull();
  });

  it('"Use suggestion" copies the served suggestion', () => {
    const { w } = wizard();
    const rows = previewFixture().classes;
    w.useSuggestion(rows[0]!);
    w.useSuggestion(rows[1]!);
    expect(w.mappingEntries()).toEqual([
      { dataset_class: 'Widget', action: 'map', class_id: 2 },
      { dataset_class: 'sprocket', action: 'create', new_class_name: 'sprocket' },
    ]);
  });

  it('force is sent only while the served preview allows it', async () => {
    const { w } = wizard(previewFixture({ blocking: true, force_allowed: false }));
    w.setSource('/data/widgets');
    w.setOption('force', true);
    await settle();
    expect(w.requestBody().options).not.toHaveProperty('force');
    expect(w.canStart).toBe(false);
  });
});

describe('Start', () => {
  function mapped(): DatasetPreview {
    return previewFixture({
      classes: previewFixture().classes.map((c) => ({
        ...c,
        resolved: {
          dataset_class: c.dataset_class,
          kind: 'item',
          class_id: 2,
          class_name: 'widget',
        },
      })),
    });
  }

  it('is blocked while a served row with boxes has no resolved target', async () => {
    const { w } = wizard();
    w.setSource('/data/widgets');
    await settle();
    expect(w.unmappedRows.map((r) => r.dataset_class)).toEqual(['Widget', 'sprocket']);
    expect(w.canStart).toBe(false);
  });

  it('sends the preview import_key as expected_import_key and returns the job', async () => {
    const { w, deps } = wizard(mapped());
    w.setSource('/data/widgets');
    await settle();
    expect(w.canStart).toBe(true);
    const job = await w.start();
    expect(job?.import_id).toBe('imp_20260927T120000_1a2b3c4d');
    expect(deps.startDatasetImport.mock.calls[0]![0]).toMatchObject({
      source: { path: '/data/widgets', format: 'auto' },
      expected_import_key: 'k'.repeat(64),
    });
  });

  it('a blocking preview starts only with an allowed force', async () => {
    const { w, deps } = wizard({ ...mapped(), blocking: true, force_allowed: true });
    w.setSource('/data/widgets');
    await settle();
    expect(w.canStart).toBe(false);
    w.setOption('force', true);
    await settle();
    expect(w.canStart).toBe(true);
    await w.start();
    expect(deps.startDatasetImport.mock.calls[0]![0].options).toEqual({ force: true });
  });

  it('200 reused -> no navigation, the existing job kept for its link', async () => {
    const { w, deps } = wizard(mapped());
    deps.startDatasetImport.mockResolvedValue(
      jobFixture({ reused: true, status: 'completed' }),
    );
    w.setSource('/data/widgets');
    await settle();
    expect(await w.start()).toBeNull();
    expect(w.reusedJob?.status).toBe('completed');
  });

  it('409 import_resumable -> the refusal names the import; Resume resumes it', async () => {
    const { w, deps } = wizard(mapped());
    deps.startDatasetImport.mockRejectedValue(
      refusal(409, {
        error: 'import_resumable',
        message: 'This dataset has an interrupted import.',
        import_id: 'imp_old',
      }),
    );
    w.setSource('/data/widgets');
    await settle();
    await w.start();
    expect(w.refusal).toMatchObject({
      code: 'import_resumable',
      message: 'This dataset has an interrupted import.',
      importId: 'imp_old',
    });
    const job = await w.resume('imp_old');
    expect(deps.resumeDatasetImport).toHaveBeenCalledWith('imp_old');
    expect(job?.status).toBe('running');
  });

  it('409 dataset_changed -> re-runs the preview', async () => {
    const { w, deps } = wizard(mapped());
    deps.startDatasetImport.mockRejectedValue(
      refusal(409, { error: 'dataset_changed', message: 'The dataset changed.' }),
    );
    w.setSource('/data/widgets');
    await settle();
    expect(deps.previewDataset).toHaveBeenCalledTimes(1);
    await w.start();
    await vi.advanceTimersByTimeAsync(0);
    expect(deps.previewDataset).toHaveBeenCalledTimes(2);
    expect(w.refusal?.message).toBe('The dataset changed.');
  });

  it('422 class_mapping_incomplete -> the served unmapped rows are highlighted', async () => {
    const { w, deps } = wizard(mapped());
    deps.startDatasetImport.mockRejectedValue(
      refusal(422, {
        error: 'class_mapping_incomplete',
        message: 'One class still needs a mapping.',
        unmapped: ['sprocket'],
      }),
    );
    w.setSource('/data/widgets');
    await settle();
    await w.start();
    expect(w.unmappedRows.map((r) => r.dataset_class)).toEqual(['sprocket']);
  });

  it('422 import_blocked -> the served issues are kept for display', async () => {
    const { w, deps } = wizard(mapped());
    const issue = previewFixture().issues[0]!;
    deps.startDatasetImport.mockRejectedValue(
      refusal(422, { error: 'import_blocked', message: 'Blocked.', issues: [issue] }),
    );
    w.setSource('/data/widgets');
    await settle();
    await w.start();
    expect(w.refusal?.issues).toEqual([issue]);
  });
});

describe('prefill and upload', () => {
  it('pre-fills by dataset class name from a previous import, ignoring unknown names', async () => {
    const { w } = wizard();
    w.setSource('/data/widgets');
    await settle();
    w.prefillFromJob(
      jobFixture({
        mapping: [
          { dataset_class: 'Widget', kind: 'item', class_id: 7, class_name: 'widget' },
          { dataset_class: 'sprocket', kind: 'skip', class_id: null, class_name: null },
          { dataset_class: 'nope', kind: 'item', class_id: 1, class_name: 'x' },
        ],
      }),
    );
    expect(w.mappingEntries()).toEqual([
      { dataset_class: 'Widget', action: 'map', class_id: 7 },
      { dataset_class: 'sprocket', action: 'skip' },
    ]);
  });

  it('uploads, then previews the served dataset_path; refuses above the cap', async () => {
    const { w, deps } = wizard();
    await w.upload(new File(['abcd'], 'x.zip'), 3, 'Too large.');
    expect(w.uploadError).toBe('Too large.');
    expect(deps.uploadDatasetArchive).not.toHaveBeenCalled();
    await w.upload(new File(['abc'], 'x.zip'), 3, 'Too large.');
    expect(w.sourcePath).toBe('/up/abc');
    expect(deps.previewDataset.mock.calls.at(-1)![0].source.path).toBe('/up/abc');
  });
});
