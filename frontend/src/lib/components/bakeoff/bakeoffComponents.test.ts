/**
 * Mount tests for the /bakeoff v2 components: tied winners all bold,
 * not-covered classes and unmapped model classes rendered, the overlap
 * warning only for a real overlap, the 409 legacy note, and served
 * failure details.
 */
import { afterEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import BakeoffMatrixTable from './BakeoffMatrixTable.svelte';
import ComparisonView from './ComparisonView.svelte';
import ModelPicker from './ModelPicker.svelte';
import RunStatusPanel from './RunStatusPanel.svelte';
import DatasetPicker from './DatasetPicker.svelte';
import {
  ACCEPTED,
  COMPARISON,
  DS_CURRENT,
  DS_OLDER,
  EVAL_DATASETS,
  MATRIX,
  status,
  trainedModel,
} from '$lib/test/fixtures/bakeoff';
import { LEGACY_RESULTS_MESSAGE } from '$lib/bakeoff/view';

const mounted: { instance: ReturnType<typeof mount>; target: HTMLElement }[] = [];
function render<P extends Record<string, unknown>>(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  Component: any,
  props: P,
): HTMLElement {
  const target = document.createElement('div');
  document.body.appendChild(target);
  const instance = mount(Component, { target, props });
  flushSync();
  mounted.push({ instance, target });
  return target;
}
afterEach(() => {
  for (const { instance, target } of mounted.splice(0)) {
    unmount(instance);
    target.remove();
  }
});

const name = (id: string) => id;

describe('BakeoffMatrixTable', () => {
  it('bolds every tied winner for the served rank metric and nothing else', () => {
    const t = render(BakeoffMatrixTable, { matrix: MATRIX, datasetName: name });
    const cells = [...t.querySelectorAll<HTMLElement>('[data-testid="matrix-cell"]')];
    const bold = cells
      .filter((c) => c.dataset.best === 'true')
      .map((c) => c.closest('tr')!.dataset.model);
    expect(bold).toEqual(['run:run-a', 'run:run-b']);
    for (const c of cells) {
      expect(c.classList.contains('font-bold')).toBe(c.dataset.best === 'true');
    }
  });

  it('follows the selected metric and renders a missing cell as "—"', () => {
    const t = render(BakeoffMatrixTable, { matrix: MATRIX, datasetName: name });
    const select = t.querySelector<HTMLSelectElement>('[data-testid="matrix-metric"]')!;
    select.value = 'precision';
    select.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    const rows = [...t.querySelectorAll<HTMLElement>('tbody tr')];
    const byModel = Object.fromEntries(
      rows.map((r) => [r.dataset.model, r.querySelector('[data-testid="matrix-cell"]')!]),
    );
    expect(byModel['run:run-a'].getAttribute('data-best')).toBe('true');
    expect(byModel['run:run-b'].getAttribute('data-best')).toBe('false');
    expect(byModel['baseline:ref'].textContent).toContain('—');
  });
});

describe('ComparisonView', () => {
  it('shows a not-covered cell and the unmapped model classes', () => {
    const t = render(ComparisonView, {
      comparison: COMPARISON,
      legacy: false,
      error: null,
      loading: false,
    });
    const boltRow = t.querySelector<HTMLElement>('[data-class-id="1"]')!;
    const cells = boltRow.querySelectorAll('td');
    // class, objects, full model, subset model
    expect(cells[2].textContent).toContain('50.0');
    expect(cells[3].textContent).toContain('not covered');
    expect(t.querySelectorAll('[data-testid="not-covered"]').length).toBe(1);
    expect(t.querySelector('[data-testid="unmapped-classes"]')!.textContent).toContain(
      'sprocket (7 predictions)',
    );
  });

  it('renders a null rank as "—" and flags a real train/test overlap', () => {
    const t = render(ComparisonView, {
      comparison: COMPARISON,
      legacy: false,
      error: null,
      loading: false,
    });
    const sub = t.querySelector<HTMLElement>('[data-model="run:run-b"]')!;
    expect(sub.querySelector('td')!.textContent!.trim()).toBe('—');
    expect(t.querySelectorAll('[data-testid="row-overlap-warning"]').length).toBe(1);
  });

  it('shows the legacy note for a pre-v2 result instead of an error', () => {
    const t = render(ComparisonView, {
      comparison: null,
      legacy: true,
      error: null,
      loading: false,
    });
    expect(t.querySelector('[data-testid="comparison-legacy"]')!.textContent).toContain(
      LEGACY_RESULTS_MESSAGE,
    );
    expect(t.querySelector('[data-testid="comparison-error"]')).toBeNull();
  });

  it('shows a served error verbatim', () => {
    const t = render(ComparisonView, {
      comparison: null,
      legacy: false,
      error: 'no comparison for dataset export:x',
      loading: false,
    });
    expect(t.querySelector('[data-testid="comparison-error"]')!.textContent).toContain(
      'no comparison for dataset export:x',
    );
  });
});

describe('ModelPicker', () => {
  const base = {
    baselines: [],
    customRefs: [],
    selectedRuns: [],
    selectedBaselines: [],
    selectedDatasets: [DS_CURRENT, DS_OLDER],
    datasetName: name,
    onToggleRun: () => {},
    onToggleBaseline: () => {},
    onAddCustom: () => {},
    onRemoveCustom: () => {},
  };
  const fact = (n: number | null) => ({
    dataset_id: DS_CURRENT,
    same_export: true,
    same_frozen_test: true,
    n_classes_mapped: 2,
    train_test_overlap: n === null ? null : { n_images: n, fraction: n / 10 },
  });

  it('warns about overlap only where the served overlap is non-null and > 0', () => {
    const t = render(ModelPicker, {
      ...base,
      trained: [trainedModel({ run_id: 'a' }), trainedModel({ run_id: 'b' })],
      facts: {
        [DS_CURRENT]: { a: fact(3), b: fact(0) },
        [DS_OLDER]: { a: fact(null), b: fact(null) },
      },
    });
    const warnings = [...t.querySelectorAll('[data-testid="overlap-warning"]')];
    expect(warnings).toHaveLength(1);
    expect(warnings[0].closest('[data-run-id]')!.getAttribute('data-run-id')).toBe('a');
    expect(warnings[0].textContent).toContain('3 test images');
  });

  it('labels the trainer number as the trainer\'s and a missing one as "—"', () => {
    const t = render(ModelPicker, {
      ...base,
      selectedDatasets: [],
      trained: [trainedModel({ trainer_map50: null, trainer_map50_split: null })],
      facts: {},
    });
    expect(t.textContent).toContain('trainer mAP50 —');
  });
});

describe('RunStatusPanel', () => {
  it('lists every failed piece and the job error, and the enqueue mapping', () => {
    const t = render(RunStatusPanel, {
      jobId: 'job-1',
      status: status({
        state: 'error',
        error: 'evaluator crashed',
        failed: [
          { stage: 'throughput', dataset: null, model: null, error: 'no GPU' },
          {
            stage: null,
            dataset: DS_CURRENT,
            model: 'run:run-b',
            error: 'weights missing',
          },
        ],
      }),
      accepted: ACCEPTED,
      datasetName: name,
    });
    expect(t.querySelector('[data-testid="run-job-error"]')!.textContent).toContain(
      'evaluator crashed',
    );
    const items = [...t.querySelectorAll('[data-testid="run-failures"] li')].map(
      (l) => l.textContent,
    );
    expect(items).toEqual([
      'throughput — no GPU',
      `${DS_CURRENT} · run:run-b — weights missing`,
    ]);
    expect(t.querySelector('[data-testid="enqueue-summary"]')!.textContent).toContain(
      'not covered: bolt',
    );
  });
});

describe('DatasetPicker', () => {
  it('lists export test splits first with the current one flagged, then external groups', () => {
    const t = render(DatasetPicker, {
      datasets: [EVAL_DATASETS[2], EVAL_DATASETS[1], EVAL_DATASETS[0]],
      selected: [DS_CURRENT],
      onToggle: () => {},
    });
    const ids = [...t.querySelectorAll<HTMLElement>('[data-testid="dataset-row"]')].map(
      (r) => r.dataset.datasetId,
    );
    expect(ids).toEqual([DS_OLDER, DS_CURRENT, EVAL_DATASETS[2].id]);
    const current = t.querySelectorAll('[data-testid="dataset-current"]');
    expect(current).toHaveLength(1);
    expect(current[0].closest('[data-dataset-id]')!.getAttribute('data-dataset-id')).toBe(
      DS_CURRENT,
    );
    expect(t.textContent).toContain('External · curated');
  });
});
