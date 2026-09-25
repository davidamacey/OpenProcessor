/**
 * Mount-based behavior test for RunResults (finished-run results view,
 * /train). Uses the real served fixtures from the live
 * 2026-09-24T23-47-55_yolo26n train-smoke run
 * (`$lib/test/fixtures/trainRun.ts`, mirroring
 * `artifacts_local/cw-live/train-smoke/f_status.json` /
 * `f_manifest.json` verbatim).
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { mount, unmount, flushSync, tick } from 'svelte';
import type { TrainJobStatus } from '$lib/types_train';
import { trainManifestFixture, trainStatusFixture } from '$lib/test/fixtures/trainRun';

const getTrainManifestMock = vi.fn();

vi.mock('$lib/api', () => ({
  getTrainManifest: (...args: unknown[]) => getTrainManifestMock(...args),
}));

const { default: RunResults } = await import('./RunResults.svelte');

let target: HTMLDivElement;
let instance: unknown;

function renderRunResults(status: TrainJobStatus, startOpen = true) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(RunResults, { target, props: { status, startOpen } });
  flushSync();
  return target;
}

afterEach(() => {
  if (instance) {
    unmount(instance);
    instance = undefined;
  }
  target?.remove();
  getTrainManifestMock.mockReset();
});

describe('RunResults — val vs test labelling', () => {
  it('labels the overall eval figures "validation (last epoch)" and per-class "test split" for the real fixture (no eval.split served)', async () => {
    getTrainManifestMock.mockResolvedValue(trainManifestFixture);
    const el = renderRunResults(trainStatusFixture);
    await tick();
    flushSync();
    expect(el.textContent).toContain('validation (last epoch)');
    expect(el.textContent).toContain('test split (frozen holdout)');
  });

  it('labels best/last metric as validation, never as test', () => {
    getTrainManifestMock.mockResolvedValue(trainManifestFixture);
    const el = renderRunResults(trainStatusFixture);
    expect(el.textContent).toContain('Metrics — validation');
  });
});

describe('RunResults — per-class table', () => {
  it('renders one row per served per_class entry with name/precision/recall/f1/ap50/support', async () => {
    getTrainManifestMock.mockResolvedValue(trainManifestFixture);
    const el = renderRunResults(trainStatusFixture);
    await tick();
    flushSync();
    const rows = el.querySelectorAll('table tbody tr');
    // Two tables render (per-class eval + class remap) — the per-class
    // table is the first and has exactly 5 rows for the fixture's 5 classes.
    const perClassTable = el.querySelectorAll('table')[0];
    const perClassRows = perClassTable.querySelectorAll('tbody tr');
    expect(perClassRows.length).toBe(5);
    expect(perClassTable.textContent).toContain('miata');
    expect(perClassTable.textContent).toContain('0.755'); // miata ap50
    expect(perClassTable.textContent).toContain('5'); // support
    expect(rows.length).toBeGreaterThanOrEqual(5);
  });
});

describe('RunResults — null rendering', () => {
  it('renders "—" for checkpoint_sha256 when neither status nor manifest serve it', async () => {
    getTrainManifestMock.mockResolvedValue({
      ...trainManifestFixture,
      results: { ...trainManifestFixture.results, checkpoint_sha256: null },
    });
    const status: TrainJobStatus = {
      ...trainStatusFixture,
      checkpoint_sha256: undefined,
    };
    const el = renderRunResults(status);
    await tick();
    flushSync();
    const shaDt = Array.from(el.querySelectorAll('dt')).find(
      (d) => d.textContent === 'Checkpoint SHA-256',
    );
    expect(shaDt?.nextElementSibling?.textContent?.trim()).toBe('—');
  });

  it('renders the real fixture checkpoint_sha256 from the manifest, not a false —', async () => {
    getTrainManifestMock.mockResolvedValue(trainManifestFixture);
    const el = renderRunResults(trainStatusFixture);
    await tick();
    flushSync();
    expect(el.textContent).toContain(
      'cc5ffb75e020b54d87df6f534de2a7a74eaa02519c69658a9504e2fe42d15e81',
    );
  });

  it('renders "—" for a null lineage field (dataset_sha is null on the fixture)', async () => {
    getTrainManifestMock.mockResolvedValue(trainManifestFixture);
    const el = renderRunResults(trainStatusFixture);
    await tick();
    flushSync();
    const dsDt = Array.from(el.querySelectorAll('dt')).find(
      (d) => d.textContent === 'dataset_sha',
    );
    expect(dsDt?.nextElementSibling?.textContent?.trim()).toBe('—');
  });
});

describe('RunResults — mlflow', () => {
  it('shows "pending" when mlflow_run_id is null and the run is still active', () => {
    getTrainManifestMock.mockResolvedValue(trainManifestFixture);
    const status: TrainJobStatus = {
      ...trainStatusFixture,
      state: 'running',
      mlflow_run_id: null,
      mlflow_run_url: null,
    };
    const el = renderRunResults(status);
    expect(el.textContent).toContain('pending');
  });

  it('shows "—" when mlflow_run_id is null and the run is terminal (not "pending")', () => {
    getTrainManifestMock.mockResolvedValue(trainManifestFixture);
    const status: TrainJobStatus = {
      ...trainStatusFixture,
      state: 'failed',
      mlflow_run_id: null,
      mlflow_run_url: null,
      error: 'trainer crashed',
    };
    const el = renderRunResults(status);
    const dt = Array.from(el.querySelectorAll('dt')).find(
      (d) => d.textContent === 'MLflow run',
    );
    expect(dt?.nextElementSibling?.textContent?.trim()).toBe('—');
  });

  it('renders mlflow_run_url as a clickable link when non-null (backend contract: null unless publicly reachable)', () => {
    getTrainManifestMock.mockResolvedValue(trainManifestFixture);
    const el = renderRunResults(trainStatusFixture);
    const link = el.querySelector('a[href="' + trainStatusFixture.mlflow_run_url + '"]');
    expect(link).not.toBeNull();
    expect(link?.getAttribute('target')).toBe('_blank');
  });

  it('renders the mlflow_run_id as plain text (no link) when mlflow_run_url is null', () => {
    getTrainManifestMock.mockResolvedValue(trainManifestFixture);
    const status: TrainJobStatus = {
      ...trainStatusFixture,
      mlflow_run_url: null,
    };
    const el = renderRunResults(status);
    expect(el.querySelector('a')).toBeNull();
    expect(el.textContent).toContain(trainStatusFixture.mlflow_run_id);
  });
});

describe('RunResults — failed run', () => {
  it('shows the served error banner for a failed run', () => {
    getTrainManifestMock.mockResolvedValue(trainManifestFixture);
    const status: TrainJobStatus = {
      ...trainStatusFixture,
      state: 'failed',
      error: 'CUDA out of memory',
      eval: null,
      best_metric: null,
      last_metric: null,
    };
    const el = renderRunResults(status);
    expect(el.textContent).toContain('CUDA out of memory');
  });
});

describe('RunResults — confusion matrix', () => {
  it('renders confusion_matrix_path as text only, never as an <img>, when no url is served', async () => {
    getTrainManifestMock.mockResolvedValue(trainManifestFixture);
    const el = renderRunResults(trainStatusFixture);
    await tick();
    flushSync();
    expect(el.textContent).toContain(trainStatusFixture.eval?.confusion_matrix_path);
    expect(el.querySelector('img')).toBeNull();
  });

  it('renders an <img src> from confusion_matrix_url when the backend serves one', () => {
    getTrainManifestMock.mockResolvedValue(trainManifestFixture);
    const status: TrainJobStatus = {
      ...trainStatusFixture,
      eval: {
        ...trainStatusFixture.eval,
        confusion_matrix_url: '/curation/train/artifacts/job1/confusion_matrix.png',
      },
    };
    const el = renderRunResults(status);
    const img = el.querySelector('img');
    expect(img?.getAttribute('src')).toBe(
      '/curation/train/artifacts/job1/confusion_matrix.png',
    );
  });
});

describe('RunResults — lazy manifest load', () => {
  it('does not call getTrainManifest before the section is opened', () => {
    getTrainManifestMock.mockResolvedValue(trainManifestFixture);
    renderRunResults(trainStatusFixture, false);
    expect(getTrainManifestMock).not.toHaveBeenCalled();
  });

  it('calls getTrainManifest exactly once when opened', async () => {
    getTrainManifestMock.mockResolvedValue(trainManifestFixture);
    const el = renderRunResults(trainStatusFixture, false);
    const toggle = el.querySelector('button');
    toggle?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    flushSync();
    await tick();
    expect(getTrainManifestMock).toHaveBeenCalledTimes(1);
    expect(getTrainManifestMock).toHaveBeenCalledWith(trainStatusFixture.job_id);
  });

  it('renders the class remap table (original->new ids with names) from the manifest', async () => {
    getTrainManifestMock.mockResolvedValue(trainManifestFixture);
    const el = renderRunResults(trainStatusFixture);
    await tick();
    flushSync();
    expect(el.textContent).toContain('miata');
    // new id 0 -> original id 38 (miata)
    const tables = el.querySelectorAll('table');
    const remapTable = tables[tables.length - 1];
    expect(remapTable.textContent).toContain('38');
  });
});
