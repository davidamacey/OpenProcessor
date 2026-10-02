/**
 * `/bakeoff`'s state: discovery lists, the operator's selection, the
 * active job's poll, and the run being viewed. Factory-function
 * convention (`ingestRunController`, `clusterController`); the API calls
 * are injectable so tests drive it without `fetch`.
 *
 * Thin by design (docs/design/bakeoff-v2-ui-plan-2026-09-25.md): nothing
 * here computes a metric, a class mapping, a rank or a winner.
 */
import {
  ApiError,
  bakeoffBaselineModels,
  bakeoffEvalDatasets,
  bakeoffMatrix,
  bakeoffProfiles,
  bakeoffResults,
  bakeoffRun,
  bakeoffRuns,
  bakeoffStatus,
  bakeoffTrainedModels,
} from '$lib/api';
import type {
  BakeoffComparison,
  BakeoffMatrix,
  BakeoffProfile,
  BakeoffRunAccepted,
  BakeoffRunRow,
  BakeoffStatus,
  BaselineModel,
  CustomModelRef,
  EvalDataset,
  TrainedModel,
  TrainedModelForDataset,
} from '$lib/types_bakeoff';
import { buildRunRequest, isTerminal, selectedModelRefs } from './view';

export interface BakeoffApi {
  profiles: typeof bakeoffProfiles;
  baselines: typeof bakeoffBaselineModels;
  datasets: typeof bakeoffEvalDatasets;
  trainedModels: typeof bakeoffTrainedModels;
  run: typeof bakeoffRun;
  status: typeof bakeoffStatus;
  runs: typeof bakeoffRuns;
  results: typeof bakeoffResults;
  matrix: typeof bakeoffMatrix;
}

export const defaultBakeoffApi: BakeoffApi = {
  profiles: bakeoffProfiles,
  baselines: bakeoffBaselineModels,
  datasets: bakeoffEvalDatasets,
  trainedModels: bakeoffTrainedModels,
  run: bakeoffRun,
  status: bakeoffStatus,
  runs: bakeoffRuns,
  results: bakeoffResults,
  matrix: bakeoffMatrix,
};

export const TRAINED_MODELS_LIMIT = 100;

/** The served error text: `detail` when the server sent one. */
export function errorText(e: unknown): string {
  if (e instanceof ApiError) return e.detail ?? e.message;
  return e instanceof Error ? e.message : String(e);
}

export function createBakeoffController(
  api: BakeoffApi = defaultBakeoffApi,
  opts: { pollMs?: number } = {},
) {
  const pollMs = opts.pollMs ?? 3000;

  const s = $state({
    profiles: [] as BakeoffProfile[],
    profile: '',
    profileDefaultError: null as string | null,
    datasets: [] as EvalDataset[],
    trained: [] as TrainedModel[],
    baselines: [] as BaselineModel[],
    /** dataset id → run id → that run's facts for the dataset. */
    facts: {} as Record<string, Record<string, TrainedModelForDataset | null>>,
    selectedDatasets: [] as string[],
    selectedRuns: [] as string[],
    selectedBaselines: [] as string[],
    customRefs: [] as CustomModelRef[],
    runs: [] as BakeoffRunRow[],
    /** Discovery / list failures, one line each. */
    loadErrors: [] as string[],
    /** The last `POST /run` failure, verbatim. */
    runError: null as string | null,
    submitting: false,
    activeJob: null as string | null,
    activeStatus: null as BakeoffStatus | null,
    accepted: null as BakeoffRunAccepted | null,
    viewJob: null as string | null,
    matrix: null as BakeoffMatrix | null,
    matrixError: null as string | null,
    viewDataset: null as string | null,
    comparison: null as BakeoffComparison | null,
    comparisonLegacy: false,
    comparisonError: null as string | null,
    loadingResults: false,
  });

  let timer: ReturnType<typeof setInterval> | undefined;
  let destroyed = false;

  function noteLoadError(what: string, e: unknown) {
    s.loadErrors = [...s.loadErrors, `${what}: ${errorText(e)}`];
  }

  async function loadProfiles() {
    try {
      const r = await api.profiles();
      s.profiles = r.profiles ?? [];
      s.profileDefaultError = r.default_error ?? null;
      if (!s.profile && r.default_profile) s.profile = r.default_profile;
    } catch (e) {
      noteLoadError('profiles', e);
    }
  }

  async function loadBaselines() {
    try {
      const r = await api.baselines(s.profile || undefined);
      s.baselines = r.baselines ?? [];
      // eslint-disable-next-line svelte/prefer-svelte-reactivity -- local, synchronous lookup set consumed within this function only, never stored in reactive state
      const names = new Set(s.baselines.map((b) => b.name));
      s.selectedBaselines = s.selectedBaselines.filter((n) => names.has(n));
    } catch (e) {
      s.baselines = [];
      noteLoadError('baseline models', e);
    }
  }

  async function loadDatasets() {
    try {
      const r = await api.datasets();
      s.datasets = r.datasets ?? [];
      if (s.selectedDatasets.length === 0) {
        const current = s.datasets.find((d) => d.source === 'export' && d.is_current);
        if (current) await setDatasetSelected(current.id, true);
      }
    } catch (e) {
      noteLoadError('datasets', e);
    }
  }

  async function loadTrained() {
    try {
      const r = await api.trainedModels({ limit: TRAINED_MODELS_LIMIT });
      s.trained = r.models ?? [];
    } catch (e) {
      noteLoadError('trained runs', e);
    }
  }

  async function loadFacts(datasetId: string) {
    if (s.facts[datasetId]) return;
    try {
      const r = await api.trainedModels({ datasetId, limit: TRAINED_MODELS_LIMIT });
      const byRun: Record<string, TrainedModelForDataset | null> = {};
      for (const m of r.models ?? []) byRun[m.run_id] = m.for_dataset ?? null;
      s.facts = { ...s.facts, [datasetId]: byRun };
    } catch (e) {
      noteLoadError(`run facts for ${datasetId}`, e);
    }
  }

  async function refreshRuns() {
    try {
      s.runs = (await api.runs()).runs ?? [];
      if (!s.activeJob) {
        const live = s.runs.find((r) => r.state === 'queued' || r.state === 'running');
        if (live) {
          s.activeJob = live.job_id;
          startPolling();
        }
      }
    } catch (e) {
      noteLoadError('previous runs', e);
    }
  }

  async function init() {
    await Promise.all([
      loadProfiles().then(loadBaselines),
      loadDatasets(),
      loadTrained(),
      refreshRuns(),
    ]);
  }

  async function setProfile(name: string) {
    s.profile = name;
    await loadBaselines();
  }

  function toggle(list: string[], id: string, on: boolean): string[] {
    const has = list.includes(id);
    if (on && !has) return [...list, id];
    if (!on && has) return list.filter((x) => x !== id);
    return list;
  }

  async function setDatasetSelected(id: string, on: boolean) {
    s.selectedDatasets = toggle(s.selectedDatasets, id, on);
    if (on) await loadFacts(id);
  }

  function setRunSelected(runId: string, on: boolean) {
    s.selectedRuns = toggle(s.selectedRuns, runId, on);
  }

  function setBaselineSelected(name: string, on: boolean) {
    s.selectedBaselines = toggle(s.selectedBaselines, name, on);
  }

  function addCustom(ref: CustomModelRef) {
    s.customRefs = [...s.customRefs, ref];
  }

  function removeCustom(index: number) {
    s.customRefs = s.customRefs.filter((_, i) => i !== index);
  }

  function selection() {
    return {
      datasetIds: s.selectedDatasets,
      runIds: s.selectedRuns,
      baselineNames: s.selectedBaselines,
      customRefs: s.customRefs,
      profile: s.profile,
    };
  }

  function modelCount(): number {
    return selectedModelRefs(selection()).length;
  }

  async function submit(): Promise<boolean> {
    s.runError = null;
    s.submitting = true;
    try {
      const accepted = await api.run(buildRunRequest(selection()));
      s.accepted = accepted;
      s.activeJob = accepted.job_id;
      s.activeStatus = null;
      startPolling();
      void refreshRuns();
      return true;
    } catch (e) {
      s.runError = errorText(e);
      return false;
    } finally {
      s.submitting = false;
    }
  }

  async function pollOnce() {
    const job = s.activeJob;
    if (!job) return;
    try {
      const st = await api.status(job);
      if (s.activeJob !== job) return;
      s.activeStatus = st;
      if (isTerminal(st.state)) {
        stopPolling();
        await refreshRuns();
        if (st.state === 'done') await viewRun(job);
      }
    } catch {
      // A transient failure keeps polling; the status panel keeps the last value.
    }
  }

  function startPolling() {
    stopPolling();
    if (destroyed) return;
    void pollOnce();
    timer = setInterval(() => void pollOnce(), pollMs);
  }

  function stopPolling() {
    if (timer) clearInterval(timer);
    timer = undefined;
  }

  async function loadComparison(jobId: string, datasetId: string | undefined) {
    s.comparison = null;
    s.comparisonLegacy = false;
    s.comparisonError = null;
    try {
      const c = await api.results(jobId, datasetId);
      if (s.viewJob !== jobId) return;
      s.comparison = c;
      if (!s.viewDataset) s.viewDataset = c.dataset?.id ?? null;
    } catch (e) {
      if (s.viewJob !== jobId) return;
      if (e instanceof ApiError && e.status === 409) s.comparisonLegacy = true;
      else s.comparisonError = errorText(e);
    }
  }

  async function viewRun(jobId: string) {
    s.viewJob = jobId;
    s.matrix = null;
    s.matrixError = null;
    s.viewDataset = null;
    s.loadingResults = true;
    try {
      try {
        const m = await api.matrix(jobId);
        if (s.viewJob !== jobId) return;
        s.matrix = m;
        s.viewDataset = m.datasets[0]?.id ?? null;
      } catch (e) {
        if (s.viewJob !== jobId) return;
        if (e instanceof ApiError && e.status === 409) s.comparisonLegacy = true;
        else s.matrixError = errorText(e);
      }
      await loadComparison(jobId, s.viewDataset ?? undefined);
    } finally {
      if (s.viewJob === jobId) s.loadingResults = false;
    }
  }

  async function setViewDataset(datasetId: string) {
    if (!s.viewJob) return;
    s.viewDataset = datasetId;
    await loadComparison(s.viewJob, datasetId);
  }

  function destroy() {
    destroyed = true;
    stopPolling();
  }

  return {
    state: s,
    init,
    setProfile,
    setDatasetSelected,
    setRunSelected,
    setBaselineSelected,
    addCustom,
    removeCustom,
    modelCount,
    submit,
    pollOnce,
    viewRun,
    setViewDataset,
    refreshRuns,
    destroy,
  };
}

export type BakeoffController = ReturnType<typeof createBakeoffController>;
