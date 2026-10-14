/**
 * /train's operator actions: preflight, start (single run or campaign),
 * cancel, reproduce, and the promote modal wiring. Status reads and the
 * past-runs refresh go through the runs controller.
 */
import {
  ApiError,
  apiErrorText,
  cancelTrainCampaign,
  cancelTrainJob,
  getTrainManifest,
  getTrainStatus,
  trainPreflight,
  trainStart,
  trainStartCampaign,
} from '$lib/api';
import { defaultTritonName, findActivePromote } from '$lib/promote';
import { toastStore } from '$stores/toast.svelte';
import type {
  PreflightReport,
  PromoteJobStatus,
  TrainCampaignSpec,
  TrainJobSpec,
  TrainJobStatus,
} from '$lib/types_train';
import type { TrainRunsController } from './trainRunsController.svelte';

export function createTrainActionsController(runs: TrainRunsController) {
  // ---- Preflight -------------------------------------------------------
  let preflight = $state<PreflightReport | null>(null);
  let preflighting = $state<boolean>(false);
  let preflightAbort: AbortController | null = null;

  async function runPreflight(spec: TrainJobSpec): Promise<void> {
    preflightAbort?.abort();
    preflightAbort = new AbortController();
    preflighting = true;
    try {
      preflight = await trainPreflight(spec, preflightAbort.signal);
    } catch (e) {
      if ((e as Error).name === 'AbortError') return;
      preflight = null;
    } finally {
      preflighting = false;
    }
  }

  // ---- Submit handlers -------------------------------------------------
  let starting = $state<boolean>(false);
  let cancellingRun = $state<boolean>(false);
  let cancellingCampaign = $state<boolean>(false);

  function maybeRenderPreflight(err: unknown): void {
    if (err instanceof ApiError && err.body && typeof err.body === 'object') {
      const body = err.body as {
        detail?: {
          preflight?: PreflightReport;
          message?: string;
          // df01309: an unknown `augmentation.preset` 422s with
          // `{message, field: 'augmentation.preset', valid_presets}`
          // instead of/alongside a preflight report.
          field?: string;
          valid_presets?: string[];
        };
      };
      const pf = body.detail?.preflight;
      if (pf) preflight = pf;
      let msg = body.detail?.message ?? err.message;
      if (body.detail?.field === 'augmentation.preset' && body.detail.valid_presets) {
        msg += ` (valid presets: ${body.detail.valid_presets.join(', ')})`;
      }
      toastStore.error(`Start failed: ${msg}`);
    } else {
      toastStore.error(`Start failed: ${apiErrorText(err)}`);
    }
  }

  async function startSingle(spec: TrainJobSpec, force: boolean): Promise<void> {
    starting = true;
    try {
      const res = await trainStart(spec, force);
      preflight = res.preflight;
      runs.trackedJobId = res.job_id;
      toastStore.success(`Submitted ${res.job_id}`);
      await runs.pullStatus();
    } catch (e) {
      maybeRenderPreflight(e);
    } finally {
      starting = false;
    }
  }

  async function startCampaign(spec: TrainCampaignSpec, force: boolean): Promise<void> {
    starting = true;
    try {
      const res = await trainStartCampaign(spec, force);
      if (res.preflight) preflight = res.preflight;
      runs.trackedJobId = res.job_ids[0] ?? null;
      toastStore.success(
        `Submitted campaign ${res.campaign_id} (${res.job_ids.length} runs)`,
      );
      await runs.pullStatus();
    } catch (e) {
      maybeRenderPreflight(e);
    } finally {
      starting = false;
    }
  }

  async function cancelActive(jobId: string): Promise<void> {
    cancellingRun = true;
    try {
      await cancelTrainJob(jobId);
      toastStore.info(`Cancel sentinel dropped for ${jobId}`);
      await runs.pullStatus();
    } catch (e) {
      toastStore.error(`Cancel failed: ${apiErrorText(e)}`);
    } finally {
      cancellingRun = false;
    }
  }

  async function cancelActiveCampaign(campaignId: string): Promise<void> {
    cancellingCampaign = true;
    try {
      const res = await cancelTrainCampaign(campaignId);
      const n = typeof res.cancelled === 'number' ? res.cancelled : Number(res.cancelled);
      toastStore.info(`Cancelled ${n} run(s) in ${campaignId}`);
      await runs.pullStatus();
      await runs.refreshRuns();
    } catch (e) {
      toastStore.error(`Cancel failed: ${apiErrorText(e)}`);
    } finally {
      cancellingCampaign = false;
    }
  }

  // ---- Promote modal ---------------------------------------------------
  let promoteOpen = $state<boolean>(false);
  let promoteJobId = $state<string | null>(null);
  let promoteDefaultName = $state<string>('');

  /** A promote already running for `promoteJobId`, followed by the modal. */
  let promoteAttach = $state<PromoteJobStatus | null>(null);

  async function openPromote(r: TrainJobStatus): Promise<void> {
    promoteAttach = null;
    try {
      // The runs list does not carry `promote`; the per-run status does.
      const st = await getTrainStatus(r.job_id);
      if (st?.promote) promoteAttach = st.promote;
    } catch {
      // Best effort: the form still works, and a duplicate promote is
      // answered by the server with the already-active job.
    }
    promoteJobId = r.job_id;
    // the run's own id, Triton-safe. No version suffix — nothing
    // client-side knows this deployment's versioning.
    promoteDefaultName = defaultTritonName(r.job_id);
    promoteOpen = true;
  }

  /** Reload resume: re-open the modal on a promote still running. */
  async function resumeActivePromote(): Promise<void> {
    await runs.refreshRuns();
    const hit = await findActivePromote(
      runs.runs
        .filter((r) => r.state === 'finished' || r.state === 'exporting')
        .slice(0, 3)
        .map((r) => r.job_id),
      getTrainStatus,
    );
    if (hit) {
      promoteJobId = hit.runJobId;
      promoteDefaultName = hit.promote.triton_name;
      promoteAttach = hit.promote;
      promoteOpen = true;
    }
  }

  // ---- Reproduce-this-run (Phase 6, design §15.4) ---------------------
  // Fetches <job_id>.manifest.json, builds a fresh TrainJobSpec from the
  // saved spec + lineage, and POSTs {API_PREFIX}/train/start. Lets the user repeat
  // a known-good run without re-typing every knob.
  let reproducingId = $state<string | null>(null);

  async function reproduceRun(r: TrainJobStatus): Promise<void> {
    const ok = window.confirm(
      `Reproduce ${r.job_id}? Submits a new training job with the same spec.`,
    );
    if (!ok) return;
    reproducingId = r.job_id;
    try {
      const manifest = await getTrainManifest(r.job_id);
      const spec = (manifest.spec ?? {}) as Record<string, unknown>;
      const lineage = manifest.lineage ?? {};
      const body: Partial<TrainJobSpec> = {
        dataset_export_dir: lineage.export_dir as string,
        include_classes: lineage.include_classes ?? null,
        single_cls: lineage.single_cls ?? false,
        cuda_visible_devices:
          (spec.cuda_visible_devices as string | undefined) ?? undefined,
        model_family: (spec.model_family as TrainJobSpec['model_family']) ?? 'yolo26',
        model_size: (spec.model_size as TrainJobSpec['model_size']) ?? 'm',
        profile: (spec.profile as TrainJobSpec['profile']) ?? 'medium',
        hyperparameters:
          (spec.hyperparameters as Record<string, unknown> | undefined) ?? {},
        augmentation: (spec.augmentation as TrainJobSpec['augmentation']) ?? null,
      };
      const res = await trainStart(body as TrainJobSpec);
      toastStore.success(`Reproduced as ${res.job_id}`);
      await runs.refreshRuns();
    } catch (e) {
      toastStore.error(`Reproduce failed: ${apiErrorText(e)}`);
    } finally {
      reproducingId = null;
    }
  }

  function dispose(): void {
    preflightAbort?.abort();
  }

  return {
    get preflight() {
      return preflight;
    },
    get preflighting() {
      return preflighting;
    },
    get starting() {
      return starting;
    },
    get cancellingRun() {
      return cancellingRun;
    },
    get cancellingCampaign() {
      return cancellingCampaign;
    },
    get promoteOpen() {
      return promoteOpen;
    },
    set promoteOpen(v: boolean) {
      promoteOpen = v;
    },
    get promoteJobId() {
      return promoteJobId;
    },
    get promoteDefaultName() {
      return promoteDefaultName;
    },
    get promoteAttach() {
      return promoteAttach;
    },
    get reproducingId() {
      return reproducingId;
    },
    runPreflight,
    startSingle,
    startCampaign,
    cancelActive,
    cancelActiveCampaign,
    openPromote,
    resumeActivePromote,
    reproduceRun,
    dispose,
  };
}

export type TrainActionsController = ReturnType<typeof createTrainActionsController>;
