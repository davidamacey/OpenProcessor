<script lang="ts">
  /**
   * /train — model training cockpit (Phase 2 of the legacy_train_pipeline
   * design doc).
   *
   * Owns:
   *   - Reference data (profiles, presets) — fetched once on mount.
   *   - Form preflight: debounced calls so the inline check panel
   *     refreshes as the user adjusts the form.
   *   - The active run: polls `/curation/train/status` every 5s and the log
   *     tail every 2s while state ∈ {starting, running, exporting, queued}.
   *     Polling stops on terminal states.
   *   - Past-runs table with the standard `infiniteScroll` action.
   *   - Promote modal wiring.
   *
   * Sub-components own their own UI; this page is the data layer.
   */
  import { goto } from '$app/navigation';
  import { onDestroy, onMount } from 'svelte';
  import {
    ApiError,
    cancelTrainCampaign,
    cancelTrainJob,
    exportStatus,
    getTrainingCandidates,
    getTrainManifest,
    getTrainPresets,
    getTrainProfiles,
    getTrainRuns,
    getTrainStatus,
    tailTrainLog,
    trainPreflight,
    trainStart,
    trainStartCampaign,
    type PlateBrowseItem,
    type TrainingCohortMode,
  } from '$lib/api';
  import { infiniteScroll } from '$lib/actions/infiniteScroll';
  import CampaignCard from '$components/CampaignCard.svelte';
  import LogTail from '$components/LogTail.svelte';
  import PlateCard from '$components/PlateCard.svelte';
  import PromoteModal from '$components/PromoteModal.svelte';
  import TrainForm from '$components/TrainForm.svelte';
  import TrainProgress from '$components/TrainProgress.svelte';
  import { classesStore } from '$stores/classes.svelte';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { toastStore } from '$stores/toast.svelte';
  import type {
    ClassSubsetPreset,
    PreflightReport,
    Profile,
    TrainCampaignSpec,
    TrainJobSpec,
    TrainJobStatus,
  } from '$lib/types_train';

  $effect(() => {
    keyboardStore.setScope('train');
  });

  // Subscribe to the classes store so the picker has live data — the
  // parent layout already does this on /clusters/* but not on standalone
  // routes like /train.
  $effect(() => {
    const release = classesStore.acquire();
    return release;
  });

  // ---- Reference data --------------------------------------------------
  let profiles = $state<Profile[]>([]);
  let presets = $state<ClassSubsetPreset[]>([]);
  let datasetExportDir = $state<string>('');
  let datasetMessage = $state<string | null>(null);
  let refreshing = $state<boolean>(false);

  async function refreshDataset(): Promise<void> {
    refreshing = true;
    datasetMessage = null;
    try {
      const e = await exportStatus();
      datasetExportDir = e.export_dir ?? '';
      if (!datasetExportDir) {
        datasetMessage =
          'No frozen export available — run /export first to produce a dataset.';
      }
    } catch (e) {
      datasetMessage = `Export status fetch failed: ${(e as Error).message}`;
    } finally {
      refreshing = false;
    }
  }

  // ---- Active job + status polling -------------------------------------
  let activeStatus = $state<TrainJobStatus | null>(null);
  let statusPoll: ReturnType<typeof setInterval> | null = null;
  let statusAbort: AbortController | null = null;

  // We keep a separate reference to the *most recent* job_id we kicked
  // off ourselves so the log tail and progress panel can stay glued to
  // it across status churn.
  let trackedJobId = $state<string | null>(null);

  const ACTIVE_STATES: ReadonlySet<string> = new Set([
    'queued',
    'starting',
    'running',
    'exporting',
  ]);
  const isActive = $derived(activeStatus ? ACTIVE_STATES.has(activeStatus.state) : false);

  async function pullStatus(): Promise<void> {
    statusAbort?.abort();
    statusAbort = new AbortController();
    try {
      const next = trackedJobId
        ? await getTrainStatus(trackedJobId, statusAbort.signal)
        : await getTrainStatus(undefined, statusAbort.signal);
      activeStatus = next;
      if (next?.state === 'finished' || next?.state === 'failed' || next?.state === 'cancelled') {
        // Refresh the past-runs table as soon as the active one
        // terminates so the row appears immediately.
        void refreshRuns();
      }
    } catch (e) {
      if ((e as Error).name === 'AbortError') return;
      // Don't toast on every poll failure — only the first.
    }
  }

  // Always poll. Fast (5s) while a run is active so the epoch counter and
  // GPU strip stay live; slow (10s) when idle so a job submitted from
  // somewhere else (curl, another browser tab) shows up automatically.
  // pullStatus() also kicks refreshRuns() on terminal-state transitions
  // so the past-runs table picks up newly-finished jobs without a manual
  // page reload.
  $effect(() => {
    void pullStatus();
    void refreshRuns();
    const interval = isActive ? 5000 : 10000;
    if (statusPoll) clearInterval(statusPoll);
    statusPoll = setInterval(() => {
      void pullStatus();
      if (!isActive) {
        // Idle path: also refresh the past-runs table on every tick so
        // jobs created out-of-band appear without a manual reload.
        void refreshRuns();
      }
    }, interval);
    return () => {
      if (statusPoll) clearInterval(statusPoll);
      statusPoll = null;
      statusAbort?.abort();
    };
  });

  // ---- Past runs (paginated) -------------------------------------------
  const RUNS_PAGE = 25;
  let runs = $state<TrainJobStatus[]>([]);
  let runsTotal = $state<number>(0);
  let runsLoading = $state<boolean>(false);
  let runsLoadingMore = $state<boolean>(false);
  let runsError = $state<string | null>(null);
  const runsHasMore = $derived(runs.length < runsTotal);

  async function refreshRuns(): Promise<void> {
    runsLoading = true;
    runsError = null;
    try {
      const res = await getTrainRuns(RUNS_PAGE, 0);
      runs = res.items ?? [];
      runsTotal = res.total ?? runs.length;
    } catch (e) {
      runsError = (e as Error).message;
    } finally {
      runsLoading = false;
    }
  }

  async function loadMoreRuns(): Promise<void> {
    if (runsLoadingMore || !runsHasMore) return;
    runsLoadingMore = true;
    try {
      const res = await getTrainRuns(RUNS_PAGE, runs.length);
      const seen = new Set(runs.map((r) => r.job_id));
      const fresh = (res.items ?? []).filter((r) => !seen.has(r.job_id));
      runs = [...runs, ...fresh];
      runsTotal = res.total ?? runsTotal;
    } catch (e) {
      runsError = (e as Error).message;
    } finally {
      runsLoadingMore = false;
    }
  }

  // ---- Campaigns -------------------------------------------------------
  // Active campaign = the campaign_id on the active run, if any. We
  // group sibling runs from the past-runs feed.
  const activeCampaignId = $derived(activeStatus?.campaign_id ?? null);
  const campaignRuns = $derived.by(() => {
    if (!activeCampaignId) return [] as TrainJobStatus[];
    const live = activeStatus;
    const all = [...runs];
    if (live && !all.some((r) => r.job_id === live.job_id)) {
      all.unshift(live);
    }
    return all.filter((r) => r.campaign_id === activeCampaignId);
  });

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
      const body = err.body as { detail?: { preflight?: PreflightReport; message?: string } };
      const pf = body.detail?.preflight;
      if (pf) preflight = pf;
      const msg = body.detail?.message ?? err.message;
      toastStore.error(`Start failed: ${msg}`);
    } else {
      toastStore.error(`Start failed: ${(err as Error).message}`);
    }
  }

  async function startSingle(spec: TrainJobSpec, force: boolean): Promise<void> {
    starting = true;
    try {
      const res = await trainStart(spec, force);
      preflight = res.preflight;
      trackedJobId = res.job_id;
      toastStore.success(`Submitted ${res.job_id}`);
      await pullStatus();
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
      trackedJobId = res.job_ids[0] ?? null;
      toastStore.success(
        `Submitted campaign ${res.campaign_id} (${res.job_ids.length} runs)`,
      );
      await pullStatus();
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
      await pullStatus();
    } catch (e) {
      toastStore.error(`Cancel failed: ${(e as Error).message}`);
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
      await pullStatus();
      await refreshRuns();
    } catch (e) {
      toastStore.error(`Cancel failed: ${(e as Error).message}`);
    } finally {
      cancellingCampaign = false;
    }
  }

  // ---- Promote modal ---------------------------------------------------
  let promoteOpen = $state<boolean>(false);
  let promoteJobId = $state<string | null>(null);
  let promoteDefaultName = $state<string>('');

  function openPromote(r: TrainJobStatus): void {
    promoteJobId = r.job_id;
    // Server-side default: `<run_name>_v7`. Run name maps to job_id
    // minus the colons / dots Triton dislikes.
    const safe = r.job_id.replace(/[^A-Za-z0-9_-]+/g, '_');
    promoteDefaultName = `${safe}_v7`;
    promoteOpen = true;
  }

  // ---- Reproduce-this-run (Phase 6, design §15.4) ---------------------
  // Fetches <job_id>.manifest.json, builds a fresh TrainJobSpec from the
  // saved spec + lineage, and POSTs /curation/train/start. Lets the user repeat
  // a known-good run without re-typing every knob.
  let reproducingId = $state<string | null>(null);

  async function reproduceRun(r: TrainJobStatus): Promise<void> {
    const ok = window.confirm(
      `Reproduce ${r.job_id}? Submits a new training job with the same spec.`,
    );
    if (!ok) return;
    reproducingId = r.job_id;
    try {
      const manifest = (await getTrainManifest(r.job_id)) as {
        spec?: Record<string, unknown>;
        lineage?: Record<string, unknown>;
      };
      const spec = (manifest.spec ?? {}) as Record<string, unknown>;
      const lineage = (manifest.lineage ?? {}) as Record<string, unknown>;
      const body: Partial<TrainJobSpec> = {
        dataset_export_dir: lineage.export_dir as string,
        include_classes: (lineage.include_classes as number[] | null) ?? null,
        single_cls: (lineage.single_cls as boolean | null) ?? false,
        cuda_visible_devices:
          (spec.cuda_visible_devices as string | undefined) ?? undefined,
        model_family: (spec.model_family as TrainJobSpec['model_family']) ?? 'yolo26',
        model_size: (spec.model_size as TrainJobSpec['model_size']) ?? 'm',
        profile: (spec.profile as TrainJobSpec['profile']) ?? 'medium',
        hyperparameters:
          (spec.hyperparameters as Record<string, unknown> | undefined) ?? {},
        augmentation:
          (spec.augmentation as TrainJobSpec['augmentation']) ?? null,
      };
      const res = await trainStart(body as TrainJobSpec);
      toastStore.success(`Reproduced as ${res.job_id}`);
      await refreshRuns();
    } catch (e) {
      toastStore.error(`Reproduce failed: ${(e as Error).message}`);
    } finally {
      reproducingId = null;
    }
  }

  // ---- Lifecycle -------------------------------------------------------
  onMount(async () => {
    await Promise.allSettled([
      (async () => {
        try {
          const p = await getTrainProfiles();
          profiles = p.profiles ?? [];
        } catch (e) {
          toastStore.error(`Profiles fetch failed: ${(e as Error).message}`);
        }
      })(),
      (async () => {
        try {
          const p = await getTrainPresets();
          presets = p.class_subset_presets ?? [];
        } catch (e) {
          toastStore.error(`Presets fetch failed: ${(e as Error).message}`);
        }
      })(),
      refreshDataset(),
      refreshRuns(),
      refreshPlateCohortCounts(),
    ]);
  });

  onDestroy(() => {
    if (statusPoll) clearInterval(statusPoll);
    statusAbort?.abort();
    preflightAbort?.abort();
  });

  // ---- Display helpers -------------------------------------------------
  function statePillClass(s: string): string {
    switch (s) {
      case 'running':
      case 'starting':
        return 'bg-blue-500/20 text-blue-200 border-blue-500/40';
      case 'exporting':
        return 'bg-purple-500/20 text-purple-200 border-purple-500/40';
      case 'finished':
        return 'bg-green-500/20 text-green-200 border-green-500/40';
      case 'failed':
        return 'bg-red-500/20 text-red-200 border-red-500/40';
      case 'cancelled':
      case 'skipped':
        return 'bg-zinc-700 text-zinc-300 border-zinc-600';
      case 'lost':
        return 'bg-yellow-500/20 text-yellow-200 border-yellow-500/40';
      case 'queued':
      default:
        return 'bg-zinc-800 text-zinc-300 border-zinc-700';
    }
  }

  function familyLabel(_r: TrainJobStatus): string {
    // The status doc doesn't carry model_family; the trainer guarantees
    // YOLO26 today. Keep the column ready for future families though.
    return 'yolo26';
  }

  // -- Plate training-cohort picker (Wave 2c E4) -------------------------
  //
  // Backed by /curation/plates/training_candidates. Counts for each mode are
  // fetched in parallel on mount, then again whenever the operator hits
  // 'Refresh'. Selecting a cohort fetches a 24-card preview so the
  // operator can sanity-check the cohort before committing to a training
  // run targeted at it.

  interface PlateCohortInfo {
    mode: TrainingCohortMode;
    label: string;
    description: string;
  }

  const PLATE_COHORTS: PlateCohortInfo[] = [
    {
      mode: 'lpr_blind_spots',
      label: 'LPR blind spots',
      description:
        'SAM3 found the plate, Gemma confirmed, LPR missed — high-signal training examples',
    },
    {
      mode: 'lpr_low_conf_correct',
      label: 'LPR low confidence',
      description: 'LPR + Gemma agreed but LPR score < 0.6 — high-loss training rows',
    },
    {
      mode: 'disagreement',
      label: 'Model disagreements',
      description: 'LPR + SAM3 both fired; review for IoU disagreement',
    },
    {
      mode: 'human_corrected',
      label: 'Human corrected',
      description: 'Human reviewed and corrected a model output — gold standard',
    },
  ];

  let plateCohortCounts = $state<Record<string, number | null>>({});
  let plateCohortMode = $state<TrainingCohortMode | null>(null);
  let plateCohortPreview = $state<PlateBrowseItem[]>([]);
  let plateCohortPreviewLoading = $state<boolean>(false);
  let plateCohortPreviewError = $state<string | null>(null);

  async function refreshPlateCohortCounts(): Promise<void> {
    const results = await Promise.allSettled(
      PLATE_COHORTS.map((c) => getTrainingCandidates(c.mode, { page_size: 1 })),
    );
    const next: Record<string, number | null> = {};
    PLATE_COHORTS.forEach((c, i) => {
      const r = results[i];
      next[c.mode] = r?.status === 'fulfilled' ? r.value.total : null;
    });
    plateCohortCounts = next;
  }

  async function loadPlateCohortPreview(mode: TrainingCohortMode): Promise<void> {
    plateCohortMode = mode;
    plateCohortPreviewLoading = true;
    plateCohortPreviewError = null;
    plateCohortPreview = [];
    try {
      const res = await getTrainingCandidates(mode, { page_size: 24 });
      plateCohortPreview = res.items;
    } catch (e) {
      plateCohortPreviewError = (e as Error).message;
    } finally {
      plateCohortPreviewLoading = false;
    }
  }

  function openPlateInReview(p: PlateBrowseItem): void {
    void goto(`/review?tab=plates&crop_id=${encodeURIComponent(p.crop_id)}`);
  }
</script>

<svelte:head>
  <title>Train · legacy Labeler</title>
</svelte:head>

<div class="mx-auto max-w-6xl space-y-4 px-4 py-6">
  <!-- Header -->
  <header class="flex flex-wrap items-end justify-between gap-3">
    <div>
      <h1 class="text-xl font-semibold tracking-tight">Train model</h1>
      <p class="mt-1 text-sm text-zinc-400">
        YOLO26 detector training over the frozen export. Submits a job to the
        legacy-trainer container and tails progress until it finishes.
      </p>
    </div>
    <button
      type="button"
      class="btn"
      onclick={async () => {
        await Promise.all([refreshDataset(), refreshRuns(), pullStatus()]);
      }}
      disabled={refreshing}
    >
      {refreshing ? 'Refreshing…' : 'Refresh'}
    </button>
  </header>

  <!-- Dataset header -->
  <section class="rounded-md border border-zinc-800 bg-zinc-900 p-4">
    <h2 class="text-[11px] uppercase tracking-wide text-zinc-500">Dataset</h2>
    {#if datasetExportDir}
      <p class="mt-1 break-all font-mono text-sm text-zinc-200">{datasetExportDir}</p>
      <p class="mt-2 flex flex-wrap gap-2 text-xs text-zinc-400">
        <span class="rounded border border-zinc-700 bg-zinc-950 px-1.5 py-0.5">
          {classesStore.classes.filter((c) => !c.deprecated).length} classes
        </span>
        <span class="rounded border border-zinc-700 bg-zinc-950 px-1.5 py-0.5 font-mono">
          {classesStore.classes
            .reduce((acc, c) => acc + (c.validated_count ?? 0), 0)
            .toLocaleString()} validated crops
        </span>
      </p>
    {:else}
      <p class="mt-1 text-sm text-zinc-300">{datasetMessage ?? 'Loading…'}</p>
    {/if}
  </section>

  <!-- Active run + log (only while actually running) -->
  {#if activeStatus && isActive}
    <TrainProgress
      status={activeStatus}
      onCancel={cancelActive}
      cancelling={cancellingRun}
    />
    <LogTail
      jobId={activeStatus.job_id}
      active={isActive}
      lines={200}
      intervalMs={2000}
      fetcher={async (jid, lines) => tailTrainLog(jid, lines)}
    />
  {/if}

  <!-- Active campaign (only while a campaign is in flight) -->
  {#if activeCampaignId && campaignRuns.length > 0}
    <CampaignCard
      campaignId={activeCampaignId}
      runs={campaignRuns}
      onCancel={cancelActiveCampaign}
      onPromoteBest={openPromote}
      cancelling={cancellingCampaign}
    />
  {/if}

  <!-- Form (collapses to a hint banner when a run is active) -->
  {#if isActive}
    <p class="rounded-md border border-blue-500/40 bg-blue-500/10 px-3 py-2 text-xs text-blue-200">
      A run is in progress. Submit a new run after it finishes — the trainer
      handles one job at a time.
    </p>
  {:else if !datasetExportDir}
    <p class="rounded-md border border-yellow-500/40 bg-yellow-500/10 px-3 py-2 text-xs text-yellow-200">
      Form disabled until a frozen export is available.
    </p>
  {:else}
    <TrainForm
      datasetExportDir={datasetExportDir}
      profiles={profiles}
      presets={presets}
      preflight={preflight}
      preflighting={preflighting}
      starting={starting}
      onPreflight={runPreflight}
      onStart={startSingle}
      onStartCampaign={startCampaign}
      disabled={isActive}
    />
  {/if}

  <!-- Plate training cohorts — surfaces the 4 modes from
       /curation/plates/training_candidates so the next LPR training cycle can
       be built from "where did LPR miss but SAM3 + Gemma agree" cohorts.
       Selecting a mode loads a 24-card sanity-preview grid. -->
  <section class="rounded-md border border-zinc-800 bg-zinc-900">
    <header class="flex items-center justify-between gap-3 border-b border-zinc-800 px-3 py-2">
      <div class="flex flex-col">
        <h2 class="text-sm font-semibold text-zinc-100">Plate training cohorts</h2>
        <p class="text-[11px] text-zinc-500">
          Provenance-derived slices for the next LPR training cycle. Pick a
          mode to preview a sanity grid before committing.
        </p>
      </div>
      <button
        type="button"
        class="btn"
        onclick={() => void refreshPlateCohortCounts()}
        title="Refresh cohort counts"
      >
        Refresh
      </button>
    </header>
    <div class="grid grid-cols-2 gap-2 p-3 sm:grid-cols-4">
      {#each PLATE_COHORTS as c (c.mode)}
        {@const count = plateCohortCounts[c.mode]}
        {@const selected = plateCohortMode === c.mode}
        <button
          type="button"
          class="flex flex-col items-start gap-1 rounded-md border px-3 py-2 text-left text-xs transition-colors
                 {selected
            ? 'border-blue-500 bg-blue-500/10 text-blue-100'
            : 'border-zinc-700 bg-zinc-950 text-zinc-300 hover:border-blue-500/50'}"
          onclick={() => void loadPlateCohortPreview(c.mode)}
        >
          <span class="font-semibold">{c.label}</span>
          <span class="font-mono text-[11px] {selected ? 'text-blue-200' : 'text-zinc-400'}">
            {count == null ? '…' : count.toLocaleString()} rows
          </span>
          <span class="text-[10px] text-zinc-500">{c.description}</span>
        </button>
      {/each}
    </div>
    {#if plateCohortMode}
      <div class="border-t border-zinc-800 px-3 py-3">
        {#if plateCohortPreviewError}
          <p class="text-xs text-red-300">Preview failed: {plateCohortPreviewError}</p>
        {:else if plateCohortPreviewLoading && plateCohortPreview.length === 0}
          <p class="text-xs text-zinc-500">Loading preview…</p>
        {:else if plateCohortPreview.length === 0}
          <p class="text-xs text-zinc-500">
            No rows match this cohort yet — the re-detection drain may still
            be populating provenance. Check back as the queue drains.
          </p>
        {:else}
          <div class="grid grid-cols-3 gap-2 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8">
            {#each plateCohortPreview as p (p.crop_id)}
              <PlateCard crop={p} onclick={openPlateInReview} compact />
            {/each}
          </div>
        {/if}
      </div>
    {/if}
  </section>

  <!-- Past runs -->
  <section class="rounded-md border border-zinc-800 bg-zinc-900">
    <header class="flex items-center justify-between gap-3 border-b border-zinc-800 px-3 py-2">
      <h2 class="text-sm font-semibold text-zinc-100">Past runs</h2>
      <span class="font-mono text-xs text-zinc-500">{runs.length} / {runsTotal}</span>
    </header>
    {#if runsError}
      <p class="px-3 py-2 text-xs text-red-300">Past runs fetch failed: {runsError}</p>
    {/if}
    <div class="max-h-[40rem] overflow-auto">
      {#if runsLoading && runs.length === 0}
        <p class="px-3 py-3 text-xs text-zinc-500">Loading…</p>
      {:else if runs.length === 0}
        <p class="px-3 py-3 text-xs text-zinc-500">
          No runs yet. Start one above.
        </p>
      {:else}
        <table class="w-full table-fixed text-sm">
          <thead class="sticky top-0 bg-zinc-900 text-[11px] uppercase tracking-wide text-zinc-500">
            <tr>
              <th class="px-3 py-2 text-left">Name</th>
              <th class="w-20 px-3 py-2 text-left">Family</th>
              <th class="w-24 px-3 py-2 text-left">Status</th>
              <th class="w-28 px-3 py-2 text-right">Best mAP50</th>
              <th class="w-32 px-3 py-2 text-right">Actions</th>
            </tr>
          </thead>
          <tbody>
            {#each runs as r (r.job_id)}
              <tr class="border-t border-zinc-800 hover:bg-zinc-800/50">
                <td class="px-3 py-2">
                  <div class="truncate font-mono text-xs text-zinc-200" title={r.job_id}>
                    {r.job_id}
                  </div>
                  {#if r.campaign_id}
                    <div class="truncate font-mono text-[10px] text-zinc-500" title={r.campaign_id}>
                      ↳ {r.campaign_id}
                    </div>
                  {/if}
                </td>
                <td class="px-3 py-2">
                  <span class="font-mono text-xs text-zinc-400">{familyLabel(r)}</span>
                </td>
                <td class="px-3 py-2">
                  <span
                    class="rounded-sm border px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide {statePillClass(
                      r.state,
                    )}"
                  >
                    {r.state}
                  </span>
                </td>
                <td class="px-3 py-2 text-right font-mono text-xs text-zinc-200">
                  {r.best_metric?.map50?.toFixed(3) ?? '—'}
                </td>
                <td class="flex justify-end gap-1.5 px-3 py-2 text-right">
                  {#if r.state === 'finished' || r.state === 'exporting'}
                    <button
                      type="button"
                      class="rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-blue-300 hover:border-blue-500 hover:bg-blue-500/10"
                      onclick={() => openPromote(r)}
                      title="Promote to Triton"
                    >
                      Promote ↑
                    </button>
                  {/if}
                  {#if r.state === 'finished' || r.state === 'failed'}
                    <button
                      type="button"
                      class="rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-300 hover:border-zinc-500 hover:bg-zinc-800"
                      onclick={() => reproduceRun(r)}
                      disabled={reproducingId === r.job_id}
                      title="Submit a new run with the same spec"
                    >
                      {reproducingId === r.job_id ? '…' : 'Reproduce'}
                    </button>
                  {/if}
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
        <div
          use:infiniteScroll={{
            onload: loadMoreRuns,
            disabled: runsLoadingMore || !runsHasMore || runsLoading,
          }}
          class="h-1"
          aria-hidden="true"
        ></div>
      {/if}
    </div>
  </section>
</div>

<PromoteModal
  open={promoteOpen}
  jobId={promoteJobId}
  defaultName={promoteDefaultName}
  onclose={() => (promoteOpen = false)}
  onpromoted={() => {
    promoteOpen = false;
    void refreshRuns();
  }}
/>
