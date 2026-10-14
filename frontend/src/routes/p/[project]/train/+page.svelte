<script lang="ts">
  import { apiErrorText } from '$lib/api_errors';
  /**
   * /train — model training cockpit (Phase 2 of the training pipeline
   * design doc).
   *
   * Owns:
   *   - Reference data (profiles, presets) — fetched once on mount.
   *   - Form preflight: debounced calls so the inline check panel
   *     refreshes as the user adjusts the form.
   *   - The active run: polls `{API_PREFIX}/train/status` every 5s and the log
   *     tail every 2s while state ∈ {starting, running, exporting, queued}.
   *     Polling stops on terminal states.
   *   - Past-runs table with the standard `infiniteScroll` action.
   *   - Promote modal wiring.
   *
   * The controllers in `$lib/train/` hold the state and requests; the
   * sections in `$lib/components/train/` render them.
   */
  import { onDestroy, onMount } from 'svelte';
  import { getTestHoldoutStats } from '$lib/api_export';
  import { getTrainPresets, getTrainProfiles, tailTrainLog } from '$lib/api_train';
  import ResourceLinksRow from '$lib/components/ResourceLinksRow.svelte';
  import CampaignCard from '$components/CampaignCard.svelte';
  import LogTail from '$components/LogTail.svelte';
  import PromoteModal from '$components/PromoteModal.svelte';
  import TrainForm from '$components/TrainForm.svelte';
  import TrainProgress from '$components/TrainProgress.svelte';
  import TrainDatasetSection from '$components/train/TrainDatasetSection.svelte';
  import PastRunsTable from '$components/train/PastRunsTable.svelte';
  import TrainingCohortsPanel from '$components/train/TrainingCohortsPanel.svelte';
  import { classesStore } from '$stores/classes.svelte';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { toastStore } from '$stores/toast.svelte';
  import { createTrainDatasetController } from '$lib/train/trainDatasetController.svelte';
  import { createTrainSingleClassExport } from '$lib/train/trainSingleClassExport.svelte';
  import { createTrainRunsController } from '$lib/train/trainRunsController.svelte';
  import { createTrainActionsController } from '$lib/train/trainActionsController.svelte';
  import { createTrainCohortsController } from '$lib/train/trainCohortsController.svelte';
  import type { TestHoldoutStats } from '$lib/types_export';
  import type { ClassSubsetPreset, Profile } from '$lib/types_train';

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
  // `GET {API_PREFIX}/test_holdout/stats` — per-class test-holdout
  // counts, shown alongside (never subtracted from) the class picker's
  // "validated crops" total so an operator can see how many of those
  // are actually trainable. `null` while unloaded or on fetch failure —
  // the picker omits the holdout clause entirely rather than guessing.
  let holdout = $state<TestHoldoutStats | null>(null);

  const dataset = createTrainDatasetController();
  const exporter = createTrainSingleClassExport(dataset);
  const runs = createTrainRunsController();
  const actions = createTrainActionsController(runs);
  const cohorts = createTrainCohortsController();

  // ---- Lifecycle -------------------------------------------------------
  onMount(async () => {
    await Promise.allSettled([
      (async () => {
        try {
          const p = await getTrainProfiles();
          profiles = p.profiles ?? [];
        } catch (e) {
          toastStore.error(`Profiles fetch failed: ${apiErrorText(e)}`);
        }
      })(),
      (async () => {
        try {
          const p = await getTrainPresets();
          presets = p.class_subset_presets ?? [];
        } catch (e) {
          toastStore.error(`Presets fetch failed: ${apiErrorText(e)}`);
        }
      })(),
      dataset.refreshDataset(),
      actions.resumeActivePromote(),
      (async () => {
        try {
          holdout = await getTestHoldoutStats();
        } catch {
          // Non-fatal — the class picker just omits the holdout clause.
        }
      })(),
      // Deliberately NOT auto-fetched here: with N classes each carrying
      // ~4 cohorts, an eager fetch-on-mount is an N×4+ parallel-request
      // storm against the backend (§9.10's flagged risk). Counts load
      // lazily, only for the class group actually scrolled into view.
    ]);
  });

  onDestroy(() => {
    runs.dispose();
    actions.dispose();
  });
</script>

<svelte:head>
  <title>Train · Cropwright</title>
</svelte:head>

<div class="mx-auto max-w-6xl space-y-4 px-4 py-6">
  <!-- Header -->
  <header class="flex flex-wrap items-end justify-between gap-3">
    <div>
      <h1 class="text-xl font-semibold tracking-tight">Train model</h1>
      <p class="mt-1 text-sm text-zinc-400">
        YOLO26 detector training over the frozen export. Submits a job to the
        curation-trainer container and tails progress until it finishes.
      </p>
      <div class="mt-2">
        <ResourceLinksRow />
      </div>
    </div>
    <button
      type="button"
      class="btn"
      onclick={async () => {
        await Promise.all([
          dataset.refreshDataset(),
          runs.refreshRuns(),
          runs.pullStatus(),
        ]);
      }}
      disabled={dataset.refreshing}
    >
      {dataset.refreshing ? 'Refreshing…' : 'Refresh'}
    </button>
  </header>

  <TrainDatasetSection {dataset} {exporter} />

  <!-- Active run + log (only while actually running) -->
  {#if runs.activeStatus && runs.isActive}
    <TrainProgress
      status={runs.activeStatus}
      onCancel={actions.cancelActive}
      cancelling={actions.cancellingRun}
    />
    <LogTail
      jobId={runs.activeStatus.job_id}
      active={runs.isActive}
      lines={200}
      intervalMs={2000}
      fetcher={async (jid, lines) => tailTrainLog(jid, lines)}
    />
  {/if}

  <!-- Active campaign (only while a campaign is in flight) -->
  {#if runs.activeCampaignId && runs.campaignRuns.length > 0}
    <CampaignCard
      campaignId={runs.activeCampaignId}
      runs={runs.campaignRuns}
      onCancel={actions.cancelActiveCampaign}
      onPromoteBest={actions.openPromote}
      cancelling={actions.cancellingCampaign}
    />
  {/if}

  <!-- Form. Stays mounted (disabled) while a run is active, so
       the operator's choices survive the run instead of resetting to
       the page defaults when it finishes. -->
  {#if runs.isActive}
    <p
      class="rounded-md border border-blue-500/40 bg-blue-500/10 px-3 py-2 text-xs text-blue-200"
    >
      A run is in progress. Submit a new run after it finishes — the trainer handles one
      job at a time.
    </p>
  {/if}
  {#if !dataset.datasetExportDir}
    <p
      class="rounded-md border border-yellow-500/40 bg-yellow-500/10 px-3 py-2 text-xs text-yellow-200"
    >
      Form disabled until a frozen export is available.
    </p>
  {:else}
    <TrainForm
      datasetExportDir={dataset.datasetExportDir}
      {profiles}
      {presets}
      {holdout}
      preflight={actions.preflight}
      preflighting={actions.preflighting}
      starting={actions.starting}
      onPreflight={actions.runPreflight}
      onStart={actions.startSingle}
      onStartCampaign={actions.startCampaign}
      disabled={runs.isActive}
      singleClassExport={dataset.datasetExportSpec != null &&
        dataset.datasetKind === dataset.datasetExportSpec.datasetKind &&
        dataset.datasetExportSpec.singleClass}
    />
  {/if}

  <!-- Past runs -->
  <PastRunsTable ctl={runs} {actions} />

  <!-- Training cohorts -->
  <TrainingCohortsPanel ctl={cohorts} />
</div>

<PromoteModal
  open={actions.promoteOpen}
  jobId={actions.promoteJobId}
  defaultName={actions.promoteDefaultName}
  attachJob={actions.promoteAttach}
  onclose={() => (actions.promoteOpen = false)}
  onpromoted={() => {
    actions.promoteOpen = false;
    void runs.refreshRuns();
  }}
/>
