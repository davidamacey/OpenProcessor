<script lang="ts">
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
   * Sub-components own their own UI; this page is the data layer.
   */
  import { goto } from '$app/navigation';
  import { onDestroy, onMount } from 'svelte';
  import {
    ApiError,
    cancelTrainCampaign,
    cancelTrainJob,
    exportSingleClass,
    exportSingleClassStatus,
    exportStatus,
    getCrops,
    getReviewQueue,
    getTestHoldoutStats,
    getTrainingCandidates,
    getTrainingCohorts,
    getTrainManifest,
    getTrainPresets,
    getTrainProfiles,
    getTrainRuns,
    getTrainStatus,
    listDatasets,
    tailTrainLog,
    trainPreflight,
    trainStart,
    trainStartCampaign,
    type RegionBrowseItem,
    type ServedTrainingCohort,
    type TrainingCohortMode,
  } from '$lib/api';
  import { formatCount } from '$lib/formatCount';
  import { infiniteScroll } from '$lib/actions/infiniteScroll';
  import MonitoringLinks from '$lib/components/MonitoringLinks.svelte';
  import CampaignCard from '$components/CampaignCard.svelte';
  import LogTail from '$components/LogTail.svelte';
  import SlotCard from '$components/SlotCard.svelte';
  import CropCard from '$components/CropCard.svelte';
  import PromoteModal from '$components/PromoteModal.svelte';
  import RunResults from '$components/RunResults.svelte';
  import TrainForm from '$components/TrainForm.svelte';
  import TrainProgress from '$components/TrainProgress.svelte';
  import { classesStore } from '$stores/classes.svelte';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { toastStore } from '$stores/toast.svelte';
  import { slotRegistry, registeredSlots } from '$lib/annotations/registeredSlots';
  import type { SlotSpec } from '$lib/annotations/types';
  import {
    CORE_COHORTS,
    cohortsForClass,
    cohortEndpointKind,
    type CohortSpec,
  } from '$lib/annotations/cohorts';
  import { datasetExportForSlot } from '$lib/annotations/datasetExport';
  import { splitCohortGroups } from '$lib/trainCohortGroups';
  import { bestMapDisplay } from '$lib/trainRunsTable';
  import { isDatasetExportAvailable } from '$lib/strategies';
  import { isTerminalTrainState } from '$lib/trainResults';
  import { strategiesStore } from '$stores/strategies.svelte';
  import type {
    Crop,
    ExportDataset,
    ExportStatus,
    ReviewItem,
    TestHoldoutStats,
  } from '$lib/types';
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
  // `GET {API_PREFIX}/test_holdout/stats` — per-class test-holdout
  // counts, shown alongside (never subtracted from) the class picker's
  // "validated crops" total so an operator can see how many of those
  // are actually trainable. `null` while unloaded or on fetch failure —
  // the picker omits the holdout clause entirely rather than guessing.
  let holdout = $state<TestHoldoutStats | null>(null);

  /**
   * The one registered slot that declares a dataset export, read off
   * `registeredSlots` rather than importing a specific profile, so
   * `/train` follows the same "register a new domain in exactly one
   * file" rule every other slot-aware route follows (see
   * `registeredSlots.ts`'s header). `undefined` when no registered slot
   * declares one, in which case the panel never renders at all.
   */
  const datasetExportSpec = registeredSlots
    .map(datasetExportForSlot)
    .find((s) => s !== undefined);

  // Capability discovery is a cached, never-rejecting one-shot
  // (strategiesStore.init() is idempotent; getMethods() degrades to
  // FALLBACK_METHODS on any failure), so calling it from an $effect is
  // the same pattern /clusters uses for the embedding-plot toggle.
  $effect(() => {
    void strategiesStore.init();
  });

  /**
   * Whether the backend advertises this export kind on
   * `GET {API_PREFIX}/methods`'s `export` axis. Absent ⇒ the panel and
   * the dataset-kind toggle that selects it are not rendered at all —
   * absent, not disabled — and `refreshSingleClassExportStatus()` is never
   * called, so
   * a deployment that never ported the LPR exporter produces zero 404s
   * on this route. Never probe the export endpoint to find out; see
   * `isDatasetExportAvailable`.
   */
  const datasetExportAvailable = $derived(
    datasetExportSpec != null &&
      isDatasetExportAvailable(
        strategiesStore.methods.dataset_exports,
        datasetExportSpec.kind,
      ),
  );

  // Which frozen export the training run targets: the multi-class dataset
  // (`MULTI_CLASS`, the `yolo` export's `current`) or the registered slot's
  // single-class dataset. `datasetKind` is `MULTI_CLASS` or whatever
  // string the active slot's `extras.datasetExport.datasetKind` declares
  // — never a hardcoded second literal, so a second registered slot with
  // its own dataset export needs no change here.
  const MULTI_CLASS = 'yolo';
  let datasetKind = $state<string>(MULTI_CLASS);
  let vehiclesDir = $state<string>('');
  // Full GET {API_PREFIX}/export/status response for the current
  // multi-class export — `class_split_counts`/`split_counts`/
  // `image_count`/`class_count` drive the dataset card's "current
  // export" numbers below, in place of the classesStore-wide validated
  // total (which double-counted holdout crops). `null`/missing fields
  // on a pre-6c77deb backend fall back to the labelled global-pool
  // numbers — see the card markup.
  let vehiclesExportState = $state<ExportStatus | null>(null);
  let singleClassExportDir = $state<string>('');
  // All materialized dataset versions on disk (both kinds), newest first.
  let datasets = $state<ExportDataset[]>([]);
  // Explicit operator pick. Empty => fall back to the `current` symlink for the
  // selected kind, so the default behaviour (train the latest export) is
  // unchanged. Picking any past export lets us reuse the exact same data when
  // upsizing nano -> small, etc.
  let selectedExportDir = $state<string>('');
  // /export/status describes only the latest export, so its counts apply
  // when nothing is picked or the pick is that same directory.
  const showsCurrentExport = $derived(
    !selectedExportDir ||
      selectedExportDir === vehiclesExportState?.export_dir ||
      selectedExportDir === vehiclesExportState?.path,
  );
  // Multi-class rows are `kind: 'yolo'`; a slot's rows are `single_class`
  // rows written under its own `profile_name`.
  const kindDatasets = $derived(
    datasets.filter((d) =>
      datasetExportSpec && datasetKind === datasetExportSpec.datasetKind
        ? d.kind === datasetExportSpec.kind &&
          d.profile_name === datasetExportSpec.profileName
        : d.kind === MULTI_CLASS,
    ),
  );
  let datasetExportDir = $derived(
    selectedExportDir ||
      (datasetKind === datasetExportSpec?.datasetKind
        ? singleClassExportDir
        : vehiclesDir),
  );
  let datasetMessage = $state<string | null>(null);
  let refreshing = $state<boolean>(false);

  function selectDatasetKind(kind: string): void {
    datasetKind = kind;
    selectedExportDir = ''; // reset to the current export of the new kind
  }

  // `/methods` resolves after mount, so the single-class toggle can
  // disappear while its kind is selected. Fall back to the always-present
  // multi-class dataset rather than leaving the picker pointed at a kind with no UI
  // behind it — same "force the stale selection off" pattern
  // /clusters uses for the embedding-plot toggle.
  $effect(() => {
    if (!datasetExportAvailable && datasetKind !== MULTI_CLASS) {
      selectDatasetKind(MULTI_CLASS);
    }
  });

  async function refreshDataset(): Promise<void> {
    refreshing = true;
    datasetMessage = null;
    try {
      const [e, ds] = await Promise.all([exportStatus(), listDatasets()]);
      vehiclesDir = e.export_dir ?? '';
      vehiclesExportState = e;
      datasets = ds.datasets ?? [];
      if (!vehiclesDir) {
        datasetMessage =
          'No frozen export available — run /export first to produce a dataset.';
      }
    } catch (e) {
      datasetMessage = `Export status fetch failed: ${(e as Error).message}`;
    } finally {
      refreshing = false;
    }
  }

  // Human-readable label for a dataset option in the picker.
  function datasetLabel(d: ExportDataset): string {
    const n = d.image_count != null ? d.image_count.toLocaleString() : '?';
    const tag = d.version_tag ? ` · ${d.version_tag}` : '';
    const samp = d.sampling === 'stratified_even' ? ' · sampled' : '';
    const cur = d.is_current ? ' · current' : '';
    const when = d.exported_at ? d.exported_at.slice(0, 16).replace('T', ' ') : '';
    const dir = d.export_dir.split('/').pop() ?? d.export_dir;
    return `${dir} (${n} imgs${samp}${tag}${cur}) ${when}`.trim();
  }

  // ---- Single-class dataset export (the export-capable slot) ------------
  // Built on demand via the active slot's `extras.datasetExport` spec,
  // through OpenProcessor's generic `POST /export/single_class`. Backend
  // is synchronous, so we just await it and surface the resulting dir +
  // counts. Only the four typed option controls below stay hand-written —
  // see docs/design/bakeoff-train-genericization-plan-2026-09-21.md
  // §3.3/§3.4 for why generalizing them would produce a form builder with
  // exactly one form to build.
  let singleClassExporting = $state<boolean>(false);
  let singleClassExportMessage = $state<string | null>(null);
  // Export options. whole_frame = full source frame (deployment distribution);
  // item_crop = parent item crop with the region re-projected. 640 for a
  // fast pass, 1280 for the full run. dedup collapses >=0.98 near-dup frames.
  let singleClassImageMode = $state<'whole_frame' | 'item_crop'>('whole_frame');
  let singleClassImgSize = $state<640 | 1280>(1280);
  let singleClassDedup = $state<boolean>(true);
  // Optional N: sample at most this many positive (region-bearing) frames,
  // spread EVENLY across region clusters. Blank/0 == every positive. Lets us
  // build progressively larger dataset versions from the same labeled pool.
  let singleClassMaxPositives = $state<number | null>(null);

  async function refreshSingleClassExportStatus(): Promise<void> {
    try {
      if (!datasetExportSpec) return;
      const s = await exportSingleClassStatus(datasetExportSpec);
      singleClassExportDir = s.export_dir ?? '';
    } catch {
      // Non-fatal — the export just hasn't run yet.
    }
  }

  // Only ask for export status once the capability gate says the route
  // exists. Firing it unconditionally on mount would 404 on a backend
  // that doesn't advertise this export kind, and
  // refreshSingleClassExportStatus()'s bare `catch {}` would make that
  // invisible outside the network tab. Plain `let`, not `$state` — writing it must
  // not re-trigger this effect.
  let singleClassStatusRequested = false;
  $effect(() => {
    if (datasetExportAvailable && !singleClassStatusRequested) {
      singleClassStatusRequested = true;
      void refreshSingleClassExportStatus();
    }
  });

  async function runSingleClassExport(): Promise<void> {
    // Non-null: this handler only runs from a button inside
    // `{#if datasetExportAvailable && datasetExportSpec}`.
    const spec = datasetExportSpec!;
    singleClassExporting = true;
    singleClassExportMessage = null;
    try {
      const r = await exportSingleClass(spec, {
        image_mode: singleClassImageMode,
        img_max_side: singleClassImgSize,
        dedup_threshold: singleClassDedup ? 0.98 : null,
        max_positive_images:
          singleClassMaxPositives && singleClassMaxPositives > 0
            ? singleClassMaxPositives
            : undefined,
      });
      singleClassExportDir = r.export_dir;
      // Refresh the picker so the new version shows up immediately.
      void refreshDataset();
      const pos = r.positive_images ?? '?';
      const bg = r.background_images ?? '?';
      singleClassExportMessage = `${spec.label} done — ${r.image_count} images (${pos} positives, ${bg} backgrounds), ${singleClassImageMode} @ ${singleClassImgSize}px${singleClassDedup ? ', dedup 0.98' : ''}. dataset_sha ${r.dataset_sha.slice(0, 12)}`;
      if (r.positives_zero_warning) {
        singleClassExportMessage +=
          ' Warning: zero positives — check labeling for this slot.';
      }
      toastStore.success(`${spec.label} complete`);
    } catch (e) {
      singleClassExportMessage = `${spec.label} failed: ${(e as Error).message}`;
      toastStore.error(`${spec.label} failed`);
    } finally {
      singleClassExporting = false;
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
      if (
        next?.state === 'finished' ||
        next?.state === 'failed' ||
        next?.state === 'cancelled'
      ) {
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
      const body = err.body as {
        detail?: {
          preflight?: PreflightReport;
          message?: string;
          // 6c77deb: an unknown `augmentation.preset` 422s with
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

  // -- Training-cohort picker (2026-09-24 logic-moves W6, item 13;
  // originally P2.14, docs/genericization-plan-2026-09-13.md §9.3) -----
  //
  // Cohort definitions now come from the backend's own `GET
  // {API_PREFIX}/training_cohorts?class_id=` — `id`/`label`/`description`/
  // `endpoint`/`params`/`row_kind` are served verbatim, already resolved
  // for the requested class (no client `{classId}` template compilation
  // for these). This replaces the old client-side `cohortsForClass()`
  // (CORE_COHORTS + a slot's own declared modes) as the
  // primary source: the backend serves the same 4 core + N region
  // cohorts today, so nothing the operator sees changes, but a
  // threshold like `low_confidence`'s `classifier_conf_lt` now comes
  // from the server rather than a client constant.
  //
  // `cohortsForClass()`/`CORE_COHORTS`/`cohorts.ts`'s tier-2 mechanism
  // is NOT deleted — a deployment can still register a brand-new slot
  // via `annotation-profiles.json` (parseSlotConfig.ts) that the
  // backend has no region profile for, and that slot's own hand-declared
  // `capabilities.trainingCohorts.cohorts` (checked at `loadGroupCohorts`
  // below) still surfaces here as a fallback for any cohort id the
  // server didn't already send — the server always wins on an id
  // collision. See secondSlotIntegration.test.ts / cohorts.test.ts for
  // the tier-2 mechanism's own coverage, independent of this page.
  //
  // Scope note: cohorts are computed for every non-deprecated class
  // (the "default to all classes" recommendation, §9.11) rather than
  // synced to TrainForm's own class-subset selection — that tighter
  // coupling (lifting `selectedClasses` into this page) is deliberately
  // NOT done here; TrainForm's selection continues to drive the actual
  // training run unchanged.
  const classesById = $derived(new Map(classesStore.classes.map((c) => [c.id, c.name])));
  const coreCohortIds = new Set(CORE_COHORTS.map((c) => c.id));

  interface CohortGroup {
    classId: number;
    className: string;
    cohorts: CohortSpec[];
  }

  // Cohort *definitions* load lazily per class (see lazyLoadGroupCounts
  // below) — `classCohorts` starts empty for every class and is filled
  // in the same intersection-observer callback that used to load only
  // counts, so mounting this page never fires an eager N-classes
  // request storm.
  let classCohorts = $state<Record<number, CohortSpec[]>>({});

  const cohortGroups = $derived.by<CohortGroup[]>(() =>
    classesStore.classes
      .filter((c) => !c.deprecated)
      .map((c) => ({
        classId: c.id,
        className: c.name,
        cohorts: classCohorts[c.id] ?? [],
      })),
  );

  /** Converts one served cohort into the local `CohortSpec` shape the
   *  preview grid already understands — `params` are used verbatim
   *  (already resolved for this class by the server), never
   *  re-templated. `row_kind: 'region'` renders via `SlotCard`
   *  ('slot') and jumps to the owning slot's review queue on click;
   *  `'crop'` renders via `CropCard` and jumps to the All review tab. */
  function fromServedCohort(c: ServedTrainingCohort): CohortSpec {
    return {
      id: c.id,
      label: c.label,
      description: c.description,
      query: {
        kind: 'endpoint',
        path: c.endpoint,
        params: c.params as Record<string, string | number | boolean>,
      },
      rowKind: c.row_kind === 'region' ? 'slot' : 'crop',
      reviewTarget: c.row_kind === 'region' ? 'slotQueue' : 'all',
    };
  }

  /** Server cohorts first; a slot's own tier-2-declared cohort fills in
   *  only an id the server didn't already send — see the header comment
   *  above. `predicateCohortsAvailable=false`: no backend support for a
   *  tier-2 predicate cohort exists yet, so `cohortsForClass` here can
   *  only ever contribute CORE_COHORTS (filtered out below, since the
   *  server already sent its own core cohorts) or a slot's declared
   *  cohorts. */
  async function loadGroupCohorts(group: CohortGroup): Promise<CohortSpec[]> {
    const served = await getTrainingCohorts(group.classId).catch(
      () => ({ cohorts: [] }) as { cohorts: ServedTrainingCohort[] },
    );
    const servedCohorts = served.cohorts.map(fromServedCohort);
    const servedIds = new Set(servedCohorts.map((c) => c.id));
    const declaredFallback = cohortsForClass(
      group.classId,
      group.className,
      slotRegistry,
      classesById,
      false,
    ).filter((c) => !coreCohortIds.has(c.id) && !servedIds.has(c.id));
    return [...servedCohorts, ...declaredFallback];
  }

  let cohortCounts = $state<Record<string, number | null>>({});
  let selectedCohortKey = $state<string | null>(null);
  let cohortPreview = $state<Array<RegionBrowseItem | Crop | ReviewItem>>([]);
  let cohortPreviewLoading = $state<boolean>(false);
  let cohortPreviewError = $state<string | null>(null);

  /** Stable key across (class, cohort) pairs — a cohort id alone isn't
   *  unique once multiple classes are shown (e.g. every class has a
   *  'validated' core cohort). */
  function cohortKey(classId: number, cohort: CohortSpec): string {
    return `${classId}:${cohort.id}`;
  }

  // m-train-cohorts (2026-09-24 interactive pass): with ~85 classes × up
  // to 8 cohorts, most classes' chips are all 0 — a wall of zeros
  // ("subaru_brz", "dumptruck", "class_e", …) dominating the section above
  // the actually-useful Past runs table. Collapsing logic lives in
  // `$lib/trainCohortGroups.ts` (pure, unit-tested) — a class only
  // collapses once every one of its cohorts SERVED a count of exactly 0.
  const cohortGroupSplit = $derived(splitCohortGroups(cohortGroups, cohortCounts));
  const visibleCohortGroups = $derived(cohortGroupSplit.visible);
  const zeroCandidateGroups = $derived(cohortGroupSplit.zero);
  const cohortGroupsPending = $derived(cohortGroupSplit.pending);

  /** Dispatches a cohort's `endpoint`/`params` (served verbatim by
   *  `{API_PREFIX}/training_cohorts`, or a tier-2 declared fallback
   *  compiled the same way) to the one existing api.ts function that
   *  already answers that endpoint shape — keyed structurally by
   *  `cohortEndpointKind()` rather than a path string-equality check (Wave 2 C12 — see cohorts.ts's doc comment
   *  on `cohortEndpointKind` for why the old check silently broke on a
   *  base-path rename). Anything else (a future slot's endpoint cohort
   *  naming a path none of these three recognize) fails closed to an
   *  empty/null result rather than guessing at an endpoint shape. */
  async function runCohortQuery(
    cohort: CohortSpec,
    pageSize: number,
  ): Promise<{ total: number; items: Array<RegionBrowseItem | Crop | ReviewItem> }> {
    if (cohort.query.kind !== 'endpoint') return { total: 0, items: [] };
    const { path, params } = cohort.query;
    const classId =
      typeof params.class_id === 'number'
        ? params.class_id
        : typeof params.class_id === 'string'
          ? Number(params.class_id)
          : undefined;
    const kind = cohortEndpointKind(path);

    if (kind === 'training_candidates') {
      const mode = params.mode as TrainingCohortMode;
      const res = await getTrainingCandidates(mode, {
        page_size: pageSize,
        class_id: classId,
      });
      return { total: res.total, items: res.items };
    }
    if (kind === 'crops') {
      const res = await getCrops({
        class_id: classId,
        label_validated:
          typeof params.label_validated === 'boolean'
            ? params.label_validated
            : undefined,
        classifier_conf_lt:
          typeof params.classifier_conf_lt === 'number'
            ? params.classifier_conf_lt
            : undefined,
        limit: pageSize,
      });
      return { total: res.total, items: res.items };
    }
    if (kind === 'model_disagreements') {
      const res = await getReviewQueue('model_disagreements', 1, pageSize, {
        class_id: classId,
      });
      return { total: res.total, items: res.items };
    }
    return { total: 0, items: [] };
  }

  // Lazy per-group cohorts + counts (§9.10's mitigation for the
  // N-classes × M-cohorts request storm): a group's cohort definitions
  // AND counts load once, the first time its header scrolls into the
  // viewport, instead of every group firing requests on mount.
  const groupCountsLoaded = new Set<number>();
  function lazyLoadGroupCounts(node: HTMLElement, group: CohortGroup) {
    const observer = new IntersectionObserver(
      (entries) => {
        if (
          entries.some((e) => e.isIntersecting) &&
          !groupCountsLoaded.has(group.classId)
        ) {
          groupCountsLoaded.add(group.classId);
          void loadGroupCounts(group);
        }
      },
      { rootMargin: '200px' },
    );
    observer.observe(node);
    return { destroy: () => observer.disconnect() };
  }

  async function loadGroupCounts(group: CohortGroup): Promise<void> {
    const cohorts = await loadGroupCohorts(group);
    classCohorts = { ...classCohorts, [group.classId]: cohorts };
    const results = await Promise.allSettled(cohorts.map((c) => runCohortQuery(c, 1)));
    const next: Record<string, number | null> = { ...cohortCounts };
    cohorts.forEach((cohort, i) => {
      const r = results[i];
      next[cohortKey(group.classId, cohort)] =
        r?.status === 'fulfilled' ? r.value.total : null;
    });
    cohortCounts = next;
  }

  async function refreshCohortCounts(): Promise<void> {
    const all = cohortGroups.flatMap((g) =>
      g.cohorts.map((c) => ({ group: g, cohort: c })),
    );
    const results = await Promise.allSettled(
      all.map(({ cohort }) => runCohortQuery(cohort, 1)),
    );
    const next: Record<string, number | null> = { ...cohortCounts };
    all.forEach(({ group, cohort }, i) => {
      const r = results[i];
      next[cohortKey(group.classId, cohort)] =
        r?.status === 'fulfilled' ? r.value.total : null;
    });
    cohortCounts = next;
  }

  async function loadCohortPreview(
    group: CohortGroup,
    cohort: CohortSpec,
  ): Promise<void> {
    selectedCohortKey = cohortKey(group.classId, cohort);
    cohortPreviewLoading = true;
    cohortPreviewError = null;
    cohortPreview = [];
    try {
      const res = await runCohortQuery(cohort, 24);
      cohortPreview = res.items;
    } catch (e) {
      cohortPreviewError = (e as Error).message;
    } finally {
      cohortPreviewLoading = false;
    }
  }

  /** The slot a region cohort's rows render with: the slot bound to this
   *  class group, else the only registered queue slot with a sub-box.
   *  `undefined` renders no region cards rather than guessing a slot. */
  function regionSlotForGroup(group: CohortGroup): SlotSpec | undefined {
    const bound = slotRegistry.forClass(group.classId, classesById)[0];
    if (bound) return bound;
    const regionSlots = slotRegistry.queues.filter((s) => s.capabilities.subBox != null);
    return regionSlots.length === 1 ? regionSlots[0] : undefined;
  }

  /** Click target from `reviewTarget` (§9.3 step 7): the owning slot's
   *  queue tab, else the All tab. */
  function openCohortItem(
    group: CohortGroup,
    cohort: CohortSpec,
    item: RegionBrowseItem | Crop | ReviewItem,
  ): void {
    const cropId = 'crop_id' in item ? item.crop_id : item.id;
    if (cohort.reviewTarget === 'slotQueue') {
      const slot = slotRegistry.forClass(group.classId, classesById)[0];
      const urlId = slot?.capabilities.queue?.urlId ?? 'all';
      void goto(`/review?tab=${urlId}&crop_id=${encodeURIComponent(cropId)}`);
      return;
    }
    void goto(`/review?tab=all&crop_id=${encodeURIComponent(cropId)}`);
  }
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
        <MonitoringLinks mlflowRunUrls={runs.map((r) => r.mlflow_run_url)} />
      </div>
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
    <div class="flex items-center justify-between gap-3">
      <h2 class="text-[11px] uppercase tracking-wide text-zinc-500">Dataset</h2>
      <div class="flex gap-1 text-xs">
        <button
          type="button"
          class="rounded border px-2 py-0.5 {datasetKind === MULTI_CLASS
            ? 'border-blue-500 bg-blue-950 text-blue-200'
            : 'border-zinc-700 bg-zinc-950 text-zinc-400 hover:bg-zinc-800'}"
          onclick={() => selectDatasetKind(MULTI_CLASS)}
        >
          Multi-class
        </button>
        {#if datasetExportAvailable}
          <button
            type="button"
            class="rounded border px-2 py-0.5 {datasetKind ===
            datasetExportSpec?.datasetKind
              ? 'border-blue-500 bg-blue-950 text-blue-200'
              : 'border-zinc-700 bg-zinc-950 text-zinc-400 hover:bg-zinc-800'}"
            onclick={() => selectDatasetKind(datasetExportSpec!.datasetKind)}
          >
            {datasetExportSpec!.label}{datasetExportSpec!.singleClass
              ? ' (single-class)'
              : ''}
          </button>
        {/if}
      </div>
    </div>
    {#if kindDatasets.length > 0}
      <label class="mt-3 block">
        <span class="mb-1 block text-xs text-zinc-400">
          dataset version ({kindDatasets.length} available — pick a sample, subset, or the full
          set)
        </span>
        <select
          bind:value={selectedExportDir}
          class="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 font-mono text-xs text-zinc-100 focus:border-blue-500 focus:outline-none"
        >
          <option value="">
            current ({datasetKind === datasetExportSpec?.datasetKind
              ? `${datasetExportSpec.profileName} current`
              : 'current'} symlink — latest)
          </option>
          {#each kindDatasets as d (d.export_dir)}
            <option value={d.export_dir}>{datasetLabel(d)}</option>
          {/each}
        </select>
      </label>
    {/if}
    {#if datasetExportDir}
      <p class="mt-1 break-all font-mono text-sm text-zinc-200">{datasetExportDir}</p>
      {#if datasetKind === datasetExportSpec?.datasetKind}
        <p class="mt-2 text-xs text-zinc-400">
          {datasetExportSpec.blurb}
        </p>
      {:else if showsCurrentExport && vehiclesExportState?.class_split_counts}
        <!-- Current export's own contents (OpenProcessor 6c77deb's
             GET {API_PREFIX}/export/status) — only valid for the current
             export, so this branch renders for the `current` symlink or an
             explicit pick of that same directory, never a past version. m-train-card (2026-09-24): this used to show the
             classesStore-wide validated total (which counted
             test_holdout crops too), a different — and often much
             bigger — number than what this specific export actually
             contains. -->
        <div class="mt-2 flex flex-wrap gap-2 text-xs text-zinc-400">
          {#if vehiclesExportState.class_count != null}
            <span class="rounded border border-zinc-700 bg-zinc-950 px-1.5 py-0.5">
              {vehiclesExportState.class_count} classes
            </span>
          {/if}
          {#if vehiclesExportState.image_count != null || vehiclesExportState.object_count != null}
            <span
              class="rounded border border-zinc-700 bg-zinc-950 px-1.5 py-0.5 font-mono"
            >
              {formatCount(vehiclesExportState.object_count)} objects in {formatCount(
                vehiclesExportState.image_count,
              )} images
              {#if vehiclesExportState.group_key}
                <span class="text-zinc-500">(by {vehiclesExportState.group_key})</span>
              {/if}
            </span>
          {/if}
          {#if vehiclesExportState.split_counts}
            <span
              class="rounded border border-zinc-700 bg-zinc-950 px-1.5 py-0.5 font-mono"
              title="Images per split"
            >
              images: train {vehiclesExportState.split_counts.train.toLocaleString()} · val
              {vehiclesExportState.split_counts.val.toLocaleString()}
              · test {vehiclesExportState.split_counts.test.toLocaleString()}
            </span>
          {/if}
          {#if vehiclesExportState.split_object_counts}
            <span
              class="rounded border border-zinc-700 bg-zinc-950 px-1.5 py-0.5 font-mono"
              title="Objects (label lines) per split"
            >
              objects: train {vehiclesExportState.split_object_counts.train.toLocaleString()}
              · val {vehiclesExportState.split_object_counts.val.toLocaleString()}
              · test {vehiclesExportState.split_object_counts.test.toLocaleString()}
            </span>
          {/if}
        </div>
        <details class="mt-2 text-xs text-zinc-400">
          <summary class="cursor-pointer hover:text-zinc-200">
            per-class object counts ({vehiclesExportState.class_split_counts.length})
          </summary>
          <div class="mt-1 max-h-48 overflow-auto rounded border border-zinc-800">
            <table class="w-full text-xs">
              <thead
                class="sticky top-0 border-b border-zinc-800 bg-zinc-950 text-left uppercase text-zinc-500"
              >
                <tr>
                  <th class="px-2 py-1 font-medium">Class</th>
                  <th class="px-2 py-1 text-right font-medium">Train (objects)</th>
                  <th class="px-2 py-1 text-right font-medium">Val (objects)</th>
                  <th class="px-2 py-1 text-right font-medium">Test (objects)</th>
                </tr>
              </thead>
              <tbody>
                {#each vehiclesExportState.class_split_counts as c (c.class_id)}
                  {@const missing = c.train === 0 || c.val === 0}
                  <tr
                    class="border-b border-zinc-900 {missing
                      ? 'bg-red-500/10 text-red-200'
                      : 'text-zinc-300'}"
                  >
                    <td class="px-2 py-1">{c.class_name}</td>
                    <td class="px-2 py-1 text-right font-mono"
                      >{c.train.toLocaleString()}</td
                    >
                    <td class="px-2 py-1 text-right font-mono"
                      >{c.val.toLocaleString()}</td
                    >
                    <td class="px-2 py-1 text-right font-mono"
                      >{c.test.toLocaleString()}</td
                    >
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
        </details>
      {:else}
        <!-- Fallback: no per-export split data (older backend, or the
             operator picked a specific past export version this endpoint
             can't describe) — clearly labelled as the dataset-wide global
             pool, not this export's own contents. -->
        <p class="mt-2 flex flex-wrap gap-2 text-xs text-zinc-400">
          <span class="rounded border border-zinc-700 bg-zinc-950 px-1.5 py-0.5">
            {classesStore.classes.filter((c) => !c.deprecated).length} classes (global pool)
          </span>
          <span
            class="rounded border border-zinc-700 bg-zinc-950 px-1.5 py-0.5 font-mono"
          >
            {classesStore.classes
              .reduce((acc, c) => acc + (c.validated_count ?? 0), 0)
              .toLocaleString()} validated crops (global pool, not this export)
          </span>
        </p>
      {/if}
    {:else}
      <p class="mt-1 text-sm text-zinc-300">
        {datasetKind === datasetExportSpec?.datasetKind
          ? `No ${datasetExportSpec?.label} yet — build one below.`
          : (datasetMessage ?? 'Loading…')}
      </p>
    {/if}
  </section>

  {#if datasetExportAvailable && datasetExportSpec}
    <!-- Single-class dataset export, gated on the /methods `export` axis -->
    <section class="rounded-md border border-zinc-800 bg-zinc-900 p-4">
      <div class="flex items-center justify-between gap-3">
        <h2 class="text-[11px] uppercase tracking-wide text-zinc-500">
          {datasetExportSpec.label}
        </h2>
        <button
          type="button"
          class="rounded border border-zinc-700 bg-zinc-950 px-2.5 py-1 text-xs text-zinc-200 hover:bg-zinc-800 disabled:opacity-50"
          onclick={runSingleClassExport}
          disabled={singleClassExporting}
        >
          {singleClassExporting ? 'Exporting…' : `Build ${datasetExportSpec.label}`}
        </button>
      </div>
      <div class="mt-3 flex flex-wrap items-end gap-4">
        <label class="block">
          <span class="mb-1 block text-xs text-zinc-400">image mode</span>
          <select
            bind:value={singleClassImageMode}
            disabled={singleClassExporting}
            class="rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
          >
            <option value="whole_frame">whole frame</option>
            <option value="item_crop">parent crop</option>
          </select>
        </label>
        <label class="block">
          <span class="mb-1 block text-xs text-zinc-400">image size</span>
          <select
            bind:value={singleClassImgSize}
            disabled={singleClassExporting}
            class="rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
          >
            <option value={640}>640 (fast)</option>
            <option value={1280}>1280 (full)</option>
          </select>
        </label>
        <label class="block">
          <span class="mb-1 block text-xs text-zinc-400"
            >sample N positives (blank = all)</span
          >
          <input
            type="number"
            min="0"
            step="500"
            placeholder="all"
            bind:value={singleClassMaxPositives}
            disabled={singleClassExporting}
            class="w-32 rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
          />
        </label>
        <label class="flex items-center gap-2 pb-1.5">
          <input
            type="checkbox"
            bind:checked={singleClassDedup}
            disabled={singleClassExporting}
          />
          <span class="text-xs text-zinc-400">dedup near-dup frames (cos ≥ 0.98)</span>
        </label>
      </div>
      <p class="mt-2 text-[11px] text-zinc-500">
        N samples positives spread <em>evenly across clusters</em> — build a small set first,
        then a larger one from the same labeled pool for progressive training.
      </p>
      {#if singleClassExportDir}
        <p class="mt-1 break-all font-mono text-sm text-zinc-200">
          {singleClassExportDir}
        </p>
      {/if}
      {#if singleClassExportMessage}
        <p class="mt-2 text-xs text-zinc-400">{singleClassExportMessage}</p>
      {:else}
        <p class="mt-2 text-xs text-zinc-500">{datasetExportSpec.blurb}</p>
      {/if}
    </section>
  {/if}

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
    <p
      class="rounded-md border border-blue-500/40 bg-blue-500/10 px-3 py-2 text-xs text-blue-200"
    >
      A run is in progress. Submit a new run after it finishes — the trainer handles one
      job at a time.
    </p>
  {:else if !datasetExportDir}
    <p
      class="rounded-md border border-yellow-500/40 bg-yellow-500/10 px-3 py-2 text-xs text-yellow-200"
    >
      Form disabled until a frozen export is available.
    </p>
  {:else}
    <TrainForm
      {datasetExportDir}
      {profiles}
      {presets}
      {holdout}
      {preflight}
      {preflighting}
      {starting}
      onPreflight={runPreflight}
      onStart={startSingle}
      onStartCampaign={startCampaign}
      disabled={isActive}
      singleClassExport={datasetExportSpec != null &&
        datasetKind === datasetExportSpec.datasetKind &&
        datasetExportSpec.singleClass}
    />
  {/if}

  <!-- Training cohorts (2026-09-24 logic-moves W6; originally P2.14)
       — GET {API_PREFIX}/training_cohorts?class_id=
       serves both the region profile's cohorts AND every other
       class's 4 class-agnostic core cohorts; a tier-2 slot's own
       declared cohort fills in only if the server didn't already send
       that id (see loadGroupCohorts). Grouped by class so "pick the
       classes you want to train and it goes" reads directly off the
       screen. Selecting a chip
       loads a 24-card sanity-preview grid. -->
  <section class="rounded-md border border-zinc-800 bg-zinc-900">
    <header
      class="flex items-center justify-between gap-3 border-b border-zinc-800 px-3 py-2"
    >
      <div class="flex flex-col">
        <h2 class="text-sm font-semibold text-zinc-100">Training cohorts</h2>
        <p class="text-[11px] text-zinc-500">
          Curation-preview slices for the next training cycle, grouped by class. Pick a
          chip to preview a sanity grid before committing.
        </p>
      </div>
      <button
        type="button"
        class="btn"
        onclick={() => void refreshCohortCounts()}
        title="Refresh cohort counts"
      >
        Refresh
      </button>
    </header>
    {#each visibleCohortGroups as group (group.classId)}
      <div
        class="border-b border-zinc-800 p-3 last:border-b-0"
        use:lazyLoadGroupCounts={group}
      >
        <h3 class="mb-2 text-xs font-semibold text-zinc-300">{group.className}</h3>
        <div class="flex flex-wrap gap-1.5">
          {#each group.cohorts as cohort (cohort.id)}
            {@const key = cohortKey(group.classId, cohort)}
            {@const count = cohortCounts[key]}
            {@const selected = selectedCohortKey === key}
            <button
              type="button"
              class="flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition-colors
                     {selected
                ? 'border-blue-500 bg-blue-500/10 text-blue-100'
                : 'border-zinc-700 bg-zinc-950 text-zinc-300 hover:border-blue-500/50'}"
              onclick={() => void loadCohortPreview(group, cohort)}
              title={cohort.description}
            >
              <span class="font-medium">{cohort.label}</span>
              <span
                class="font-mono text-[10px] {selected
                  ? 'text-blue-200'
                  : 'text-zinc-500'}"
              >
                {count == null ? '…' : count.toLocaleString()}
              </span>
            </button>
          {/each}
        </div>
        <!-- M14 (2026-09-24 interactive pass): the preview used to render
             once, after every one of the ~85 class groups — clicking a
             chip near the top left the preview grid roughly 4,700px
             below, off-screen with no scroll-into-view, so the click
             looked like a no-op. Rendering it inline, right under the
             group whose chip was clicked, means the preview always
             appears exactly where the operator is already looking. -->
        {#if selectedCohortKey}
          {@const activeCohort = group.cohorts.find(
            (c) => cohortKey(group.classId, c) === selectedCohortKey,
          )}
          {#if activeCohort}
            <div class="mt-3 border-t border-zinc-800 pt-3">
              {#if cohortPreviewError}
                <p class="text-xs text-red-300">Preview failed: {cohortPreviewError}</p>
              {:else if cohortPreviewLoading && cohortPreview.length === 0}
                <p class="text-xs text-zinc-500">Loading preview…</p>
              {:else if cohortPreview.length === 0}
                <p class="text-xs text-zinc-500">
                  No rows match this cohort yet — the re-detection drain may still be
                  populating provenance. Check back as the queue drains.
                </p>
              {:else}
                <div
                  class="grid grid-cols-3 gap-2 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8"
                >
                  {#each cohortPreview as item ('crop_id' in item ? item.crop_id : item.id)}
                    {#if activeCohort.rowKind === 'slot'}
                      {@const cohortSlot = regionSlotForGroup(group)}
                      {#if cohortSlot}
                        <SlotCard
                          crop={item as RegionBrowseItem}
                          slot={cohortSlot}
                          onclick={(p) => openCohortItem(group, activeCohort, p)}
                          compact
                        />
                      {/if}
                    {:else}
                      <CropCard
                        crop={item as Crop}
                        onclick={(c) => openCohortItem(group, activeCohort, c)}
                      />
                    {/if}
                  {/each}
                </div>
              {/if}
            </div>
          {/if}
        {/if}
      </div>
    {/each}
    {#if zeroCandidateGroups.length > 0}
      <!-- Collapsed by default: a class only lands here once every one
           of its cohorts SERVED a count of 0 (see isAllZeroLoaded) — a
           class still loading (or one this session hasn't scrolled to
           yet) always renders in the normal list above, never here. -->
      <details class="border-b border-zinc-800 p-3 last:border-b-0">
        <summary class="cursor-pointer text-xs text-zinc-500 hover:text-zinc-300">
          {zeroCandidateGroups.length} class{zeroCandidateGroups.length === 1 ? '' : 'es'} with
          no candidates{#if cohortGroupsPending > 0}
            <span class="text-zinc-600" data-testid="zero-cohorts-so-far"
              >so far ({cohortGroupsPending} still loading)</span
            >{/if}
        </summary>
        <div class="mt-2 space-y-2">
          {#each zeroCandidateGroups as group (group.classId)}
            <div class="flex flex-wrap items-baseline gap-1.5">
              <h4 class="mr-1 text-[11px] font-semibold text-zinc-500">
                {group.className}
              </h4>
              {#each group.cohorts as cohort (cohort.id)}
                <span
                  class="rounded-full border border-zinc-800 bg-zinc-950 px-2 py-0.5 text-[10px] text-zinc-600"
                  title={cohort.description}
                >
                  {cohort.label} <span class="font-mono">0</span>
                </span>
              {/each}
            </div>
          {/each}
        </div>
      </details>
    {/if}
  </section>

  <!-- Past runs -->
  <section class="rounded-md border border-zinc-800 bg-zinc-900">
    <header
      class="flex items-center justify-between gap-3 border-b border-zinc-800 px-3 py-2"
    >
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
        <p class="px-3 py-3 text-xs text-zinc-500">No runs yet. Start one above.</p>
      {:else}
        <table class="w-full table-fixed text-sm">
          <thead
            class="sticky top-0 bg-zinc-900 text-[11px] uppercase tracking-wide text-zinc-500"
          >
            <tr>
              <th class="px-3 py-2 text-left">Name</th>
              <th class="w-20 px-3 py-2 text-left">Family</th>
              <th class="w-24 px-3 py-2 text-left">Status</th>
              <th
                class="w-28 px-3 py-2 text-right"
                title="Test split when the run's own eval reports one, otherwise best val mAP50 (a per-key max across epochs)"
              >
                mAP50
              </th>
              <th class="w-48 px-3 py-2 text-right">Actions</th>
            </tr>
          </thead>
          <tbody>
            {#each runs as r (r.job_id)}
              {@const mapDisplay = bestMapDisplay(r)}
              <tr class="border-t border-zinc-800 hover:bg-zinc-800/50">
                <td class="px-3 py-2">
                  <div class="truncate font-mono text-xs text-zinc-200" title={r.job_id}>
                    {r.job_id}
                  </div>
                  {#if r.campaign_id}
                    <div
                      class="truncate font-mono text-[10px] text-zinc-500"
                      title={r.campaign_id}
                    >
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
                <td
                  class="px-3 py-2 text-right font-mono text-xs text-zinc-200"
                  title={mapDisplay.source === 'test' ? 'test split' : 'best val mAP50'}
                >
                  {mapDisplay.value?.toFixed(3) ?? '—'}
                  <span class="block font-sans text-[9px] text-zinc-500">
                    {mapDisplay.source === 'test' ? 'test' : 'best val'}
                  </span>
                </td>
                <td class="px-3 py-2">
                  <div class="flex flex-wrap justify-end gap-1.5">
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
                  </div>
                </td>
              </tr>
              {#if isTerminalTrainState(r.state)}
                <tr class="border-t-0">
                  <td colspan="5" class="p-0">
                    <RunResults status={r} />
                  </td>
                </tr>
              {/if}
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
