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
    exportLpr,
    exportLprStatus,
    exportStatus,
    getCrops,
    getReviewQueue,
    getTrainingCandidates,
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
    type PlateBrowseItem,
    type TrainingCohortMode,
  } from '$lib/api';
  import { infiniteScroll } from '$lib/actions/infiniteScroll';
  import MonitoringLinks from '$lib/components/MonitoringLinks.svelte';
  import CampaignCard from '$components/CampaignCard.svelte';
  import LogTail from '$components/LogTail.svelte';
  import SlotCard from '$components/SlotCard.svelte';
  import CropCard from '$components/CropCard.svelte';
  import PromoteModal from '$components/PromoteModal.svelte';
  import TrainForm from '$components/TrainForm.svelte';
  import TrainProgress from '$components/TrainProgress.svelte';
  import { classesStore } from '$stores/classes.svelte';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { toastStore } from '$stores/toast.svelte';
  import { slotRegistry } from '$lib/annotations/registeredSlots';
  import { cohortsForClass, type CohortSpec } from '$lib/annotations/cohorts';
  import type { OpCrop, OpDataset, ReviewItem } from '$lib/types';
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
  // Which frozen export the training run targets: the multi-class vehicle
  // dataset (current) or the single-class LPR dataset (lpr_current).
  let datasetKind = $state<'vehicles' | 'lpr'>('vehicles');
  let vehiclesDir = $state<string>('');
  let lprExportDir = $state<string>('');
  // All materialized dataset versions on disk (both kinds), newest first.
  let datasets = $state<OpDataset[]>([]);
  // Explicit operator pick. Empty => fall back to the `current` symlink for the
  // selected kind, so the default behaviour (train the latest export) is
  // unchanged. Picking any past export lets us reuse the exact same data when
  // upsizing nano -> small, etc.
  let selectedExportDir = $state<string>('');
  const kindDatasets = $derived(datasets.filter((d) => d.kind === datasetKind));
  let datasetExportDir = $derived(
    selectedExportDir || (datasetKind === 'lpr' ? lprExportDir : vehiclesDir),
  );
  let datasetMessage = $state<string | null>(null);
  let refreshing = $state<boolean>(false);

  function selectDatasetKind(kind: 'vehicles' | 'lpr'): void {
    datasetKind = kind;
    selectedExportDir = ''; // reset to the current export of the new kind
  }

  async function refreshDataset(): Promise<void> {
    refreshing = true;
    datasetMessage = null;
    try {
      const [e, ds] = await Promise.all([exportStatus(), listDatasets()]);
      vehiclesDir = e.export_dir ?? '';
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
  function datasetLabel(d: OpDataset): string {
    const n = d.image_count != null ? d.image_count.toLocaleString() : '?';
    const tag = d.version_tag ? ` · ${d.version_tag}` : '';
    const samp = d.sampling === 'stratified_even' ? ' · sampled' : '';
    const cur = d.is_current ? ' · current' : '';
    const when = d.exported_at ? d.exported_at.slice(0, 16).replace('T', ' ') : '';
    const dir = d.export_dir.split('/').pop() ?? d.export_dir;
    return `${dir} (${n} imgs${samp}${tag}${cur}) ${when}`.trim();
  }

  // ---- LPR (license-plate) export --------------------------------------
  // Single-class plate dataset, built on demand. Backend is synchronous,
  // so we just await it and surface the resulting dir + counts.
  let lprExporting = $state<boolean>(false);
  let lprMessage = $state<string | null>(null);
  // Export options. whole_frame = full source frame (deployment distribution);
  // vehicle_crop = parent vehicle crop with the plate re-projected. 640 for a
  // fast pass, 1280 for the full run. dedup collapses >=0.98 near-dup frames.
  let lprImageMode = $state<'whole_frame' | 'vehicle_crop'>('whole_frame');
  let lprImgSize = $state<640 | 1280>(1280);
  let lprDedup = $state<boolean>(true);
  // Optional N: sample at most this many positive (plate-bearing) frames,
  // spread EVENLY across plate clusters. Blank/0 == every positive. Lets us
  // build progressively larger dataset versions from the same labeled pool.
  let lprMaxPositives = $state<number | null>(null);

  async function refreshLprStatus(): Promise<void> {
    try {
      const s = await exportLprStatus();
      lprExportDir = s.export_dir ?? '';
    } catch {
      // Non-fatal — the LPR export just hasn't run yet.
    }
  }

  async function runLprExport(): Promise<void> {
    lprExporting = true;
    lprMessage = null;
    try {
      const r = await exportLpr({
        image_mode: lprImageMode,
        img_max_side: lprImgSize,
        dedup_threshold: lprDedup ? 0.98 : null,
        max_positive_images:
          lprMaxPositives && lprMaxPositives > 0 ? lprMaxPositives : undefined,
      });
      lprExportDir = r.export_dir;
      // Refresh the picker so the new version shows up immediately.
      void refreshDataset();
      const pos = r.positive_images ?? '?';
      const fp = r.false_positive_background_images ?? '?';
      const mode = r.image_mode ?? lprImageMode;
      const size = r.img_max_side ?? lprImgSize;
      lprMessage = `LPR export done — ${r.image_count} images (${pos} positives, ${fp} FP-negatives), ${mode} @ ${size}px${lprDedup ? ', dedup 0.98' : ''}. dataset_sha ${r.dataset_sha.slice(0, 12)}`;
      if (r.positives_zero_warning) {
        lprMessage += ' ⚠ zero positives — check plate labeling.';
      }
      toastStore.success('LPR export complete');
    } catch (e) {
      lprMessage = `LPR export failed: ${(e as Error).message}`;
      toastStore.error('LPR export failed');
    } finally {
      lprExporting = false;
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
        detail?: { preflight?: PreflightReport; message?: string };
      };
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
      refreshLprStatus(),
      refreshRuns(),
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

  // -- Training-cohort picker (P2.14, docs/genericization-plan-2026-09-13.md
  // §9.3) --------------------------------------------------------------
  //
  // Generalized off the old plate-only "Wave 2c E4" panel: cohorts are
  // now derived per class via cohortsForClass() (§9.2) instead of a
  // hardcoded 4-mode PLATE_COHORTS literal. `license_plate` still gets
  // its 5 hand-tuned server-side modes (declared on the slot profile,
  // P2.13) — including the previously-unreachable 5th mode,
  // `false_positives` — every other class gets the 4 class-agnostic
  // CORE_COHORTS for free. `predicateCohortsAvailable` is hardcoded
  // false: no backend (H5, §9.4) exists yet to answer a tier-2
  // predicate cohort, so only tier-1 endpoint cohorts ever render —
  // exactly today's request shapes, nothing new sent over the wire.
  //
  // Scope note: cohorts are computed for every non-deprecated class
  // (the "default to all classes" recommendation, §9.11) rather than
  // synced to TrainForm's own class-subset selection — that tighter
  // coupling (lifting `selectedClasses` into this page) is P2.14's
  // step 1/2 in the plan and is deliberately NOT done here to keep this
  // change additive-and-reviewable; TrainForm's selection continues to
  // drive the actual training run unchanged.
  const classesById = $derived(new Map(classesStore.classes.map((c) => [c.id, c.name])));

  interface CohortGroup {
    classId: number;
    className: string;
    cohorts: CohortSpec[];
  }

  const cohortGroups = $derived.by<CohortGroup[]>(() =>
    classesStore.classes
      .filter((c) => !c.deprecated)
      .map((c) => ({
        classId: c.id,
        className: c.name,
        cohorts: cohortsForClass(c.id, c.name, slotRegistry, classesById, false),
      }))
      .filter((g) => g.cohorts.length > 0),
  );

  let cohortCounts = $state<Record<string, number | null>>({});
  let selectedCohortKey = $state<string | null>(null);
  let cohortPreview = $state<Array<PlateBrowseItem | OpCrop | ReviewItem>>([]);
  let cohortPreviewLoading = $state<boolean>(false);
  let cohortPreviewError = $state<string | null>(null);

  /** Stable key across (class, cohort) pairs — a cohort id alone isn't
   *  unique once multiple classes are shown (e.g. every class has a
   *  'validated' core cohort). */
  function cohortKey(classId: number, cohort: CohortSpec): string {
    return `${classId}:${cohort.id}`;
  }

  /** Dispatches a compiled tier-1 endpoint query to the one existing
   *  api.ts function that already answers it — the three shapes every
   *  CORE_COHORTS/licensePlateSlot cohort compiles to today (§9.1's
   *  mode table + §9.2.2's CORE_COHORTS). Anything else (a future
   *  slot's endpoint cohort naming a path none of these three
   *  recognize) fails closed to an empty/null result rather than
   *  guessing at an endpoint shape. */
  async function runCohortQuery(
    cohort: CohortSpec,
    pageSize: number,
  ): Promise<{ total: number; items: Array<PlateBrowseItem | OpCrop | ReviewItem> }> {
    if (cohort.query.kind !== 'endpoint') return { total: 0, items: [] };
    const { path, params } = cohort.query;
    const classId =
      typeof params.class_id === 'string' ? Number(params.class_id) : undefined;

    if (path === '/plates/training_candidates') {
      const mode = params.mode as TrainingCohortMode;
      const res = await getTrainingCandidates(mode, {
        page_size: pageSize,
        class_id: classId,
      });
      return { total: res.total, items: res.items };
    }
    if (path === '/crops') {
      const res = await getCrops({
        class_id: classId,
        label_validated:
          typeof params.label_validated === 'boolean'
            ? params.label_validated
            : undefined,
        v6_conf_lt: typeof params.v6_conf_lt === 'number' ? params.v6_conf_lt : undefined,
        limit: pageSize,
      });
      return { total: res.total, items: res.items };
    }
    if (path === '/review/model_disagreements') {
      const res = await getReviewQueue('model_disagreements', 1, pageSize, {
        class_id: classId,
      });
      return { total: res.total, items: res.items };
    }
    return { total: 0, items: [] };
  }

  // Lazy per-group counts (§9.10's mitigation for the N-classes ×
  // M-cohorts count-fetch storm): a group's counts load once, the first
  // time its header scrolls into the viewport, instead of every group
  // firing 4 requests on mount.
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
    const results = await Promise.allSettled(
      group.cohorts.map((c) => runCohortQuery(c, 1)),
    );
    const next: Record<string, number | null> = { ...cohortCounts };
    group.cohorts.forEach((cohort, i) => {
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

  /** Click target from `reviewTarget` (§9.3 step 7) — kills the last
   *  hardcoded `?tab=plates` literal on this route. */
  function openCohortItem(
    group: CohortGroup,
    cohort: CohortSpec,
    item: PlateBrowseItem | OpCrop | ReviewItem,
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
        legacy-trainer container and tails progress until it finishes.
      </p>
      <div class="mt-2">
        <MonitoringLinks />
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
          class="rounded border px-2 py-0.5 {datasetKind === 'vehicles'
            ? 'border-blue-500 bg-blue-950 text-blue-200'
            : 'border-zinc-700 bg-zinc-950 text-zinc-400 hover:bg-zinc-800'}"
          onclick={() => selectDatasetKind('vehicles')}
        >
          Multi-class vehicles
        </button>
        <button
          type="button"
          class="rounded border px-2 py-0.5 {datasetKind === 'lpr'
            ? 'border-blue-500 bg-blue-950 text-blue-200'
            : 'border-zinc-700 bg-zinc-950 text-zinc-400 hover:bg-zinc-800'}"
          onclick={() => selectDatasetKind('lpr')}
        >
          LPR plates (single-class)
        </button>
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
            current ({datasetKind === 'lpr' ? 'lpr_current' : 'current'} symlink — latest)
          </option>
          {#each kindDatasets as d (d.export_dir)}
            <option value={d.export_dir}>{datasetLabel(d)}</option>
          {/each}
        </select>
      </label>
    {/if}
    {#if datasetExportDir}
      <p class="mt-1 break-all font-mono text-sm text-zinc-200">{datasetExportDir}</p>
      {#if datasetKind === 'lpr'}
        <p class="mt-2 text-xs text-zinc-400">
          Single-class <span class="font-mono">license_plate</span> dataset (positives + FP
          hard-negatives + plate-free backgrounds), cluster-stratified.
        </p>
      {:else}
        <p class="mt-2 flex flex-wrap gap-2 text-xs text-zinc-400">
          <span class="rounded border border-zinc-700 bg-zinc-950 px-1.5 py-0.5">
            {classesStore.classes.filter((c) => !c.deprecated).length} classes
          </span>
          <span
            class="rounded border border-zinc-700 bg-zinc-950 px-1.5 py-0.5 font-mono"
          >
            {classesStore.classes
              .reduce((acc, c) => acc + (c.validated_count ?? 0), 0)
              .toLocaleString()} validated crops
          </span>
        </p>
      {/if}
    {:else}
      <p class="mt-1 text-sm text-zinc-300">
        {datasetKind === 'lpr'
          ? 'No LPR export yet — build one below.'
          : (datasetMessage ?? 'Loading…')}
      </p>
    {/if}
  </section>

  <!-- LPR (license-plate) export — standalone single-class dataset -->
  <section class="rounded-md border border-zinc-800 bg-zinc-900 p-4">
    <div class="flex items-center justify-between gap-3">
      <h2 class="text-[11px] uppercase tracking-wide text-zinc-500">LPR plate dataset</h2>
      <button
        type="button"
        class="rounded border border-zinc-700 bg-zinc-950 px-2.5 py-1 text-xs text-zinc-200 hover:bg-zinc-800 disabled:opacity-50"
        onclick={runLprExport}
        disabled={lprExporting}
      >
        {lprExporting ? 'Exporting…' : 'Build LPR export'}
      </button>
    </div>
    <div class="mt-3 flex flex-wrap items-end gap-4">
      <label class="block">
        <span class="mb-1 block text-xs text-zinc-400">image mode</span>
        <select
          bind:value={lprImageMode}
          disabled={lprExporting}
          class="rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
        >
          <option value="whole_frame">whole frame</option>
          <option value="vehicle_crop">vehicle crop</option>
        </select>
      </label>
      <label class="block">
        <span class="mb-1 block text-xs text-zinc-400">image size</span>
        <select
          bind:value={lprImgSize}
          disabled={lprExporting}
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
          bind:value={lprMaxPositives}
          disabled={lprExporting}
          class="w-32 rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
        />
      </label>
      <label class="flex items-center gap-2 pb-1.5">
        <input type="checkbox" bind:checked={lprDedup} disabled={lprExporting} />
        <span class="text-xs text-zinc-400">dedup near-dup frames (cos ≥ 0.98)</span>
      </label>
    </div>
    <p class="mt-2 text-[11px] text-zinc-500">
      N samples positives spread <em>evenly across plate clusters</em> — build a small set first,
      then a larger one from the same labeled pool for progressive training.
    </p>
    {#if lprExportDir}
      <p class="mt-1 break-all font-mono text-sm text-zinc-200">{lprExportDir}</p>
    {/if}
    {#if lprMessage}
      <p class="mt-2 text-xs text-zinc-400">{lprMessage}</p>
    {:else}
      <p class="mt-2 text-xs text-zinc-500">
        Single-class plate dataset (positives + human FP hard-negatives + a sample of
        plate-free backgrounds). Train it as a YOLO26 LPR detector.
      </p>
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
      {preflight}
      {preflighting}
      {starting}
      onPreflight={runPreflight}
      onStart={startSingle}
      onStartCampaign={startCampaign}
      disabled={isActive}
      lpr={datasetKind === 'lpr'}
    />
  {/if}

  <!-- Training cohorts (P2.14, formerly "Plate training cohorts") —
       cohortsForClass() (§9.2) surfaces license_plate's 5 hand-tuned
       server-side modes AND every other class's 4 class-agnostic core
       cohorts. Grouped by class so "pick the classes you want to train
       and it goes" reads directly off the screen. Selecting a chip
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
    {#each cohortGroups as group (group.classId)}
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
      </div>
    {/each}
    {#if selectedCohortKey}
      {@const activeCohort = cohortGroups
        .flatMap((g) => g.cohorts.map((c) => ({ g, c })))
        .find(({ g, c }) => cohortKey(g.classId, c) === selectedCohortKey)}
      <div class="border-t border-zinc-800 px-3 py-3">
        {#if cohortPreviewError}
          <p class="text-xs text-red-300">Preview failed: {cohortPreviewError}</p>
        {:else if cohortPreviewLoading && cohortPreview.length === 0}
          <p class="text-xs text-zinc-500">Loading preview…</p>
        {:else if cohortPreview.length === 0}
          <p class="text-xs text-zinc-500">
            No rows match this cohort yet — the re-detection drain may still be populating
            provenance. Check back as the queue drains.
          </p>
        {:else if activeCohort}
          <div
            class="grid grid-cols-3 gap-2 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8"
          >
            {#each cohortPreview as item ('crop_id' in item ? item.crop_id : item.id)}
              {#if activeCohort.c.rowKind === 'slot'}
                <SlotCard
                  crop={item as PlateBrowseItem}
                  onclick={(p) => openCohortItem(activeCohort.g, activeCohort.c, p)}
                  compact
                />
              {:else}
                <CropCard
                  crop={item as OpCrop}
                  onclick={(c) => openCohortItem(activeCohort.g, activeCohort.c, c)}
                />
              {/if}
            {/each}
          </div>
        {/if}
      </div>
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
              <th class="w-28 px-3 py-2 text-right">Best mAP50</th>
              <th class="w-48 px-3 py-2 text-right">Actions</th>
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
                <td class="px-3 py-2 text-right font-mono text-xs text-zinc-200">
                  {r.best_metric?.map50?.toFixed(3) ?? '—'}
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
