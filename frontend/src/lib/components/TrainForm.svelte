<script lang="ts">
  /**
   * Training form. Decomposed sub-pieces (`ClassSubsetPicker`,
   * `AugmentationPanel`) own their own UI; this component owns:
   *   - model size + profile picker
   *   - GPU selector
   *   - hyperparameter override grid
   *   - submit row + multi-size campaign mode
   *   - inline preflight rendering
   *
   * State flows up to `/train/+page.svelte` via the `submit` callbacks
   * so the page is the single source of truth for in-flight runs.
   */
  import AugmentationPanel from './AugmentationPanel.svelte';
  import ClassSubsetPicker from './ClassSubsetPicker.svelte';
  import { classesStore } from '$stores/classes.svelte';
  import { defaultTrainSelection } from '$lib/trainClassSelection';
  import { defaultGpuValue, getTrainGpus, type TrainGpuOptionsResponse } from '$lib/api';
  import type { TestHoldoutStats } from '$lib/types';
  import type {
    AugmentationSpec,
    CampaignRunSpec,
    ClassSubsetPreset,
    ModelSize,
    PreflightCheck,
    PreflightReport,
    Profile,
    ProfileName,
    TrainCampaignSpec,
    TrainJobSpec,
  } from '$lib/types_train';

  interface Props {
    datasetExportDir: string;
    profiles: Profile[];
    presets: ClassSubsetPreset[];
    /** `GET {API_PREFIX}/test_holdout/stats`, or `null` while unloaded/
     *  unavailable — passed straight through to `ClassSubsetPicker`. */
    holdout?: TestHoldoutStats | null;
    /** Live preflight result rendered inline; null while none has run. */
    preflight: PreflightReport | null;
    preflighting: boolean;
    starting: boolean;
    /** Callback the page wires to update preflight as the user changes inputs. */
    onPreflight: (spec: TrainJobSpec) => void;
    onStart: (spec: TrainJobSpec, force: boolean) => void | Promise<void>;
    onStartCampaign: (
      campaign: TrainCampaignSpec,
      force: boolean,
    ) => void | Promise<void>;
    /** When true, gray out the form (a run is active). */
    disabled?: boolean;
    /** True when the selected dataset is a single-class export (trains
     *  with `single_cls`, no class subset). */
    singleClassExport?: boolean;
  }

  let {
    datasetExportDir,
    profiles,
    presets,
    holdout = null,
    preflight,
    preflighting,
    starting,
    onPreflight,
    onStart,
    onStartCampaign,
    disabled = false,
    singleClassExport = false,
  }: Props = $props();

  // ---- Form state -------------------------------------------------------
  const SIZES: ModelSize[] = ['n', 's', 'm', 'l', 'x'];
  const PROFILE_NAMES: ProfileName[] = [
    'probe',
    'nano',
    'small',
    'medium',
    'large',
    'xlarge',
  ];
  let modelSize = $state<ModelSize>('m');
  let profileName = $state<ProfileName>('medium');
  // '' until the served options load; an omitted claim lets the backend
  // default from its allowlist.
  let cudaDevices = $state<string>('');
  let gpuOptions = $state<TrainGpuOptionsResponse | null>(null);
  let gpuError = $state<string | null>(null);
  const gpuAdvisory = $derived(
    gpuOptions?.options.find((o) => o.value === cudaDevices)?.advisory ?? null,
  );

  $effect(() => {
    const ctrl = new AbortController();
    getTrainGpus(ctrl.signal)
      .then((res) => {
        gpuOptions = res;
        if (!cudaDevices) cudaDevices = defaultGpuValue(res);
      })
      .catch((e: unknown) => {
        if ((e as Error).name !== 'AbortError') gpuError = (e as Error).message;
      });
    return () => ctrl.abort();
  });

  // Class subset selection. `null` means "all classes".
  let selectedClasses = $state<number[] | null>(null);
  let singleCls = $state<boolean>(false);
  // V-4: once the registry loads, default to the classes the server
  // doesn't report as short of data (served `trainable_gap`), so a
  // 0-crop class doesn't block preflight out of the box. Seeded once;
  // any operator change afterwards wins.
  let selectionSeeded = false;
  $effect(() => {
    const classes = classesStore.classes;
    if (selectionSeeded || classes.length === 0) return;
    selectionSeeded = true;
    if (selectedClasses === null) selectedClasses = defaultTrainSelection(classes);
  });

  // Augmentation: null means "no augmentation block".
  let augmentation = $state<AugmentationSpec | null>(null);

  // Hyperparameter overrides. Keyed by Ultralytics arg name. `null` =
  // "use the profile default". Updated when the user picks a profile.
  let hpEpochs = $state<number | null>(null);
  let hpBatch = $state<number | null>(null);
  let hpImgsz = $state<number | null>(null);
  let hpLr0 = $state<number | null>(null);
  let hpLrf = $state<number | null>(null);
  let hpMomentum = $state<number | null>(null);
  let hpWeightDecay = $state<number | null>(null);
  let hpMosaic = $state<number | null>(null);
  let hpMixup = $state<number | null>(null);
  let hpCopyPaste = $state<number | null>(null);
  let hpPatience = $state<number | null>(null);
  let hpExpanded = $state<boolean>(false);

  // Campaign mode.
  let campaignMode = $state<boolean>(false);
  let campaignSizes = $state<ModelSize[]>(['n', 's', 'm']);
  let autoPromoteBest = $state<boolean>(false);
  let stopWhenMap50 = $state<number | null>(0.9);
  // Opt-in: on successful finish, auto-export the model to all deployable
  // formats (FP16/INT8 ONNX, + CoreML on the Mac) and benchmark them.
  let autoQuantizeBakeoff = $state<boolean>(false);

  function applyProfile(p: ProfileName): void {
    profileName = p;
    const def = profiles.find((x) => x.name === p)?.defaults ?? {};
    if (
      typeof def.model_size === 'string' &&
      SIZES.includes(def.model_size as ModelSize)
    ) {
      modelSize = def.model_size as ModelSize;
    }
    hpEpochs = (def.epochs as number | undefined) ?? null;
    hpBatch = (def.batch as number | undefined) ?? null;
    hpImgsz = (def.imgsz as number | undefined) ?? null;
    hpLr0 = (def.lr0 as number | undefined) ?? null;
    hpLrf = (def.lrf as number | undefined) ?? null;
    hpMomentum = (def.momentum as number | undefined) ?? null;
    hpWeightDecay = (def.weight_decay as number | undefined) ?? null;
    hpMosaic = (def.mosaic as number | undefined) ?? null;
    hpMixup = (def.mixup as number | undefined) ?? null;
    hpCopyPaste = (def.copy_paste as number | undefined) ?? null;
    hpPatience = (def.patience as number | undefined) ?? null;
  }

  // A single-class export (nc=1) is one class. Collapse to one
  // class so the run never depends on the multi-class registry.
  $effect(() => {
    if (singleClassExport) singleCls = true;
  });

  // First time profiles arrive, seed the defaults.
  let seededProfiles = $state<boolean>(false);
  $effect(() => {
    if (!seededProfiles && profiles.length > 0) {
      applyProfile('medium');
      seededProfiles = true;
    }
  });

  // Build the spec for preflight / submit. Only set hyperparameter fields
  // the user actually overrode (anything still null is left to profile
  // defaults inside the trainer container).
  function buildHyperparameters(): Record<string, unknown> {
    const hp: Record<string, unknown> = {
      // optimizer is fixed for YOLO26 (preflight rejects 'auto').
      optimizer: 'MuSGD',
    };
    if (hpEpochs != null) hp.epochs = hpEpochs;
    if (hpBatch != null) hp.batch = hpBatch;
    if (hpImgsz != null) hp.imgsz = hpImgsz;
    if (hpLr0 != null) hp.lr0 = hpLr0;
    if (hpLrf != null) hp.lrf = hpLrf;
    if (hpMomentum != null) hp.momentum = hpMomentum;
    if (hpWeightDecay != null) hp.weight_decay = hpWeightDecay;
    if (hpMosaic != null) hp.mosaic = hpMosaic;
    if (hpMixup != null) hp.mixup = hpMixup;
    if (hpCopyPaste != null) hp.copy_paste = hpCopyPaste;
    if (hpPatience != null) hp.patience = hpPatience;
    return hp;
  }

  function buildSpec(): TrainJobSpec {
    return {
      dataset_export_dir: datasetExportDir,
      model_family: 'yolo26',
      model_size: modelSize,
      profile: profileName,
      cuda_visible_devices: cudaDevices || undefined,
      // A single-class export is already a class-0 dataset, so never filter
      // it by the multi-class registry ids — that drops every label. Send
      // no class subset for single-class runs.
      include_classes: singleClassExport ? null : selectedClasses,
      single_cls: singleClassExport ? true : singleCls,
      hyperparameters: buildHyperparameters(),
      augmentation: augmentation && augmentation.enabled ? augmentation : null,
      auto_quantize_bakeoff: autoQuantizeBakeoff,
    };
  }

  function buildCampaign(): TrainCampaignSpec {
    const runs: CampaignRunSpec[] = campaignSizes.map((size) => ({
      // For now we re-use the currently picked profile for every size —
      // a power-user can still tweak each size's run after submit by
      // editing the queued job.json. Could expand to per-size pickers
      // later if it proves useful.
      profile: profileName,
      model_size: size,
    }));
    const stopWhen =
      stopWhenMap50 != null && stopWhenMap50 > 0
        ? { map50_at_least: stopWhenMap50 }
        : null;
    return {
      dataset_export_dir: datasetExportDir,
      include_classes: singleClassExport ? null : selectedClasses,
      single_cls: singleClassExport ? true : singleCls,
      cuda_visible_devices: cudaDevices || undefined,
      augmentation: augmentation && augmentation.enabled ? augmentation : null,
      runs,
      stop_when: stopWhen,
      auto_promote_best: autoPromoteBest,
    };
  }

  // Debounced preflight: any user input nudges this; the page handles
  // the actual fetch + abort. Kicks off immediately on the first
  // mounted render so the inline panel fills in.
  let preflightDebounce: ReturnType<typeof setTimeout> | null = null;
  function schedulePreflight(): void {
    if (preflightDebounce) clearTimeout(preflightDebounce);
    preflightDebounce = setTimeout(() => {
      onPreflight(buildSpec());
    }, 350);
  }

  $effect(() => {
    // Track the field set explicitly so Svelte knows what to invalidate
    // on. (Reading buildSpec()'s output here would be opaque.)
    void datasetExportDir;
    void modelSize;
    void profileName;
    void cudaDevices;
    void selectedClasses;
    void singleCls;
    void hpEpochs;
    void hpBatch;
    void hpImgsz;
    void hpLr0;
    void hpLrf;
    void hpMomentum;
    void hpWeightDecay;
    void hpMosaic;
    void hpMixup;
    void hpCopyPaste;
    void hpPatience;
    void augmentation;
    if (!datasetExportDir) return;
    schedulePreflight();
  });

  function toggleCampaignSize(size: ModelSize): void {
    if (campaignSizes.includes(size)) {
      campaignSizes = campaignSizes.filter((s) => s !== size);
    } else {
      campaignSizes = [...campaignSizes, size];
    }
  }

  function checkSeverityClass(s: PreflightCheck['severity']): string {
    switch (s) {
      case 'block':
        return 'border-red-500/40 bg-red-500/10 text-red-200';
      case 'warn':
        return 'border-yellow-500/40 bg-yellow-500/10 text-yellow-200';
      default:
        return 'border-zinc-700 bg-zinc-950 text-zinc-300';
    }
  }

  async function submitSingle(): Promise<void> {
    await onStart(buildSpec(), false);
  }

  async function submitCampaign(): Promise<void> {
    await onStartCampaign(buildCampaign(), false);
  }

  const isBlocked = $derived(!!preflight?.blocked);
  // submit disabled = backend says blocked OR a run is active OR we're
  // mid-flight. Allow the user to still click during preflighting so
  // we don't lose a click between debounce + fetch.
  const submitDisabled = $derived(disabled || starting || isBlocked);
</script>

<section class="space-y-3">
  <!-- Model size + profile -->
  <div class="rounded-md border border-zinc-800 bg-zinc-900 p-3">
    <div class="mb-3 flex flex-wrap items-center gap-3">
      <span class="text-[11px] uppercase tracking-wide text-zinc-500">Model size</span>
      {#each SIZES as s (s)}
        <button
          type="button"
          class="rounded-md border px-2 py-1 text-sm transition {modelSize === s
            ? 'border-blue-500 bg-blue-500/15 text-white'
            : 'border-zinc-700 bg-zinc-950 text-zinc-300 hover:border-zinc-500'}"
          onclick={() => (modelSize = s)}
        >
          {s}
        </button>
      {/each}
      <span class="text-xs text-zinc-500">YOLO26 (n=fastest, x=most accurate)</span>
    </div>

    <div class="flex flex-wrap items-center gap-3">
      <span class="text-[11px] uppercase tracking-wide text-zinc-500">Profile</span>
      {#each PROFILE_NAMES as p (p)}
        {@const desc = profiles.find((x) => x.name === p)?.description ?? ''}
        <button
          type="button"
          class="rounded-md border px-2 py-1 text-sm transition {profileName === p
            ? 'border-blue-500 bg-blue-500/15 text-white'
            : 'border-zinc-700 bg-zinc-950 text-zinc-300 hover:border-zinc-500'}"
          onclick={() => applyProfile(p)}
          title={desc}
        >
          {p}
        </button>
      {/each}
    </div>
  </div>

  <!-- GPU -->
  <div class="rounded-md border border-zinc-800 bg-zinc-900 p-3">
    <span class="mb-2 block text-[11px] uppercase tracking-wide text-zinc-500">GPUs</span>
    {#if gpuError}
      <p class="text-[11px] text-red-300">GPU options unavailable: {gpuError}</p>
    {:else if !gpuOptions}
      <p class="text-[11px] text-zinc-500">Loading GPU options…</p>
    {:else if gpuOptions.unrestricted}
      <input
        type="text"
        class="input-sm w-40 font-mono"
        placeholder="e.g. 0 or 0,2"
        aria-label="CUDA visible devices"
        bind:value={cudaDevices}
      />
    {:else}
      <div class="flex flex-wrap gap-3 text-sm">
        {#each gpuOptions.options as opt (opt.value)}
          <label class="flex cursor-pointer items-center gap-2">
            <input
              type="radio"
              name="cuda-devices"
              value={opt.value}
              checked={cudaDevices === opt.value}
              onchange={() => (cudaDevices = opt.value)}
              class="accent-blue-500"
            />
            <span class="text-zinc-200">{opt.label}</span>
          </label>
        {/each}
      </div>
    {/if}
    {#if gpuAdvisory}
      <p class="mt-2 text-[11px] text-yellow-300">{gpuAdvisory}</p>
    {/if}
  </div>

  <!-- Class subset — m23 (2026-09-24 interactive pass): irrelevant for a
       single-class export (`include_classes`/`single_cls` are already
       forced above regardless of any selection here), so it showed "All
       84 classes · N validated crops" for the item registry even
       while training a single-class dataset. Replaced with a
       plain note instead of an interactive picker that has no effect. -->
  {#if singleClassExport}
    <p
      class="rounded-md border border-zinc-800 bg-zinc-900 px-3 py-2 text-xs text-zinc-400"
    >
      Single-class dataset — trains the one class the export was built for, not a
      selection from the class registry.
    </p>
  {:else}
    <ClassSubsetPicker
      classes={classesStore.classes}
      selected={selectedClasses}
      setSelected={(ids) => (selectedClasses = ids)}
      {singleCls}
      setSingleCls={(v) => (singleCls = v)}
      {presets}
      {holdout}
    />
  {/if}

  <!-- Augmentation -->
  <AugmentationPanel value={augmentation} setValue={(v) => (augmentation = v)} />

  <!-- Hyperparameters -->
  <section class="rounded-md border border-zinc-800 bg-zinc-900">
    <button
      type="button"
      class="flex w-full items-center gap-3 px-3 py-2 text-left text-sm hover:bg-zinc-800"
      onclick={() => (hpExpanded = !hpExpanded)}
      aria-expanded={hpExpanded}
    >
      <span class="font-mono text-xs text-zinc-500">{hpExpanded ? '▼' : '▶'}</span>
      <span class="font-semibold text-zinc-100">Hyperparameters</span>
      <span class="grow text-xs text-zinc-400">
        epochs {hpEpochs ?? '—'} · batch {hpBatch ?? '—'} · imgsz {hpImgsz ?? '—'}
      </span>
    </button>

    {#if hpExpanded}
      <div
        class="grid grid-cols-2 gap-3 border-t border-zinc-800 p-3 text-sm sm:grid-cols-4"
      >
        <label class="block">
          <span class="mb-1 block text-xs text-zinc-400">epochs</span>
          <input
            type="number"
            min="1"
            bind:value={hpEpochs}
            class="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
          />
        </label>
        <label class="block">
          <span class="mb-1 flex items-center justify-between text-xs text-zinc-400">
            <span>batch</span>
            <span class="flex items-center gap-1 text-[10px] text-zinc-500">
              <input
                type="checkbox"
                aria-label="batch: auto"
                checked={hpBatch === -1}
                onchange={(e) => (hpBatch = e.currentTarget.checked ? -1 : 64)}
              />
              auto
            </span>
          </span>
          <input
            type="number"
            min="-1"
            aria-label="batch"
            bind:value={hpBatch}
            disabled={hpBatch === -1}
            placeholder={hpBatch === -1 ? 'auto (−1)' : ''}
            class="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none disabled:opacity-50"
          />
        </label>
        <label class="block">
          <span class="mb-1 block text-xs text-zinc-400">imgsz</span>
          <input
            type="number"
            min="64"
            step="32"
            bind:value={hpImgsz}
            class="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
          />
        </label>
        <label class="block">
          <span class="mb-1 block text-xs text-zinc-400">optimizer</span>
          <input
            type="text"
            value="MuSGD"
            readonly
            class="w-full cursor-not-allowed rounded-md border border-zinc-800 bg-zinc-950 px-2 py-1.5 font-mono text-sm text-zinc-500"
          />
        </label>
        <label class="block">
          <span class="mb-1 block text-xs text-zinc-400">lr0</span>
          <input
            type="number"
            step="0.0001"
            bind:value={hpLr0}
            class="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
          />
        </label>
        <label class="block">
          <span class="mb-1 block text-xs text-zinc-400">lrf</span>
          <input
            type="number"
            step="0.001"
            bind:value={hpLrf}
            class="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
          />
        </label>
        <label class="block">
          <span class="mb-1 block text-xs text-zinc-400">momentum</span>
          <input
            type="number"
            step="0.001"
            bind:value={hpMomentum}
            class="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
          />
        </label>
        <label class="block">
          <span class="mb-1 block text-xs text-zinc-400">weight_decay</span>
          <input
            type="number"
            step="0.00001"
            bind:value={hpWeightDecay}
            class="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
          />
        </label>
        <label class="block">
          <span class="mb-1 block text-xs text-zinc-400">mosaic</span>
          <input
            type="number"
            step="0.001"
            min="0"
            max="1"
            bind:value={hpMosaic}
            class="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
          />
        </label>
        <label class="block">
          <span class="mb-1 block text-xs text-zinc-400">mixup</span>
          <input
            type="number"
            step="0.001"
            min="0"
            max="1"
            bind:value={hpMixup}
            class="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
          />
        </label>
        <label class="block">
          <span class="mb-1 block text-xs text-zinc-400">copy_paste</span>
          <input
            type="number"
            step="0.001"
            min="0"
            max="1"
            bind:value={hpCopyPaste}
            class="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
          />
        </label>
        <label class="block">
          <span class="mb-1 block text-xs text-zinc-400">patience</span>
          <input
            type="number"
            min="0"
            bind:value={hpPatience}
            class="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
          />
        </label>
      </div>
    {/if}
  </section>

  <!-- Preflight panel -->
  <section class="rounded-md border border-zinc-800 bg-zinc-900 p-3">
    <header class="mb-2 flex items-center gap-3">
      <h3 class="text-[11px] uppercase tracking-wide text-zinc-500">Preflight</h3>
      {#if preflighting}
        <span class="text-[11px] text-zinc-400">checking…</span>
      {:else if preflight}
        <span
          class="rounded-sm border px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide {preflight.blocked
            ? 'border-red-500/40 bg-red-500/10 text-red-200'
            : 'border-green-500/40 bg-green-500/10 text-green-200'}"
        >
          {preflight.blocked ? 'blocked' : 'ok'}
        </span>
        <span class="text-xs text-zinc-400">{preflight.summary ?? ''}</span>
      {/if}
    </header>
    {#if preflight}
      <ul class="space-y-1.5">
        {#each preflight.checks as c (c.name)}
          <li
            class="flex items-start gap-2 rounded border px-2 py-1.5 text-xs {checkSeverityClass(
              c.severity,
            )}"
          >
            <span class="font-mono text-[10px] uppercase tracking-wide opacity-70">
              {c.severity}
            </span>
            <span class="font-mono text-[10px] text-zinc-400">{c.name}</span>
            <span class="grow">
              {c.message}
              {#if c.detail && Object.keys(c.detail).length > 0}
                <details class="mt-1">
                  <summary class="cursor-pointer text-[10px] text-zinc-500"
                    >detail</summary
                  >
                  <pre
                    class="mt-1 whitespace-pre-wrap break-all font-mono text-[10px] text-zinc-400">{JSON.stringify(
                      c.detail,
                      null,
                      2,
                    )}</pre>
                </details>
              {/if}
            </span>
          </li>
        {/each}
      </ul>
    {/if}
  </section>

  <!-- Submit row -->
  <div
    class="flex flex-wrap items-center gap-3 rounded-md border border-zinc-800 bg-zinc-900 p-3"
  >
    <label class="flex cursor-pointer items-center gap-2 text-xs text-zinc-300">
      <input
        type="checkbox"
        bind:checked={campaignMode}
        class="h-4 w-4 cursor-pointer accent-blue-500"
      />
      Multi-size campaign
    </label>

    <label
      class="flex cursor-pointer items-center gap-2 text-xs text-zinc-300"
      title="On successful finish, auto-export FP16/INT8 ONNX (+ CoreML on the Mac) and benchmark size, accuracy, and speed."
    >
      <input
        type="checkbox"
        bind:checked={autoQuantizeBakeoff}
        class="h-4 w-4 cursor-pointer accent-emerald-500"
      />
      Auto-export &amp; benchmark on finish
    </label>

    {#if campaignMode}
      <span class="text-[11px] text-zinc-500">sizes</span>
      {#each SIZES as s (s)}
        <button
          type="button"
          class="rounded-md border px-2 py-0.5 text-xs {campaignSizes.includes(s)
            ? 'border-blue-500 bg-blue-500/15 text-white'
            : 'border-zinc-700 bg-zinc-950 text-zinc-400 hover:border-zinc-500'}"
          onclick={() => toggleCampaignSize(s)}
        >
          {s}
        </button>
      {/each}
      <label class="flex cursor-pointer items-center gap-2 text-xs text-zinc-300">
        <input
          type="checkbox"
          bind:checked={autoPromoteBest}
          class="h-4 w-4 cursor-pointer accent-blue-500"
        />
        auto-promote best
      </label>
      <label class="flex items-center gap-2 text-xs text-zinc-300">
        stop_when mAP50 ≥
        <input
          type="number"
          step="0.01"
          min="0"
          max="1"
          bind:value={stopWhenMap50}
          class="w-20 rounded border border-zinc-700 bg-zinc-950 px-1.5 py-0.5 font-mono text-xs text-zinc-100 focus:border-blue-500 focus:outline-none"
        />
      </label>
    {/if}

    <span class="grow"></span>

    {#if campaignMode}
      <button
        type="button"
        class="btn btn-primary"
        onclick={submitCampaign}
        disabled={submitDisabled || campaignSizes.length === 0}
      >
        {starting ? 'Submitting…' : `Start campaign (${campaignSizes.length} runs)`}
      </button>
    {:else}
      <button
        type="button"
        class="btn btn-primary"
        onclick={submitSingle}
        disabled={submitDisabled}
      >
        {starting ? 'Starting…' : 'Start training'}
      </button>
    {/if}
  </div>
</section>
