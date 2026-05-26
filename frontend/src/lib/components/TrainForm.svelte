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
    /** True when the selected dataset is the single-class LPR export. */
    lpr?: boolean;
  }

  let {
    datasetExportDir,
    profiles,
    presets,
    preflight,
    preflighting,
    starting,
    onPreflight,
    onStart,
    onStartCampaign,
    disabled = false,
    lpr = false,
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
  const GPU_OPTIONS = [
    { value: '0,2', label: '2× A6000 (slots 0, 2)', warn: false },
    { value: '0', label: '1× A6000 (slot 0)', warn: true },
  ] as const;

  let modelSize = $state<ModelSize>('m');
  let profileName = $state<ProfileName>('medium');
  let cudaDevices = $state<string>('0,2');

  // Class subset selection. `null` means "all classes".
  let selectedClasses = $state<number[] | null>(null);
  let singleCls = $state<boolean>(false);

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

  function applyProfile(p: ProfileName): void {
    profileName = p;
    const def = profiles.find((x) => x.name === p)?.defaults ?? {};
    if (typeof def.model_size === 'string' && SIZES.includes(def.model_size as ModelSize)) {
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

  // The LPR export is single-class (nc=1, license_plate). Collapse to one
  // class so the run never depends on the multi-class registry.
  $effect(() => {
    if (lpr) singleCls = true;
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
      cuda_visible_devices: cudaDevices,
      // The LPR export is already a single-class (class 0) dataset, so never
      // filter it by the multi-class registry ids (e.g. license_plate=80) —
      // that drops every label. Send no class subset for LPR runs.
      include_classes: lpr ? null : selectedClasses,
      single_cls: lpr ? true : singleCls,
      hyperparameters: buildHyperparameters(),
      augmentation: augmentation && augmentation.enabled ? augmentation : null,
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
      include_classes: lpr ? null : selectedClasses,
      single_cls: lpr ? true : singleCls,
      cuda_visible_devices: cudaDevices,
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
    <div class="flex flex-wrap gap-3 text-sm">
      {#each GPU_OPTIONS as opt (opt.value)}
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
    {#if cudaDevices === '0'}
      <p class="mt-2 text-[11px] text-yellow-300">
        Gemma worker stays alive on GPU 2.
      </p>
    {/if}
  </div>

  <!-- Class subset -->
  <ClassSubsetPicker
    classes={classesStore.classes}
    selected={selectedClasses}
    setSelected={(ids) => (selectedClasses = ids)}
    singleCls={singleCls}
    setSingleCls={(v) => (singleCls = v)}
    presets={presets}
  />

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
      <div class="grid grid-cols-2 gap-3 border-t border-zinc-800 p-3 text-sm sm:grid-cols-4">
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
                checked={hpBatch === -1}
                onchange={(e) => (hpBatch = e.currentTarget.checked ? -1 : 64)}
              />
              auto
            </span>
          </span>
          <input
            type="number"
            min="-1"
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
            <span class="grow">{c.message}</span>
          </li>
        {/each}
      </ul>
    {/if}
  </section>

  <!-- Submit row -->
  <div class="flex flex-wrap items-center gap-3 rounded-md border border-zinc-800 bg-zinc-900 p-3">
    <label class="flex cursor-pointer items-center gap-2 text-xs text-zinc-300">
      <input
        type="checkbox"
        bind:checked={campaignMode}
        class="h-4 w-4 cursor-pointer accent-blue-500"
      />
      Multi-size campaign
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
