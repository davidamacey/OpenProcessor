<script lang="ts">
  import { apiErrorText } from '$lib/api';
  /**
   * Training form. Decomposed sub-pieces (`ClassSubsetPicker`,
   * `AugmentationPanel`, `TrainHyperparams`, `TrainPreflightPanel`) own
   * their own UI; this component owns:
   *   - model size + profile picker
   *   - GPU selector
   *   - the hyperparameter override values
   *   - submit row + multi-size campaign mode
   *
   * The request bodies are built by `$lib/train/trainSpec`. State flows up
   * to `/train/+page.svelte` via the `submit` callbacks so the page is the
   * single source of truth for in-flight runs.
   */
  import AugmentationPanel from '$components/AugmentationPanel.svelte';
  import ClassSubsetPicker from '$components/ClassSubsetPicker.svelte';
  import TrainHyperparams from '$components/train/TrainHyperparams.svelte';
  import TrainPreflightPanel from '$components/train/TrainPreflightPanel.svelte';
  import { classesStore } from '$stores/classes.svelte';
  import { defaultTrainSelection } from '$lib/trainClassSelection';
  import { defaultGpuValue, getTrainGpus, type TrainGpuOptionsResponse } from '$lib/api';
  import {
    buildCampaign,
    buildSpec,
    profileHyperparams,
    PROFILE_NAMES,
    SIZES,
    type HyperparamOverrides,
    type TrainFormValues,
  } from '$lib/train/trainSpec';
  import type { TestHoldoutStats } from '$lib/types';
  import type {
    AugmentationSpec,
    ClassSubsetPreset,
    ModelSize,
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
        if ((e as Error).name !== 'AbortError') gpuError = apiErrorText(e);
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
  let hp = $state<HyperparamOverrides>(profileHyperparams({}));

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
    hp = profileHyperparams(def);
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

  function formValues(): TrainFormValues {
    return {
      datasetExportDir,
      modelSize,
      profileName,
      cudaDevices,
      selectedClasses,
      singleCls,
      singleClassExport,
      hp,
      augmentation,
      autoQuantizeBakeoff,
      campaignSizes,
      autoPromoteBest,
      stopWhenMap50,
    };
  }

  // Debounced preflight: any user input nudges this; the page handles
  // the actual fetch + abort. Kicks off immediately on the first
  // mounted render so the inline panel fills in.
  let preflightDebounce: ReturnType<typeof setTimeout> | null = null;
  function schedulePreflight(): void {
    if (preflightDebounce) clearTimeout(preflightDebounce);
    preflightDebounce = setTimeout(() => {
      onPreflight(buildSpec(formValues()));
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
    void hp.epochs;
    void hp.batch;
    void hp.imgsz;
    void hp.lr0;
    void hp.lrf;
    void hp.momentum;
    void hp.weight_decay;
    void hp.mosaic;
    void hp.mixup;
    void hp.copy_paste;
    void hp.patience;
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

  async function submitSingle(): Promise<void> {
    await onStart(buildSpec(formValues()), false);
  }

  async function submitCampaign(): Promise<void> {
    await onStartCampaign(buildCampaign(formValues()), false);
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
  <TrainHyperparams bind:hp />

  <TrainPreflightPanel {preflight} {preflighting} />

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
