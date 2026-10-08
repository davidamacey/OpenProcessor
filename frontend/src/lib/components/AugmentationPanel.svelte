<script lang="ts">
  import { apiErrorText } from '$lib/api';
  /**
   * Augmentation config panel (design §13).
   *
   * Two oversampling strategies:
   *   - per_class_multiplier (manual table)  — `mode='manual'`
   *   - auto_balance (target_count + max_multiplier) — `mode='auto'`
   *
   * Parent owns `value: AugmentationSpec | null`. We mutate it via a
   * setter so reactivity updates clearly cross the component boundary.
   */
  import { getAugmentationPresets } from '$lib/api';
  import type { AugmentationPresetsResponse, AugmentationSpec } from '$lib/types_train';

  interface Props {
    value: AugmentationSpec | null;
    setValue: (v: AugmentationSpec | null) => void;
  }

  let { value, setValue }: Props = $props();

  // Served from `GET {API_PREFIX}/train/augmentation_presets`
  // (OpenProcessor df01309) — the trainer's own catalog
  // (`docker/trainer/augment.py` builds its `PRESETS` from the same
  // ids), so the picker can never offer an id the trainer will reject.
  // `null` while loading; `presetsError` when the read failed.
  let presetsResponse = $state<AugmentationPresetsResponse | null>(null);
  let presetsError = $state<string | null>(null);

  $effect(() => {
    const ctrl = new AbortController();
    getAugmentationPresets(ctrl.signal)
      .then((res) => {
        presetsResponse = res;
      })
      .catch((e: unknown) => {
        if ((e as Error).name === 'AbortError') return;
        presetsError = apiErrorText(e) ?? 'failed to load presets';
      });
    return () => ctrl.abort();
  });

  // Unset until the served list loads; a spec without `preset` gets the
  // backend's own default.
  const servedDefault = $derived(presetsResponse?.default);
  const selectedPresetOption = $derived(
    presetsResponse?.presets.find((p) => p.id === (value?.preset ?? servedDefault)) ??
      null,
  );

  let expanded = $state<boolean>(false);

  // Default spec when the user enables augmentation for the first time.
  function defaultSpec(): AugmentationSpec {
    return {
      enabled: true,
      multiplier: 3,
      preset: servedDefault,
      albumentations: {},
      per_class_multiplier: {},
    };
  }

  const spec = $derived<AugmentationSpec>(value ?? { enabled: false });
  const enabled = $derived(!!spec.enabled);

  // Oversampling mode is derived from the spec but tracked in local
  // state once the user picks one — otherwise toggling between manual
  // and auto would lose their inputs. Initial value snapshotted from
  // the prop on first render via untrack so Svelte doesn't warn about
  // capturing a reactive prop in a $state initializer.
  function deriveInitialMode(): 'off' | 'manual' | 'auto' {
    const v = value;
    if (v?.auto_balance) return 'auto';
    if (Object.keys(v?.per_class_multiplier ?? {}).length > 0) return 'manual';
    return 'off';
  }
  let oversampleMode = $state<'off' | 'manual' | 'auto'>(deriveInitialMode());

  function update(patch: Partial<AugmentationSpec>): void {
    const base = value ?? defaultSpec();
    setValue({ ...base, ...patch });
  }

  function setEnabled(v: boolean): void {
    if (v) {
      setValue({ ...defaultSpec(), ...(value ?? {}), enabled: true });
    } else if (value) {
      setValue({ ...value, enabled: false });
    }
  }

  function setMultiplier(n: number): void {
    update({ multiplier: Math.max(1, Math.min(10, Math.round(n))) });
  }

  function setPreset(p: string): void {
    update({ preset: p });
  }

  function setOversampleMode(mode: 'off' | 'manual' | 'auto'): void {
    oversampleMode = mode;
    if (mode === 'off') {
      const next: AugmentationSpec = { ...(value ?? defaultSpec()) };
      delete next.auto_balance;
      next.per_class_multiplier = {};
      setValue(next);
    } else if (mode === 'auto') {
      const next: AugmentationSpec = {
        ...(value ?? defaultSpec()),
        auto_balance: value?.auto_balance ?? { target_count: 3000, max_multiplier: 10 },
      };
      next.per_class_multiplier = {};
      setValue(next);
    } else {
      // manual
      const next: AugmentationSpec = { ...(value ?? defaultSpec()) };
      delete next.auto_balance;
      if (Object.keys(next.per_class_multiplier ?? {}).length === 0) {
        next.per_class_multiplier = {};
      }
      setValue(next);
    }
  }

  function setAutoBalance(
    patch: Partial<{ target_count: number; max_multiplier: number }>,
  ): void {
    const base = value ?? defaultSpec();
    const ab = {
      target_count: 3000,
      max_multiplier: 10,
      ...(base.auto_balance ?? {}),
      ...patch,
    };
    setValue({ ...base, auto_balance: ab });
  }

  // Manual table: user types `class_id:N` lines, easier than building
  // a per-class spreadsheet for what's an advanced-mode knob. Snapshot
  // the initial prop value via a helper so the $state initializer
  // doesn't read a reactive prop directly.
  function deriveInitialManualText(): string {
    return Object.entries(value?.per_class_multiplier ?? {})
      .map(([k, v]) => `${k}:${v}`)
      .join('\n');
  }
  let manualText = $state<string>(deriveInitialManualText());

  function commitManual(): void {
    const out: Record<string, number> = {};
    for (const raw of manualText.split('\n')) {
      const line = raw.trim();
      if (!line) continue;
      const [k, v] = line.split(':').map((s) => s.trim());
      const n = Number(v);
      if (!k || !Number.isFinite(n) || n <= 0) continue;
      out[k] = Math.round(n);
    }
    update({ per_class_multiplier: out });
  }
</script>

<section class="rounded-md border border-zinc-800 bg-zinc-900">
  <button
    type="button"
    class="flex w-full items-center gap-3 px-3 py-2 text-left text-sm hover:bg-zinc-800"
    onclick={() => (expanded = !expanded)}
    aria-expanded={expanded}
  >
    <span class="font-mono text-xs text-zinc-500">{expanded ? '▼' : '▶'}</span>
    <span class="font-semibold text-zinc-100">Augmentation</span>
    <span class="grow text-xs text-zinc-400">
      {#if !enabled}
        disabled
      {:else}
        {spec.preset ?? servedDefault ?? 'default preset'} · ×{spec.multiplier ?? 1}
        {#if oversampleMode !== 'off'}· {oversampleMode} oversample{/if}
      {/if}
    </span>
  </button>

  {#if expanded}
    <div class="space-y-3 border-t border-zinc-800 p-3 text-sm">
      <label class="flex items-center gap-2 text-zinc-200">
        <input
          type="checkbox"
          checked={enabled}
          onchange={(e) => setEnabled((e.currentTarget as HTMLInputElement).checked)}
          class="h-4 w-4 cursor-pointer accent-blue-500"
        />
        Enable on-disk augmentation (Albumentations)
      </label>

      {#if enabled}
        <div class="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <label class="block">
            <span class="mb-1 block text-xs text-zinc-400">Preset</span>
            {#if presetsResponse}
              <select
                value={spec.preset ?? servedDefault}
                onchange={(e) => setPreset((e.currentTarget as HTMLSelectElement).value)}
                class="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
              >
                {#each presetsResponse.presets as p (p.id)}
                  <option value={p.id} title={p.description}>
                    {p.label}{p.orientation_sensitive ? ' (no h-flip)' : ''}
                  </option>
                {/each}
              </select>
              {#if selectedPresetOption}
                <p
                  class="mt-1 text-[11px] text-zinc-500"
                  title={selectedPresetOption.description}
                >
                  {selectedPresetOption.description}
                  {#if selectedPresetOption.orientation_sensitive}
                    · horizontal flip disabled for this preset
                  {/if}
                </p>
              {/if}
            {:else if presetsError}
              <p class="text-[11px] text-red-300">
                Could not load presets: {presetsError}
              </p>
            {:else}
              <p class="text-[11px] text-zinc-500">Loading presets…</p>
            {/if}
          </label>

          <label class="block">
            <span class="mb-1 block text-xs text-zinc-400">
              Multiplier <span class="font-mono text-zinc-500"
                >×{spec.multiplier ?? 1}</span
              >
            </span>
            <input
              type="range"
              min="1"
              max="10"
              step="1"
              value={spec.multiplier ?? 1}
              oninput={(e) =>
                setMultiplier(Number((e.currentTarget as HTMLInputElement).value))}
              class="w-full accent-blue-500"
            />
          </label>
        </div>

        <fieldset class="rounded border border-zinc-800 p-3">
          <legend class="px-1 text-[11px] uppercase tracking-wide text-zinc-500">
            Per-class oversampling
          </legend>
          <div class="mb-2 flex gap-3 text-xs text-zinc-300">
            <label class="flex cursor-pointer items-center gap-1.5">
              <input
                type="radio"
                name="oversample"
                checked={oversampleMode === 'off'}
                onchange={() => setOversampleMode('off')}
                class="accent-blue-500"
              />
              off
            </label>
            <label class="flex cursor-pointer items-center gap-1.5">
              <input
                type="radio"
                name="oversample"
                checked={oversampleMode === 'manual'}
                onchange={() => setOversampleMode('manual')}
                class="accent-blue-500"
              />
              manual table
            </label>
            <label class="flex cursor-pointer items-center gap-1.5">
              <input
                type="radio"
                name="oversample"
                checked={oversampleMode === 'auto'}
                onchange={() => setOversampleMode('auto')}
                class="accent-blue-500"
              />
              auto-balance
            </label>
          </div>

          {#if oversampleMode === 'manual'}
            <label class="block">
              <span class="mb-1 block text-[11px] text-zinc-500">
                One <code>class_id:multiplier</code> per line. Blanks ignored.
              </span>
              <textarea
                bind:value={manualText}
                onblur={commitManual}
                rows="4"
                class="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 font-mono text-xs text-zinc-100 focus:border-blue-500 focus:outline-none"
                placeholder="81:8
47:5"></textarea>
            </label>
          {:else if oversampleMode === 'auto'}
            <div class="grid grid-cols-1 gap-3 sm:grid-cols-2">
              <label class="block">
                <span class="mb-1 block text-xs text-zinc-400"
                  >Target count per class</span
                >
                <input
                  type="number"
                  min="100"
                  step="100"
                  value={spec.auto_balance?.target_count ?? 3000}
                  oninput={(e) =>
                    setAutoBalance({
                      target_count: Number((e.currentTarget as HTMLInputElement).value),
                    })}
                  class="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
                />
              </label>
              <label class="block">
                <span class="mb-1 block text-xs text-zinc-400">Max multiplier (cap)</span>
                <input
                  type="number"
                  min="1"
                  max="20"
                  value={spec.auto_balance?.max_multiplier ?? 10}
                  oninput={(e) =>
                    setAutoBalance({
                      max_multiplier: Number((e.currentTarget as HTMLInputElement).value),
                    })}
                  class="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
                />
              </label>
            </div>
          {/if}
        </fieldset>
      {/if}
    </div>
  {/if}
</section>
