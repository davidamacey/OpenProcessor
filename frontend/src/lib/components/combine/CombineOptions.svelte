<script lang="ts">
  /**
   * Step 3 — options. Each is omitted from the request until touched, so
   * the server's own default applies; the value ids print humanized (the
   * backend serves no labels for them, plan question P4-3).
   */
  import type { CombineWizard } from '$lib/combine/combineWizardController.svelte';
  import { combineLabel } from '$lib/combine/combineText';
  import type { CombineDedupMode, CombineHoldoutMode } from '$lib/types_combine';

  interface Props {
    wizard: CombineWizard;
  }
  let { wizard }: Props = $props();

  const DEDUP: CombineDedupMode[] = ['content_hash', 'none'];
  const HOLDOUT: CombineHoldoutMode[] = ['preserve_union', 'recompute', 'none'];
</script>

<section class="space-y-3" data-testid="combine-step-options">
  <h2 class="text-sm font-semibold text-zinc-200">3. Options</h2>
  <p class="text-xs text-zinc-500">Anything you leave alone uses the server's default.</p>
  <div class="grid gap-3 sm:grid-cols-2">
    <label class="block text-sm">
      <span class="mb-1 block text-xs text-zinc-400">Duplicate images</span>
      <select
        class="w-full rounded border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-100"
        data-testid="combine-opt-dedup"
        value={wizard.options.dedup ?? ''}
        onchange={(e) => {
          const v = (e.currentTarget as HTMLSelectElement).value;
          wizard.setOption('dedup', v === '' ? undefined : (v as CombineDedupMode));
        }}
      >
        <option value="">Server default</option>
        {#each DEDUP as v (v)}<option value={v}>{combineLabel(v)}</option>{/each}
      </select>
    </label>
    <label class="block text-sm">
      <span class="mb-1 block text-xs text-zinc-400">Duplicate box overlap (IoU)</span>
      <input
        type="number"
        min="0"
        max="1"
        step="0.01"
        placeholder="Server default"
        class="w-full rounded border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-100"
        data-testid="combine-opt-iou"
        value={wizard.options.dedup_iou ?? ''}
        oninput={(e) => {
          const raw = (e.currentTarget as HTMLInputElement).value;
          wizard.setOption('dedup_iou', raw === '' ? undefined : Number(raw));
        }}
      />
    </label>
    <label class="block text-sm">
      <span class="mb-1 block text-xs text-zinc-400">Test split</span>
      <select
        class="w-full rounded border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-100"
        data-testid="combine-opt-holdout"
        value={wizard.options.holdout ?? ''}
        onchange={(e) => {
          const v = (e.currentTarget as HTMLSelectElement).value;
          wizard.setOption('holdout', v === '' ? undefined : (v as CombineHoldoutMode));
        }}
      >
        <option value="">Server default</option>
        {#each HOLDOUT as v (v)}<option value={v}>{combineLabel(v)}</option>{/each}
      </select>
    </label>
    <label class="block text-sm">
      <span class="mb-1 block text-xs text-zinc-400">Copy settings from</span>
      <select
        class="w-full rounded border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-100"
        data-testid="combine-opt-settings-from"
        value={wizard.options.settings_from ?? ''}
        onchange={(e) => {
          const v = (e.currentTarget as HTMLSelectElement).value;
          wizard.setOption('settings_from', v === '' ? undefined : v);
        }}
      >
        <option value="">None</option>
        {#each wizard.sources as s (s.project)}
          <option value={s.project}>{s.project}</option>
        {/each}
      </select>
    </label>
  </div>
</section>
