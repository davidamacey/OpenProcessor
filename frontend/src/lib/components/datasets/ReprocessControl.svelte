<script lang="ts">
  /**
   * "Reprocess…" button + dialog (any_domain_plan.md §7.12 item 6,
   * W10.13). Absent unless the backend serves W10 (the one-shot
   * `datasetsAvailability` probe). The scope and region-mode ids are the
   * contract's enums (`reprocessVocabulary.ts`); the backend serves no
   * labels or lock-rule copy for them. One crop or image applies directly
   * from the dialog; several crops run a served dry run first and apply
   * on confirm. Every count about the outcome is served.
   */
  import { focusOnMount } from '$lib/actions/focusOnMount';
  import { trapFocus } from '$lib/actions/trapFocus';
  import { datasetsAvailability } from '$lib/datasets/datasetsAvailability.svelte';
  import {
    ReprocessFlow,
    type ReprocessTarget,
  } from '$lib/datasets/reprocessController.svelte';
  import type { Crop } from '$lib/types';
  import type { ReprocessRegionMode } from '$lib/types_import';
  import {
    EMBED_PARTS,
    REGION_MODES,
    REPROCESS_SCOPES,
    reprocessLabel,
  } from '$lib/datasets/reprocessVocabulary';
  import { reprocessVocabularyStore } from '$lib/stores/reprocessVocabulary.svelte';
  import ReprocessCounts from './ReprocessCounts.svelte';

  interface Props {
    target: ReprocessTarget;
    /** The served post-write crops of a single-crop reprocess. */
    onadopt?: (crops: Crop[]) => void;
    /** After a batch apply. */
    onapplied?: () => void;
    disabled?: boolean;
    buttonClass?: string;
    /** The open button's text (default "Reprocess…"). */
    buttonLabel?: string;
  }

  let {
    target,
    onadopt,
    onapplied,
    disabled = false,
    buttonClass = 'btn btn-sm',
    buttonLabel = 'Reprocess…',
  }: Props = $props();

  $effect(() => {
    void datasetsAvailability.init();
    void reprocessVocabularyStore.init();
  });

  const scopeLabel = (id: string): string => reprocessVocabularyStore.label('scopes', id);

  const available = $derived(datasetsAvailability.available === true);

  let flow = $state<ReprocessFlow | null>(null);

  function open(): void {
    flow = new ReprocessFlow(target);
  }

  function close(): void {
    flow?.destroy();
    flow = null;
  }

  $effect(() => () => flow?.destroy());

  async function apply(): Promise<void> {
    const f = flow;
    if (!f) return;
    const crops = await f.apply();
    if (!f.result) return;
    if (crops.length > 0) onadopt?.(crops);
    if (f.isBatch) onapplied?.();
  }
</script>

{#if available}
  <button
    type="button"
    class={buttonClass}
    data-testid="reprocess-open"
    {disabled}
    onclick={open}
  >
    {buttonLabel}
  </button>
{/if}

{#if flow && available}
  {@const f = flow}
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <div
    class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
    role="dialog"
    aria-modal="true"
    aria-label="Reprocess"
    tabindex="-1"
    use:focusOnMount
    use:trapFocus={{ onEscape: close }}
    onclick={(e) => {
      if (e.target === e.currentTarget) close();
    }}
  >
    <div
      class="max-h-[90vh] w-full max-w-lg space-y-3 overflow-y-auto rounded-lg border border-zinc-800 bg-zinc-950 p-5 text-sm shadow-2xl"
    >
      <h3 class="text-base font-semibold text-zinc-100">
        {#if f.target.kind === 'image'}
          Reprocess image
        {:else if f.target.kind === 'request'}
          Reprocess
        {:else}
          Reprocess {f.count.toLocaleString()}
          {f.count === 1 ? 'item' : 'items'}
        {/if}
      </h3>

      <fieldset class="space-y-1">
        {#each REPROCESS_SCOPES as id (id)}
          <label
            class="flex items-start gap-2"
            title={reprocessVocabularyStore.description('scopes', id) ?? undefined}
          >
            <input
              type="checkbox"
              checked={f.scopes.includes(id)}
              disabled={f.busy || f.result != null}
              onchange={(e) =>
                f.toggleScope(id, (e.currentTarget as HTMLInputElement).checked)}
            />
            <span class="text-zinc-200">{scopeLabel(id)}</span>
          </label>
        {/each}
      </fieldset>

      {#if f.scopes.includes('region')}
        <label class="block text-xs">
          <span class="mb-0.5 block text-zinc-400">Region mode</span>
          <select
            class="select select-sm"
            value={f.regionMode}
            disabled={f.busy || f.result != null}
            onchange={(e) =>
              f.setRegionMode(
                (e.currentTarget as HTMLSelectElement).value as ReprocessRegionMode | '',
              )}
          >
            <option value="">Server default</option>
            {#each REGION_MODES as m (m)}
              <option value={m}>{reprocessLabel(m)}</option>
            {/each}
          </select>
        </label>
      {/if}

      {#if f.embedOptionsEditable}
        <fieldset class="space-y-1 text-xs" data-testid="reprocess-embed-options">
          <legend class="mb-0.5 text-zinc-400">Embed options</legend>
          <label class="flex items-start gap-2">
            <input
              type="checkbox"
              checked={f.embedOnlyMissing === true}
              disabled={f.busy || f.result != null}
              onchange={(e) =>
                f.setEmbedOnlyMissing((e.currentTarget as HTMLInputElement).checked)}
            />
            <span class="text-zinc-200">Only items without a vector</span>
          </label>
          <div class="flex flex-wrap gap-x-3 gap-y-1">
            <span class="text-zinc-400">Parts</span>
            {#each EMBED_PARTS as part (part)}
              <label class="flex items-center gap-1">
                <input
                  type="checkbox"
                  checked={f.embedParts?.includes(part) ?? false}
                  disabled={f.busy || f.result != null}
                  onchange={(e) =>
                    f.toggleEmbedPart(
                      part,
                      (e.currentTarget as HTMLInputElement).checked,
                    )}
                />
                <span class="text-zinc-200">{reprocessLabel(part)}</span>
              </label>
            {/each}
          </div>
        </fieldset>
      {/if}

      {#if f.dryRun && !f.result}
        <div class="space-y-1" data-testid="reprocess-dry-run">
          <ReprocessCounts res={f.dryRun} {scopeLabel} />
        </div>
      {/if}
      {#if f.result}
        <div class="space-y-1" data-testid="reprocess-result">
          <ReprocessCounts res={f.result} {scopeLabel} />
          {#if f.job}
            <p class="text-xs text-zinc-300" data-testid="reprocess-job">
              Job <code class="font-mono">{f.job.job_id}</code>:
              {datasetsAvailability.statusLabel(f.job.status)}
              {#if f.job.images_total}
                · {(f.job.images_done ?? 0).toLocaleString()} / {f.job.images_total.toLocaleString()}
                images{#if f.job.images_failed}, {f.job.images_failed.toLocaleString()} failed{/if}
              {/if}
              {#if f.job.error}<span class="text-red-300"> — {f.job.error}</span>{/if}
            </p>
          {/if}
        </div>
      {/if}
      {#if f.error}
        <p class="text-sm text-red-300" data-testid="reprocess-error">{f.error}</p>
      {/if}

      <div class="flex justify-end gap-2">
        {#if f.job && f.job.poll_after_s != null}
          <button type="button" class="btn" onclick={() => void f.cancelJob()}
            >Cancel job</button
          >
        {/if}
        <button type="button" class="btn" onclick={close}
          >{f.result ? 'Close' : 'Cancel'}</button
        >
        {#if !f.result}
          {#if f.isBatch && !f.dryRun}
            <button
              type="button"
              class="btn btn-primary"
              disabled={f.scopes.length === 0 || f.busy}
              onclick={() => void f.preview()}
            >
              {f.busy ? 'Checking…' : 'Check what would run'}
            </button>
          {:else}
            <button
              type="button"
              class="btn btn-primary"
              disabled={!f.canApply}
              onclick={() => void apply()}
            >
              {f.busy ? 'Working…' : 'Reprocess'}
            </button>
          {/if}
        {/if}
      </div>
    </div>
  </div>
{/if}
