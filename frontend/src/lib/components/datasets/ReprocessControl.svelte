<script lang="ts">
  /**
   * "Reprocess…" button + dialog (any_domain_plan.md §7.12 item 6,
   * W10.13). Absent unless the backend serves W10 and its Reprocess
   * vocabulary (`/datasets/formats` `reprocess`: scopes, region modes and
   * the lock-rule copy, delta 20). One crop applies directly from the
   * dialog; several crops run a served dry run first and apply on
   * confirm. Every count and sentence about the outcome is served.
   */
  import { focusOnMount } from '$lib/actions/focusOnMount';
  import { trapFocus } from '$lib/actions/trapFocus';
  import { datasetsAvailability } from '$lib/datasets/datasetsAvailability.svelte';
  import {
    ReprocessFlow,
    type ReprocessTarget,
  } from '$lib/datasets/reprocessController.svelte';
  import type { Crop } from '$lib/types';
  import ReprocessCounts from './ReprocessCounts.svelte';

  interface Props {
    target: ReprocessTarget;
    /** The served post-write crops of a single-crop reprocess. */
    onadopt?: (crops: Crop[]) => void;
    /** After a batch apply. */
    onapplied?: () => void;
    disabled?: boolean;
    buttonClass?: string;
  }

  let {
    target,
    onadopt,
    onapplied,
    disabled = false,
    buttonClass = 'btn btn-sm',
  }: Props = $props();

  $effect(() => {
    void datasetsAvailability.init();
  });

  const vocab = $derived(
    datasetsAvailability.available === true
      ? (datasetsAvailability.formats?.reprocess ?? null)
      : null,
  );

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

  const scopeLabel = (id: string): string =>
    vocab?.scopes.find((s) => s.id === id)?.label ?? id;
</script>

{#if vocab}
  <button
    type="button"
    class={buttonClass}
    data-testid="reprocess-open"
    {disabled}
    onclick={open}
  >
    Reprocess…
  </button>
{/if}

{#if flow && vocab}
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
        Reprocess {f.count.toLocaleString()}
        {f.count === 1 ? 'item' : 'items'}
      </h3>
      <p class="text-xs text-zinc-400" data-testid="reprocess-lock-rule">
        {vocab.lock_rule}
      </p>

      <fieldset class="space-y-1">
        {#each vocab.scopes as s (s.id)}
          <label class="flex items-start gap-2">
            <input
              type="checkbox"
              checked={f.scopes.includes(s.id)}
              disabled={f.busy || f.result != null}
              onchange={(e) =>
                f.toggleScope(s.id, (e.currentTarget as HTMLInputElement).checked)}
            />
            <span>
              <span class="text-zinc-200">{s.label}</span>
              {#if s.description}
                <span class="block text-xs text-zinc-500">{s.description}</span>
              {/if}
            </span>
          </label>
        {/each}
      </fieldset>

      {#if f.scopes.includes('region') && vocab.region_modes.length > 0}
        <label class="block text-xs">
          <span class="mb-0.5 block text-zinc-400">Region mode</span>
          <select
            class="select select-sm"
            value={f.regionMode}
            disabled={f.busy || f.result != null}
            onchange={(e) =>
              f.setRegionMode((e.currentTarget as HTMLSelectElement).value)}
          >
            <option value="">Server default</option>
            {#each vocab.region_modes as m (m.id)}
              <option value={m.id}>{m.label}</option>
            {/each}
          </select>
        </label>
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
              {f.job.labels?.status?.[f.job.status] ?? f.job.status}
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
