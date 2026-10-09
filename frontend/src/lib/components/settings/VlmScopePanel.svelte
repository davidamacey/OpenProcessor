<!--
  /settings: which crops the automated VLM labels, and its daily budget
  (`GET/PUT /vlm/policy`, #119). The scope is the contract enum; a scope
  shows only the knobs it reads. The panel decides nothing: the server
  validates and a refusal is shown as served.
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import { VlmPolicyEditor } from '$lib/labelConfirmation/vlmPolicyController.svelte';
  import {
    knobsFor,
    VLM_POLICY_EFFECT,
    VLM_SCOPE_COPY,
  } from '$lib/labelConfirmation/vlmScopeCopy';
  import { VLM_SCOPES, type VlmScope } from '$lib/types_labelConfirmation';

  interface Props {
    editor?: VlmPolicyEditor;
  }
  let { editor = new VlmPolicyEditor() }: Props = $props();

  onMount(() => {
    const ctl = new AbortController();
    void editor.load(ctl.signal);
    return () => ctl.abort();
  });

  const scope = $derived<VlmScope>(editor.draft.scope ?? 'all');
  const knobs = $derived(knobsFor(scope));

  function pick(next: VlmScope): void {
    editor.draft.scope = next;
    editor.touched();
  }

  function setNumber(key: 'conf_max' | 'sample_frac', e: Event): void {
    const v = (e.currentTarget as HTMLInputElement).valueAsNumber;
    if (Number.isNaN(v)) delete editor.draft[key];
    else editor.draft[key] = v;
    editor.touched();
  }

  function setInt(key: 'per_cluster' | 'max_crops_per_day', e: Event): void {
    const v = (e.currentTarget as HTMLInputElement).valueAsNumber;
    if (Number.isNaN(v)) delete editor.draft[key];
    else editor.draft[key] = v;
    editor.touched();
  }
</script>

<section class="surface flex flex-col gap-4 p-5" data-testid="vlm-scope-panel">
  <div class="flex flex-col gap-1">
    <h2 class="text-base font-semibold">VLM scope</h2>
    <p class="text-xs text-zinc-400">
      Which crops the always-on VLM worker and auto-label runs label, and how many a day.
    </p>
  </div>

  {#if editor.loading}
    <p class="text-sm text-zinc-500">Loading…</p>
  {:else if editor.loadError}
    <p class="text-sm text-red-300" data-testid="vlm-scope-load-error">
      {editor.loadError}
    </p>
  {:else}
    <fieldset class="flex flex-col gap-2">
      <legend class="sr-only">VLM scope</legend>
      {#each VLM_SCOPES as s (s)}
        <label
          class="flex cursor-pointer items-start gap-2 rounded border px-3 py-2 text-sm {scope ===
          s
            ? 'border-blue-500/60 bg-blue-500/10'
            : 'border-zinc-800 hover:border-zinc-700'}"
        >
          <input
            type="radio"
            name="vlm-scope"
            class="mt-1"
            value={s}
            checked={scope === s}
            onchange={() => pick(s)}
            data-testid="vlm-scope-{s}"
          />
          <span class="flex min-w-0 flex-col gap-0.5">
            <span class="font-medium text-zinc-100">{VLM_SCOPE_COPY[s].label}</span>
            <span class="text-xs text-zinc-400">{VLM_SCOPE_COPY[s].blurb}</span>
          </span>
        </label>
      {/each}
    </fieldset>

    {#if knobs.length > 0}
      <div class="grid grid-cols-1 gap-3 sm:grid-cols-2" data-testid="vlm-scope-knobs">
        {#if knobs.includes('conf_max')}
          <label class="flex flex-col gap-1 text-xs text-zinc-300">
            Detector confidence limit
            <input
              class="input"
              type="number"
              min="0"
              max="1"
              step="0.05"
              value={editor.draft.conf_max ?? ''}
              oninput={(e) => setNumber('conf_max', e)}
              data-testid="vlm-knob-conf_max"
            />
            <span class="text-zinc-500"
              >Crops the detector is at least this sure of are skipped.</span
            >
          </label>
        {/if}
        {#if knobs.includes('per_cluster')}
          <label class="flex flex-col gap-1 text-xs text-zinc-300">
            Representatives per cluster
            <input
              class="input"
              type="number"
              min="1"
              max="100"
              step="1"
              value={editor.draft.per_cluster ?? ''}
              oninput={(e) => setInt('per_cluster', e)}
              data-testid="vlm-knob-per_cluster"
            />
          </label>
        {/if}
        {#if knobs.includes('sample_frac')}
          <label class="flex flex-col gap-1 text-xs text-zinc-300">
            Random sample fraction
            <input
              class="input"
              type="number"
              min="0.01"
              max="1"
              step="0.05"
              value={editor.draft.sample_frac ?? ''}
              oninput={(e) => setNumber('sample_frac', e)}
              data-testid="vlm-knob-sample_frac"
            />
            <span class="text-zinc-500">1 keeps every crop the scope selects.</span>
          </label>
        {/if}
        {#if knobs.includes('max_crops_per_day')}
          <label class="flex flex-col gap-1 text-xs text-zinc-300">
            Crops per day
            <input
              class="input"
              type="number"
              min="0"
              step="100"
              value={editor.draft.max_crops_per_day ?? ''}
              oninput={(e) => setInt('max_crops_per_day', e)}
              data-testid="vlm-knob-max_crops_per_day"
            />
            <span class="text-zinc-500">0 means no limit.</span>
          </label>
        {/if}
      </div>
    {/if}

    <p class="text-xs text-zinc-500">{VLM_POLICY_EFFECT}</p>

    {#if editor.saveLines.length > 0}
      <div
        class="rounded border border-red-500/40 bg-red-500/10 px-3 py-2 text-xs text-red-200"
        data-testid="vlm-scope-save-error"
      >
        {#each editor.saveLines as line (line)}
          <p>{line}</p>
        {/each}
        {#if editor.conflict}
          <div class="mt-2 flex gap-2">
            <button type="button" class="btn" onclick={() => void editor.reload()}
              >Reload</button
            >
            <button type="button" class="btn" onclick={() => void editor.keepMyEdits()}
              >Keep my edits</button
            >
          </div>
        {/if}
      </div>
    {/if}

    <div class="flex flex-wrap items-center gap-3">
      <button
        type="button"
        class="btn btn-primary"
        disabled={!editor.dirty || editor.saving || editor.revision == null}
        onclick={() => void editor.save()}
        data-testid="vlm-scope-save"
      >
        {editor.saving ? 'Saving…' : 'Save scope'}
      </button>
      {#if editor.saved}
        <span class="text-xs text-green-300" data-testid="vlm-scope-saved">Saved.</span>
      {/if}
    </div>
  {/if}
</section>
