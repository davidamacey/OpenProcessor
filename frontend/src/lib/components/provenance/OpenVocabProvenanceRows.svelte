<!--
  v0.4.0 open-vocabulary provenance rows for `CropMetaPanel`'s main <dl>:
  the prompt that found the item, the set@revision that ran it (linked to
  its editor while the open-vocabulary gate is open), why the region stage
  skipped it, and a link to the other items from the same set or prompt.
  Renders <dt>/<dd> pairs, or nothing. Every value is the served one, and a
  row appears only when it carries a value.
-->
<script lang="ts">
  import { resolve } from '$app/paths';
  import { SvelteURLSearchParams } from 'svelte/reactivity';
  import { openVocabAvailability } from '$lib/openVocab/openVocabAvailability.svelte';
  import { projectHref } from '$lib/projectPaths';
  import type { Crop } from '$lib/types';

  let { crop }: { crop: Crop } = $props();

  const set = $derived(crop.open_vocab_set ?? null);
  const prompt = $derived(crop.source_prompt ?? null);
  const gateSkip = $derived(crop.region_gate_skip ?? null);

  // Only an item that names a set needs to know whether the editor exists.
  $effect(() => {
    if (set != null) void openVocabAvailability.init();
  });
  const linkable = $derived(openVocabAvailability.available === true);

  const matchingQuery = $derived.by(() => {
    const p = new SvelteURLSearchParams({ mode: 'matching' });
    if (set != null) p.set('open_vocab_set', set);
    if (prompt != null) p.set('source_prompt', prompt);
    return p.toString();
  });
</script>

{#if prompt != null}
  <dt class="text-zinc-500">Prompt</dt>
  <dd class="text-zinc-200" data-testid="ov-prompt">{prompt}</dd>
{/if}
{#if set != null}
  <dt class="text-zinc-500">Open-vocabulary set</dt>
  <dd class="font-mono text-zinc-200" data-testid="ov-set">
    {#if linkable}
      <a
        class="text-blue-300 underline hover:text-blue-200"
        href={resolve(projectHref(`/settings/open-vocab/${encodeURIComponent(set)}`))}
        >{set}{crop.open_vocab_revision != null ? `@${crop.open_vocab_revision}` : ''}</a
      >
    {:else}
      {set}{crop.open_vocab_revision != null ? `@${crop.open_vocab_revision}` : ''}
    {/if}
  </dd>
{/if}
{#if gateSkip != null}
  <dt class="text-zinc-500">Region stage skipped</dt>
  <dd class="text-amber-200" data-testid="ov-gate-skip">{gateSkip}</dd>
{/if}
{#if set != null || prompt != null}
  <dt class="text-zinc-500">Same origin</dt>
  <dd>
    <a
      class="text-blue-300 underline hover:text-blue-200"
      data-testid="ov-matching-link"
      href={resolve(projectHref(`/clusters?${matchingQuery}`))}>Show matching items</a
    >
  </dd>
{/if}
