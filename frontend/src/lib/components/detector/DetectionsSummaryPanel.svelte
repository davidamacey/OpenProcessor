<!--
  The served detections summary on `/dashboard`: totals, the embedding
  breakdown and a per-label table. "Embed N detections" appears only when
  the summary serves a `suggested_reprocess`, and opens the Reprocess dialog
  with that request exactly as served (dry run first, apply behind a
  confirm).
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import ReprocessControl from '$components/datasets/ReprocessControl.svelte';
  import { embeddingStateChips } from '$components/embedding/embeddingCopy';
  import { DetectionsSummaryState } from '$lib/detector/detectionsSummaryController.svelte';

  let {
    state: summaryState = new DetectionsSummaryState(),
  }: { state?: DetectionsSummaryState } = $props();

  onMount(() => {
    const ctl = new AbortController();
    void summaryState.load(ctl.signal);
    return () => ctl.abort();
  });

  const s = $derived(summaryState.summary);
  const fmt = (n: number) => n.toLocaleString();
</script>

<section class="surface space-y-3 p-4" data-testid="detections-summary">
  <div class="flex items-center gap-3">
    <h2 class="text-sm font-semibold text-zinc-300">Detections</h2>
    <span class="grow"></span>
    <button
      type="button"
      class="btn btn-sm"
      disabled={summaryState.loading}
      onclick={() => void summaryState.load()}>Refresh</button
    >
  </div>

  {#if summaryState.error}
    <p class="text-xs text-red-300" data-testid="detections-error">
      {summaryState.error}
    </p>
  {:else if !s}
    <p class="text-xs text-zinc-500">Loading...</p>
  {:else}
    <div class="flex flex-wrap items-center gap-3 text-xs">
      <span class="font-mono text-zinc-200" data-testid="detections-total"
        >{fmt(s.total)} detections</span
      >
      <span class="text-zinc-400"
        >embedded {fmt(s.embedding.embedded)} · not embedded {fmt(
          s.embedding.not_embedded,
        )}</span
      >
      {#each embeddingStateChips(s.embedding.by_state as Record<string, number>) as chip (chip.state)}
        <span class="chip text-[10px]" title={chip.title}
          >{chip.label} {fmt(chip.count)}</span
        >
      {/each}
      {#if s.suggested_reprocess}
        <ReprocessControl
          target={{ kind: 'request', request: s.suggested_reprocess }}
          buttonClass="btn btn-sm btn-primary"
          buttonLabel="Embed {fmt(s.embedding.not_embedded)} detections"
          onapplied={() => void summaryState.load()}
        />
      {/if}
    </div>

    {#if s.by_label.length > 0}
      <table class="w-full text-left text-xs" data-testid="detections-by-label">
        <thead class="text-zinc-500">
          <tr>
            <th class="py-0.5 pr-3 font-normal">Label</th>
            <th class="py-0.5 pr-3 text-right font-normal">Detections</th>
            <th class="py-0.5 pr-3 text-right font-normal">Embedded</th>
            <th class="py-0.5 text-right font-normal">Not embedded</th>
          </tr>
        </thead>
        <tbody class="font-mono">
          {#each s.by_label as l (l.name)}
            <tr class="border-t border-zinc-800">
              <td class="py-0.5 pr-3 font-sans text-zinc-200">{l.name}</td>
              <td class="py-0.5 pr-3 text-right">{fmt(l.count)}</td>
              <td class="py-0.5 pr-3 text-right">{fmt(l.embedding.embedded)}</td>
              <td class="py-0.5 text-right">{fmt(l.embedding.not_embedded)}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}
    {#if s.labels_truncated}
      <p class="text-xs text-zinc-500" data-testid="detections-truncated">
        The label list is truncated; only the labels with the most detections are shown.
      </p>
    {/if}
  {/if}
</section>
