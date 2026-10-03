<!--
  The served cost preview of the policy draft (`POST /ingest/policy/preview`).
  It describes the detections ALREADY STORED; the policy changes only future
  ingests. Every number is the served one.
-->
<script lang="ts">
  import type { IngestPolicyEditor } from '$lib/detector/ingestPolicyController.svelte';

  let { editor }: { editor: IngestPolicyEditor } = $props();

  const p = $derived(editor.preview);
  const summary = $derived(
    p
      ? `${p.would_embed} of ${p.total_items} stored detections would be embedded, about ${p.estimated_vector_mb} MB`
      : '',
  );
  const labeledText = $derived(
    p
      ? `Of those, ${p.embedded_because_labeled} embed only because a human or validated label always embeds.`
      : '',
  );
</script>

<section class="surface space-y-2 p-4" data-testid="policy-preview">
  <h2 class="text-sm font-semibold">Cost preview</h2>
  <p class="text-xs text-zinc-500">
    For detections already stored; the policy changes only future ingests.
  </p>
  {#if editor.previewError}
    <p class="text-xs text-red-300" data-testid="policy-preview-error">
      {editor.previewError}
    </p>
  {:else if p}
    <p class="text-sm text-zinc-200" data-testid="policy-preview-summary">
      {summary}
    </p>
    <p class="text-xs text-zinc-400" data-testid="policy-preview-labeled">
      {labeledText}
    </p>
    {#if p.truncated}
      <p class="text-xs text-zinc-400" data-testid="policy-preview-truncated">
        Estimated from {p.scanned} detections.
      </p>
    {/if}
    {#if p.by_class.length > 0}
      <table class="w-full text-left text-xs">
        <thead class="text-zinc-500">
          <tr>
            <th class="pr-3">class</th>
            <th class="pr-3">would embed</th>
            <th>would not</th>
          </tr>
        </thead>
        <tbody>
          {#each p.by_class as c (c.name)}
            <tr class="border-t border-zinc-900">
              <td class="pr-3 text-zinc-200">{c.name}</td>
              <td class="pr-3 text-zinc-300">{c.would_embed}</td>
              <td class="text-zinc-400">{c.would_not_embed}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}
  {:else if editor.previewing}
    <p class="text-xs text-zinc-500">Estimating...</p>
  {/if}
</section>
