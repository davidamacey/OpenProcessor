<!--
  The `/settings` card for the ingest policy: the served embedding mode and
  a link to the editor. A failed read shows its error beside the link.
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import { resolve } from '$app/paths';
  import { getIngestPolicy, detectorErrorLines } from '$lib/api_detector';
  import { projectHref } from '$lib/projectPaths';

  let mode = $state<string | null>(null);
  let error = $state<string | null>(null);

  onMount(() => {
    const ctl = new AbortController();
    getIngestPolicy(ctl.signal)
      .then((p) => {
        mode = p.embedding?.mode ?? 'all';
      })
      .catch((e) => {
        if ((e as Error)?.name === 'AbortError') return;
        error = detectorErrorLines(e).join(' ');
      });
    return () => ctl.abort();
  });
</script>

<section
  class="surface flex flex-wrap items-center gap-3 p-5"
  data-testid="ingest-policy-card"
>
  <div class="flex min-w-0 flex-col gap-1">
    <h2 class="text-base font-semibold">Ingest policy</h2>
    <p class="text-xs text-zinc-400">
      Which detections an ingest keeps and which of those get a vector.
      {#if mode}
        <span data-testid="ingest-policy-card-mode">Embedding: {mode}.</span>
      {/if}
    </p>
    {#if error}
      <p class="text-xs text-red-300" data-testid="ingest-policy-card-error">{error}</p>
    {/if}
  </div>
  <span class="grow"></span>
  <a class="btn" href={resolve(projectHref('/settings/ingest-policy'))}
    >Open ingest policy</a
  >
</section>
