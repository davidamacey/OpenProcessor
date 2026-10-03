<!--
  The `/settings` card linking to the open-vocabulary sets editor. Absent,
  not disabled, until the backend serves the routes (the one-shot probe).
-->
<script lang="ts">
  import { resolve } from '$app/paths';
  import { openVocabAvailability } from '$lib/openVocab/openVocabAvailability.svelte';
  import { projectHref } from '$lib/projectPaths';

  $effect(() => {
    void openVocabAvailability.init();
  });
</script>

{#if openVocabAvailability.available === true}
  <section
    class="surface flex flex-wrap items-center gap-3 p-5"
    data-testid="open-vocab-card"
  >
    <div class="flex min-w-0 flex-col gap-1">
      <h2 class="text-base font-semibold">Open-vocabulary sets</h2>
      <p class="text-xs text-zinc-400">
        Find things by describing them in words, with SAM 3. Choose the prompts, the
        classes they map to, and which set is active.
      </p>
    </div>
    <span class="grow"></span>
    <a class="btn" href={resolve(projectHref('/settings/open-vocab'))}
      >Open open-vocabulary sets</a
    >
  </section>
{/if}
