<script lang="ts">
  /**
   * Renders its content only once the backend is known to serve W10
   * (`datasetsAvailability`, a one-shot `GET /datasets/formats` probe).
   * A backend without W10 gets one line instead of a broken page; this is
   * the "not yet deployed" gate, not a compatibility path.
   */
  import type { Snippet } from 'svelte';
  import { datasetsAvailability } from '$lib/datasets/datasetsAvailability.svelte';
  import type { DatasetFormatsResponse } from '$lib/types_import';

  interface Props {
    children: Snippet<[DatasetFormatsResponse]>;
  }

  let { children }: Props = $props();

  $effect(() => {
    void datasetsAvailability.init();
  });
</script>

{#if datasetsAvailability.available === true && datasetsAvailability.formats}
  {@render children(datasetsAvailability.formats)}
{:else if datasetsAvailability.available === false}
  <p class="text-sm text-zinc-400" data-testid="datasets-unavailable">
    Dataset import isn't available on this backend.
  </p>
{:else if datasetsAvailability.error}
  <div class="space-y-2 text-sm">
    <p class="text-red-300">
      Could not load the dataset formats: {datasetsAvailability.error}
    </p>
    <button
      type="button"
      class="btn btn-sm"
      onclick={() => void datasetsAvailability.retry()}>Retry</button
    >
  </div>
{:else}
  <p class="text-sm text-zinc-500">Loading…</p>
{/if}
