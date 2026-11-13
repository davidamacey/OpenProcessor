<script lang="ts">
  /**
   * Renders its content only once the backend is known to serve dataset import
   * (`datasetsAvailability`, a one-shot `GET /datasets/formats` probe).
   * A backend without dataset import gets one line instead of a broken page; this is
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
  <p class="text-body text-fg-2" data-testid="datasets-unavailable">
    Dataset import isn't available on this backend.
  </p>
{:else if datasetsAvailability.error}
  <div class="space-y-2 text-body">
    <p class="text-danger-text">
      Could not load the dataset formats: {datasetsAvailability.error}
    </p>
    <button
      type="button"
      class="btn btn-sm"
      onclick={() => void datasetsAvailability.retry()}>Retry</button
    >
  </div>
{:else}
  <p class="text-body text-fg-3">Loading…</p>
{/if}
