<script lang="ts">
  /**
   * Renders its content only once the backend is known to serve W3
   * (`packsAvailability`, a one-shot `GET /prompt_packs` probe). A
   * backend without W3 gets one line instead of a broken page; this is
   * the "not yet deployed" gate, not a compatibility path.
   */
  import type { Snippet } from 'svelte';
  import { packsAvailability } from '$lib/packs/packsAvailability.svelte';

  interface Props {
    children: Snippet;
  }

  let { children }: Props = $props();

  $effect(() => {
    void packsAvailability.init();
  });
</script>

{#if packsAvailability.available === true}
  {@render children()}
{:else if packsAvailability.available === false}
  <p class="text-sm text-zinc-400" data-testid="packs-unavailable">
    Prompt-pack editing isn't available on this backend.
  </p>
{:else if packsAvailability.error}
  <div class="space-y-2 text-sm">
    <p class="text-red-300">Could not load the prompt packs: {packsAvailability.error}</p>
    <button
      type="button"
      class="btn btn-sm"
      onclick={() => void packsAvailability.retry()}>Retry</button
    >
  </div>
{:else}
  <p class="text-sm text-zinc-500">Loading…</p>
{/if}
