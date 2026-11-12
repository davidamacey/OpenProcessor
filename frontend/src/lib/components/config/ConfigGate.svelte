<script lang="ts">
  /**
   * Renders its content only once the backend is known to serve a config
   * editor's routes (`ConfigAvailability`, a one-shot probe per project).
   * A backend without them gets one line instead of a broken page; this is
   * the "not yet deployed" gate, not a compatibility path.
   */
  import type { Snippet } from 'svelte';
  import type { ConfigAvailability } from '$lib/config/configAvailability.svelte';

  interface Props {
    store: ConfigAvailability;
    /** The one line shown when the backend doesn't serve the editor. */
    unavailableText: string;
    /** "Could not load the <what>: …" */
    what: string;
    testid: string;
    children: Snippet;
  }

  let { store, unavailableText, what, testid, children }: Props = $props();

  $effect(() => {
    void store.init();
  });
</script>

{#if store.available === true}
  {@render children()}
{:else if store.available === false}
  <p class="text-body text-fg-2" data-testid={testid}>{unavailableText}</p>
{:else if store.error}
  <div class="space-y-2 text-body">
    <p class="text-danger-text">Could not load the {what}: {store.error}</p>
    <button type="button" class="btn btn-sm" onclick={() => void store.retry()}
      >Retry</button
    >
  </div>
{:else}
  <p class="text-body text-fg-3">Loading…</p>
{/if}
