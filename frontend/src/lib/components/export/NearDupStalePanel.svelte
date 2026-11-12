<script lang="ts">
  /** Shown when the server refused a freeze or export with 409 `near_dup_stale`. */
  import { staleReasonText } from '$lib/nearDup/staleRefusal';
  import type { NearDupRefreshController } from '$lib/nearDup/nearDupRefreshController.svelte';

  interface Props {
    refresh: NearDupRefreshController;
  }

  let { refresh }: Props = $props();
</script>

{#if refresh.stale.length > 0 || refresh.message || refresh.error}
  <div
    class="mb-3 rounded border border-warn/40 bg-warn/10 px-3 py-2 text-xs text-warn-text"
    data-testid="near-dup-stale"
  >
    {#if refresh.stale.length > 0}
      <p class="mb-1 font-semibold">
        Near-duplicate groups are stale, so a test holdout could leak across them.
      </p>
      <ul class="mb-2 list-disc pl-4">
        {#each refresh.stale as lv (lv.level)}
          <li>
            {lv.level} level: {lv.stale_reasons
              .map(staleReasonText)
              .join(', ')}{lv.pending_count ? ` (${lv.pending_count} pending)` : ''}
          </li>
        {/each}
      </ul>
      <button
        type="button"
        class="btn"
        data-testid="near-dup-refresh"
        disabled={refresh.busy}
        onclick={() => void refresh.refresh()}
      >
        {refresh.busy ? 'Updating…' : 'Update near-duplicate groups'}
      </button>
    {/if}
    {#if refresh.message}<p class="mt-1" data-testid="near-dup-message">
        {refresh.message}
      </p>{/if}
    {#if refresh.error}<p class="mt-1 text-danger-text" data-testid="near-dup-error">
        {refresh.error}
      </p>{/if}
  </div>
{/if}
