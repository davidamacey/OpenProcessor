<script lang="ts">
  /**
   * The segmenter every open-vocabulary pass needs, as the server reports it
   * (`GET /open_vocab` `segmenter{configured, reachable}`). Shown on both
   * pages; it states the served fact and never hides or disables anything
   * (a set can be prepared before a segmenter is up).
   */
  import type { SegmenterAvailability } from '$lib/types_openVocab';

  interface Props {
    segmenter: SegmenterAvailability | null;
  }

  let { segmenter }: Props = $props();

  const state = $derived(
    segmenter == null
      ? null
      : !segmenter.configured
        ? 'not_configured'
        : segmenter.reachable
          ? 'ready'
          : 'unreachable',
  );
</script>

{#if state}
  <p
    class="rounded border px-3 py-2 text-xs {state === 'ready'
      ? 'border-zinc-700 bg-zinc-900/60 text-zinc-300'
      : 'border-amber-500/40 bg-amber-500/10 text-amber-200'}"
    data-testid="segmenter-notice"
    data-state={state}
  >
    {#if state === 'ready'}
      Segmenter: configured and ready.
    {:else if state === 'unreachable'}
      Segmenter: configured but not reachable. Tests and passes fail until it answers.
    {:else}
      No segmenter is configured. Tests and passes fail until one is.
    {/if}
  </p>
{/if}
