<!--
  The "Embed them" action of an empty review queue whose served reason says
  items have no vector. The request is the served `empty_state`
  `suggested_reprocess`, sent as served behind Reprocess's dry run and
  confirm. Renders nothing otherwise.
-->
<script lang="ts">
  import ReprocessControl from '$components/datasets/ReprocessControl.svelte';
  import { emptyQueueEmbedRequest, type EmptyQueueEmbedState } from './emptyQueueEmbed';

  let {
    emptyState,
    reasonText,
    onapplied,
  }: {
    emptyState: EmptyQueueEmbedState | null | undefined;
    /** The queue's served `empty_reason` and `sort_fallback_reason`. */
    reasonText: string | null | undefined;
    onapplied?: () => void;
  } = $props();

  const request = $derived(emptyQueueEmbedRequest(emptyState, reasonText));
</script>

{#if request}
  <span data-testid="empty-queue-embed">
    <ReprocessControl
      target={{ kind: 'request', request }}
      buttonClass="btn btn-sm btn-primary"
      buttonLabel="Embed them"
      {onapplied}
    />
  </span>
{/if}
