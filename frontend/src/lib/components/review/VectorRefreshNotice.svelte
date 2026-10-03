<script lang="ts">
  /**
   * After a region write: how many of the touched boxes still have no
   * vector (`vector_refresh.pending`, served; the encoder was down, the
   * image unreadable or the encoder raised). The count is the server's; the
   * remedy is a Reprocess dialog for this item seeded with the `embed`
   * scope, missing vectors only (dry run first, apply on confirm). Absent
   * when nothing is pending.
   */
  import ReprocessControl from '$lib/components/datasets/ReprocessControl.svelte';
  import type { ReprocessTarget } from '$lib/datasets/reprocessController.svelte';
  import type { VectorRefresh } from '$lib/types_itemFilter';

  interface Props {
    cropId: string;
    refresh: VectorRefresh | null;
  }

  let { cropId, refresh }: Props = $props();

  const pending = $derived(refresh?.pending ?? 0);
  const target = $derived<ReprocessTarget>({
    kind: 'request',
    request: {
      targets: { crop_ids: [cropId] },
      scopes: ['embed'],
      embed: { only_missing: true },
      dry_run: true,
    },
  });
</script>

{#if pending > 0}
  <div
    class="mt-2 flex flex-wrap items-center gap-2 rounded border border-amber-500/40 bg-amber-500/10 px-2 py-1 text-xs text-amber-200"
    data-testid="vector-refresh-notice"
  >
    <span>
      {pending}
      {pending === 1 ? 'box has' : 'boxes have'} no vector yet
    </span>
    <ReprocessControl {target} buttonLabel="Embed now" buttonClass="btn btn-sm" />
  </div>
{/if}
