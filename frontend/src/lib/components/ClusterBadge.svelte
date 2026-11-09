<script lang="ts">
  import { resolve } from '$app/paths';
  import { projectHref } from '$lib/projectPaths';
  /**
   * Cluster-origin badge for a search-result crop (global search on
   * `/clusters`, see `CropResultGrid.svelte`). Renders "{#N} ·
   * {dominant_class_name}" with the number the serving partition gave the
   * cluster (blank when it served none; the uid stays in the title), or an amber "Unlabeled"
   * variant when the cluster has no dominant class yet.
   *
   * Clicking navigates to `/clusters/{uid}`. Purely presentational
   * otherwise — the caller supplies `dominant_class_name` (batched from
   * `getClusters()`, never recomputed client-side per this repo's
   * CLAUDE.md).
   */
  import { goto } from '$app/navigation';
  import { displayNoText } from '$lib/clusters/clusterCardText';

  interface Props {
    clusterUid: string;
    dominantClassName?: string | null;
    displayNo?: number | null;
  }

  let { clusterUid, dominantClassName = null, displayNo = null }: Props = $props();

  const numberText = $derived(displayNoText({ display_no: displayNo }));

  function open(e: MouseEvent): void {
    e.stopPropagation();
    void goto(resolve(projectHref(`/clusters/${encodeURIComponent(clusterUid)}`)));
  }
</script>

{#if dominantClassName}
  <button
    type="button"
    onclick={open}
    class="pointer-events-auto block max-w-full truncate rounded border border-zinc-700 bg-zinc-900/90 px-1.5 py-0.5 text-[10px] text-zinc-200 hover:border-blue-500/60 hover:text-blue-200"
    title="Open cluster {clusterUid}"
  >
    {numberText} · {dominantClassName}
  </button>
{:else}
  <button
    type="button"
    onclick={open}
    class="pointer-events-auto block max-w-full truncate rounded border border-amber-500/50 bg-amber-500/10 px-1.5 py-0.5 text-[10px] text-amber-200 hover:bg-amber-500/20"
    title="Open cluster {clusterUid} (unlabeled)"
  >
    Unlabeled {numberText}
  </button>
{/if}
