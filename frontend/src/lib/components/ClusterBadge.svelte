<script lang="ts">
  import { resolve } from '$app/paths';
  import { projectHref } from '$lib/projectPaths';
  /**
   * Cluster-origin badge for a search-result crop (global search on
   * `/clusters`, see `CropResultGrid.svelte`). Renders "#{id} ·
   * {dominant_class_name}" in the same idiom as the cluster-card chip on
   * `/clusters` (dominant-class name + id), or an amber "Unlabeled #{id}"
   * variant when the cluster has no dominant class yet — mirrors the
   * `Unlabeled #{c.id}` branch in `src/routes/clusters/+page.svelte`.
   *
   * Clicking navigates to `/clusters/{id}`. Purely presentational
   * otherwise — the caller supplies `dominant_class_name` (batched from
   * `getClusters()`, never recomputed client-side per this repo's
   * CLAUDE.md).
   */
  import { goto } from '$app/navigation';

  interface Props {
    clusterId: number;
    dominantClassName?: string | null;
  }

  let { clusterId, dominantClassName = null }: Props = $props();

  function open(e: MouseEvent): void {
    e.stopPropagation();
    void goto(resolve(projectHref(`/clusters/${clusterId}`)));
  }
</script>

{#if dominantClassName}
  <button
    type="button"
    onclick={open}
    class="pointer-events-auto rounded border border-zinc-700 bg-zinc-900/90 px-1.5 py-0.5 text-[10px] text-zinc-200 hover:border-blue-500/60 hover:text-blue-200"
    title="Open cluster #{clusterId}"
  >
    #{clusterId} · {dominantClassName}
  </button>
{:else}
  <button
    type="button"
    onclick={open}
    class="pointer-events-auto rounded border border-amber-500/50 bg-amber-500/10 px-1.5 py-0.5 text-[10px] text-amber-200 hover:bg-amber-500/20"
    title="Open cluster #{clusterId} (unlabeled)"
  >
    Unlabeled #{clusterId}
  </button>
{/if}
