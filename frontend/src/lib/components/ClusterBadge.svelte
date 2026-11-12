<script lang="ts">
  import { resolve } from '$app/paths';
  import { clusterScopeSearch } from '$lib/workspace/workspaceUrl';
  import { projectHref } from '$lib/projectPaths';
  /**
   * Cluster-origin badge for a search-result crop (global search on
   * `/clusters`, see `CropResultGrid.svelte`). Renders "{#N} ·
   * {display_name}" with the number the serving partition gave the
   * cluster (blank when it served none; the uid stays in the title), or an amber "Unlabeled"
   * variant when the server served no name.
   *
   * Clicking opens the cluster in the Workspace. Purely presentational
   * otherwise — the caller supplies the served `display_name` (batched from
   * `getClusters()`, never recomputed client-side per this repo's
   * CLAUDE.md).
   */
  import type { ClusterNameKind } from '$lib/types';
  import { goto } from '$app/navigation';
  import { displayNameNote, displayNoText } from '$lib/clusters/clusterCardText';

  interface Props {
    clusterUid: string;
    displayName?: string | null;
    displayNameKind?: ClusterNameKind;
    displayNo?: number | null;
  }

  let {
    clusterUid,
    displayName = null,
    displayNameKind = 'none',
    displayNo = null,
  }: Props = $props();

  const numberText = $derived(displayNoText({ display_no: displayNo }));

  function open(e: MouseEvent): void {
    e.stopPropagation();
    void goto(resolve(projectHref(`/workspace${clusterScopeSearch(clusterUid)}`)));
  }
</script>

{#if displayName && displayNameKind !== 'none'}
  <button
    type="button"
    onclick={open}
    class="pointer-events-auto block max-w-full truncate rounded border border-line-strong bg-surface-1/90 px-1.5 py-0.5 text-caption text-fg-1 hover:border-accent/60 hover:text-accent-text"
    title="Open cluster {clusterUid}"
  >
    {numberText} · {displayName}{displayNameNote(displayNameKind) ? ' (detector)' : ''}
  </button>
{:else}
  <button
    type="button"
    onclick={open}
    class="pointer-events-auto block max-w-full truncate rounded border border-warn/50 bg-warn/10 px-1.5 py-0.5 text-caption text-warn-text hover:bg-warn/20"
    title="Open cluster {clusterUid} (unlabeled)"
  >
    Unlabeled {numberText}
  </button>
{/if}
