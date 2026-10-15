<script lang="ts">
  import { resolve } from '$app/paths';
  import ChevronDownIcon from '$components/ChevronDownIcon.svelte';
  import ReprocessControl from '$components/datasets/ReprocessControl.svelte';
  import ShortcutsButton from '$components/ShortcutsButton.svelte';
  import { isAssignableClass } from '$lib/classVisibility';
  import { cohesionText, COHESION_TOOLTIP } from '$lib/clusters/clusterCardText';
  import { projectHref } from '$lib/projectPaths';
  import { classesStore } from '$stores/classes.svelte';
  import type { ClusterDetailController } from '$lib/clusters/clusterDetailController.svelte';
  import type { ClusterDetailActions } from '$lib/clusters/clusterDetailActions.svelte';
  import type { ClusterAssist } from '$lib/assistRun/clusterAssist.svelte';

  // Title, cluster stats and the bulk-action row of /clusters/[id].
  let {
    view,
    actions,
    clusterAssist,
  }: {
    view: ClusterDetailController;
    actions: ClusterDetailActions;
    clusterAssist: ClusterAssist;
  } = $props();
</script>

<!-- Toolbar — title + summary + bulk-action buttons + class assignment.
     Drag-and-drop onto the left ClassSidebar is the alternative path. -->
<div class="flex flex-wrap items-center gap-3 border-b border-zinc-800 px-4 py-2.5">
  <a
    href={resolve(projectHref('/clusters'))}
    class="btn shrink-0"
    title="Back to all clusters"
  >
    ← All clusters
  </a>
  <h1 class="flex items-baseline gap-2 text-lg font-semibold">
    {#if view.clusterName}
      <span class="capitalize text-zinc-100">{view.clusterName}</span>
      <span class="text-sm font-normal text-zinc-500">#{view.clusterIdParam}</span>
    {:else if view.cluster?.cluster_kind === 'candidate'}
      <span class="text-zinc-100">Cluster #{view.clusterIdParam}</span>
      <span class="text-sm font-normal text-amber-400/80">unlabeled candidate</span>
    {:else}
      <span class="text-zinc-100">Cluster #{view.clusterIdParam}</span>
    {/if}
  </h1>
  <ShortcutsButton />
  {#if view.cardError}
    <span class="text-xs text-amber-300" data-testid="cluster-card-error"
      >Cluster details unavailable: {view.cardError}</span
    >
  {/if}
  {#if view.cluster}
    <span class="text-xs text-zinc-400">
      size {view.cluster.size.toLocaleString()} · dominant
      <strong class="text-zinc-200">{view.cluster.dominant_class_name ?? '—'}</strong>
    </span>
    <!-- DQ-M2 fix (dq-queues cutover, 2026-09-24): purity is now the
         nearest-centroid geometry share — purity_tier stays the badge
         elsewhere (the sidebar/card border), this is the number + its
         basis/n, with label_purity/labelled_share in the tooltip so an
         operator can tell "geometrically coherent" from "the labels we
         have agree with each other". -->
    {#if view.cluster.purity != null}
      <span
        class="font-mono text-xs text-zinc-500"
        data-testid="cluster-cohesion"
        title="{COHESION_TOOLTIP} · label agreement {view.cluster.label_agreement != null
          ? `${(view.cluster.label_agreement * 100).toFixed(0)}%`
          : '—'} · labelled share {view.cluster.labelled_share != null
          ? `${(view.cluster.labelled_share * 100).toFixed(0)}%`
          : '—'}"
      >
        · {cohesionText(view.cluster)}
      </span>
    {/if}
  {/if}
  {#if view.cluster}
    <!-- K1 (visual audit 2026-09-24): this chip used to read "35
         validated · 135 labeled · 134 in cluster" — the first two are
         the class registry's CLASS-WIDE counts (every crop of the class,
         test holdout included), the last this cluster's own size, so
         "labeled" could exceed "in cluster". Each number now names its
         scope; the class-wide pair stays on classesStore because it is
         what re-fetches after a label write. -->
    <span
      class="rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1 font-mono text-[11px] text-zinc-300"
      title="validated · labeled · cluster total — class-wide counts are every crop of this class (test holdout included), from the class registry; 'in this cluster' is this cluster's own member count (test holdout included). The grid lists only reviewable members."
      data-testid="cluster-header-counts"
    >
      {#if view.clsForCluster}
        <span class="text-zinc-500">class-wide:</span>
        <span class="text-emerald-300"
          >{(view.clsForCluster.validated_count ?? 0).toLocaleString()}</span
        >
        <span class="text-zinc-500">validated</span>
        <span class="mx-1 text-zinc-600">·</span>
        <span class="text-blue-300"
          >{(view.clsForCluster.count ?? 0).toLocaleString()}</span
        >
        <span class="text-zinc-500">labeled</span>
        <span class="mx-1 text-zinc-600">|</span>
      {/if}
      <span class="text-zinc-200">{view.cluster.size.toLocaleString()}</span>
      <span class="text-zinc-500">in this cluster</span>
    </span>
  {/if}

  <span class="grow"></span>

  <div class="flex flex-wrap items-center gap-x-0.5 gap-y-1">
    <button
      class="btn"
      type="button"
      onclick={view.selectAllPage}
      title={actions.kg('cluster.select_all')}
    >
      Select page
    </button>
    <button
      class="btn"
      type="button"
      onclick={view.deselectAll}
      title={actions.kg('cluster.cancel')}
    >
      Deselect
    </button>
    <span class="font-mono text-xs text-zinc-500">{view.sel.size} selected</span>

    <select
      bind:value={actions.confirmClassId}
      class="select"
      aria-label="Class to assign to the selected crops"
    >
      <option value={null}>— assign to class —</option>
      {#each classesStore.classes.filter(isAssignableClass) as cls (cls.id)}
        <option value={cls.id}>{cls.name}</option>
      {/each}
    </select>
    <button
      class="btn btn-primary"
      type="button"
      onclick={actions.confirmSelected}
      disabled={view.sel.size === 0 || actions.confirmClassId == null}
      title="{actions.kg(
        'cluster.confirm',
      )} — label the selected crops with the chosen class (relabel)"
      data-testid="assign-selected"
    >
      Assign class to selected
    </button>

    <!-- Selection-aware action chips. Always rendered so the user knows
         the actions exist; disabled until a selection is non-empty.
         Each chip shows its hotkey so the keyboard path is discoverable. -->
    <button
      class="btn"
      type="button"
      onclick={actions.openMovePicker}
      disabled={view.sel.size === 0}
      title="{actions.kg('cluster.move')} — move selected to a different cluster"
    >
      Move <kbd class="ml-1 font-mono text-[10px] text-zinc-400"
        >{actions.kc('cluster.move')}</kbd
      >
    </button>
    <button
      class="btn"
      type="button"
      onclick={() => void actions.flagSelectedForNewClass()}
      disabled={view.sel.size === 0}
      title="{actions.kg(
        'cluster.flag_new_class',
      )} — flag selected as needing a new class"
    >
      Flag <kbd class="ml-1 font-mono text-[10px] text-zinc-400"
        >{actions.kc('cluster.flag_new_class')}</kbd
      >
    </button>
    <!-- Reprocess on the selection: absent unless the
         backend serves it; a served dry run precedes the apply. -->
    <ReprocessControl
      target={{ kind: 'crops', cropIds: [...view.sel.ids] }}
      disabled={view.sel.size === 0}
      buttonClass="btn"
      onapplied={() => void view.loadFirst()}
    />
    <button
      class="btn"
      type="button"
      onclick={actions.controller.acceptAllVlmOnPage}
      title="{actions.kg(
        'cluster.accept_all_vlm',
      )} — accept all VLM suggestions on this page"
    >
      Accept VLM <kbd class="ml-1 font-mono text-[10px] text-zinc-400"
        >{actions.kc('cluster.accept_all_vlm')}</kbd
      >
    </button>

    <span class="mx-1 h-5 w-px bg-zinc-800"></span>

    <!-- Assist with the VLM on this (sub-)cluster, or on the crops selected in it. -->
    <button
      class="btn"
      type="button"
      onclick={clusterAssist.openAssist}
      title="Ask the VLM about the selected crops, this cluster or sub-cluster, or a representative sample of it"
      data-testid="assist-open"
    >
      Assist…
    </button>
    <button
      class="btn"
      type="button"
      onclick={actions.refine}
      disabled={actions.refining}
      title="Sub-cluster this cluster with AHC"
    >
      {#if actions.refining}
        <span
          class="mr-1 inline-block h-3 w-3 animate-spin rounded-full border-2 border-zinc-500 border-t-zinc-100 align-[-1px]"
        ></span>
        Refining…
      {:else}
        Refine (AHC)
      {/if}
    </button>

    <span class="mx-1 h-5 w-px bg-zinc-800"></span>

    <!-- Ignore: one-click default (generic), dropdown for a reason.
         Excludes from training + clustering; reversible (U). -->
    <div class="relative inline-flex">
      <button
        class="btn btn-join-start"
        type="button"
        title="Ignore selected — exclude from training + clustering ({actions.kg(
          'cluster.ignore',
        )})"
        onclick={() => void actions.ignoreSelected('ignore')}
      >
        Ignore <kbd class="ml-1 font-mono text-[10px] text-zinc-400"
          >{actions.kc('cluster.ignore')}</kbd
        >
      </button>
      <button
        class="btn btn-icon btn-join-end"
        type="button"
        aria-label="Choose ignore reason"
        onclick={() => (actions.ignoreMenuOpen = !actions.ignoreMenuOpen)}
      >
        <ChevronDownIcon size={18} />
      </button>
      {#if actions.ignoreMenuOpen}
        <div
          class="absolute left-0 top-full z-20 mt-1 w-44 sm:left-auto sm:right-0 rounded border border-zinc-700 bg-zinc-900 py-1 shadow-lg"
        >
          {#each actions.EXCLUDE_REASONS as r (r.value)}
            <button
              type="button"
              class="block w-full px-3 py-1 text-left text-xs text-zinc-200 hover:bg-zinc-800"
              onclick={() => void actions.ignoreSelected(r.value)}
            >
              {r.label}
            </button>
          {/each}
        </div>
      {/if}
    </div>
  </div>
</div>
