<!--
  The object-level leak groups of the current run, largest first: crops that must stay on one
  side of a train/test split. Each row opens its members in Detections (the `near_dup` scope).
-->
<script lang="ts">
  import { resolve } from '$app/paths';
  import DataTable, { type Column } from '$components/ui/DataTable.svelte';
  import { projectHref } from '$lib/projectPaths';
  import { writeWorkspaceUrl } from '$lib/workspace/workspaceUrl';
  import type { NearDupPageController } from '$lib/nearDup/nearDupPageController.svelte';
  import type { NearDupGroupSummary } from '$lib/types_nearDup';

  let { ctl }: { ctl: NearDupPageController } = $props();

  const columns: Column<NearDupGroupSummary>[] = [
    { id: 'group', label: 'Group', value: (g) => g.group_id },
    { id: 'size', label: 'Members', value: (g) => g.size, align: 'right' },
    {
      id: 'cos',
      label: 'Lowest similarity',
      value: (g) => g.min_edge_cosine,
      align: 'right',
    },
    { id: 'holdout', label: 'In the test set', value: (g) => Number(g.has_holdout) },
    { id: 'open', label: 'View' },
  ];

  function href(groupId: string): string {
    const params = new URLSearchParams();
    writeWorkspaceUrl(params, {
      view: 'detections',
      scope: { kind: 'near_dup', id: groupId },
    });
    return projectHref(`/workspace?${params.toString()}`);
  }
</script>

<div data-testid="near-dup-groups">
  {#if ctl.groupsError}
    <p
      class="text-body text-danger-text"
      role="alert"
      data-testid="near-dup-groups-error"
    >
      {ctl.groupsError}
    </p>
    <button type="button" class="btn mt-2 w-fit" onclick={() => void ctl.loadGroups()}
      >Retry</button
    >
  {:else if !ctl.groupsLoaded}
    <p class="text-meta text-fg-3">Loading groups…</p>
  {:else if ctl.groups.length === 0}
    <p class="text-body text-fg-2" data-testid="near-dup-groups-empty">
      {ctl.emptyReason ?? 'No groups.'}
    </p>
  {:else}
    <DataTable
      {columns}
      rows={ctl.groups}
      rowKey={(g) => g.group_id}
      label="Object leak groups"
      maxHeightClass="max-h-96"
    >
      {#snippet cell(g, col)}
        {#if col.id === 'group'}
          <code>{g.group_id}</code>
        {:else if col.id === 'size'}
          {g.size}
        {:else if col.id === 'cos'}
          {g.min_edge_cosine.toFixed(3)}
        {:else if col.id === 'holdout'}
          {g.has_holdout ? 'Yes' : 'No'}
        {:else}
          <a
            class="text-accent-text hover:underline"
            data-testid="near-dup-group-workspace"
            href={resolve(href(g.group_id) as '/')}>Open in Detections</a
          >
        {/if}
      {/snippet}
    </DataTable>
    {#if ctl.hasMore}
      <button
        type="button"
        class="btn mt-2 w-fit"
        disabled={ctl.groupsLoading}
        data-testid="near-dup-groups-more"
        onclick={() => void ctl.loadGroups(true)}
        >{ctl.groupsLoading ? 'Loading…' : 'Load more'}</button
      >
    {/if}
  {/if}
</div>
