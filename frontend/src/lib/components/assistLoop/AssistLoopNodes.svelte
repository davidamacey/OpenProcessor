<script lang="ts">
  /**
   * The loop's node table (a cluster or a part of one): the decision, the top
   * class, the exact agreement bound and what was written. A row opens to the
   * vote distribution and the member crops (every member, the ones the lock
   * rule kept out included). Rollback of one node starts from its row.
   */
  import type { AssistLoopController } from '$lib/assistLoop/assistLoopController.svelte';
  import { getThumbUrl } from '$lib/api_cropRead';
  import { formatCount } from '$lib/format/count';
  import type { AssistLoopDecision } from '$lib/types_assistLoop';

  interface Props {
    ctl: AssistLoopController;
  }

  let { ctl }: Props = $props();

  const DECISIONS: { id: AssistLoopDecision; label: string }[] = [
    { id: 'pass', label: 'Passed' },
    { id: 'split', label: 'Split' },
    { id: 'review_leaf', label: 'Left for review' },
    { id: 'review_small', label: 'Too small, for review' },
    { id: 'stale', label: 'Stale' },
    { id: 'unvisited', label: 'Not visited' },
  ];
  const LABEL = Object.fromEntries(DECISIONS.map((d) => [d.id, d.label]));
  const THUMBS_SHOWN = 48;
</script>

<div class="flex flex-col gap-2 text-xs" data-testid="loop-nodes">
  <div class="flex flex-wrap items-center gap-3">
    <h3 class="text-sm font-medium text-zinc-200">Clusters and parts</h3>
    <label class="flex items-center gap-1.5 text-zinc-300">
      Show
      <select
        class="select-sm"
        value={ctl.decisionFilter}
        onchange={(e) =>
          void ctl.setDecisionFilter((e.currentTarget as HTMLSelectElement).value)}
        data-testid="loop-node-filter"
      >
        <option value="">All decisions</option>
        {#each DECISIONS as d (d.id)}
          <option value={d.id}>{d.label}</option>
        {/each}
      </select>
    </label>
  </div>

  {#if ctl.nodesError}
    <p class="text-red-300" role="alert" data-testid="loop-nodes-error">
      {ctl.nodesError}
    </p>
  {/if}

  <div class="overflow-x-auto">
    <table class="w-full min-w-[44rem] text-left">
      <caption class="sr-only">Clusters the loop worked on and what it decided</caption>
      <thead class="text-zinc-500">
        <tr>
          <th scope="col" class="pr-3 font-normal">Node</th>
          <th scope="col" class="pr-3 font-normal">Decision</th>
          <th scope="col" class="pr-3 font-normal">Top class</th>
          <th scope="col" class="pr-3 text-right font-normal">Free members</th>
          <th scope="col" class="pr-3 text-right font-normal">Agreed / asked</th>
          <th scope="col" class="pr-3 text-right font-normal">Bound</th>
          <th scope="col" class="pr-3 text-right font-normal">Propagated</th>
          <th scope="col" class="font-normal"><span class="sr-only">Actions</span></th>
        </tr>
      </thead>
      <tbody class="font-mono text-zinc-200">
        {#each ctl.nodes as n (n.path)}
          {@const open = ctl.openNode === n.path}
          <tr class="border-t border-zinc-800/70" data-testid="loop-node-row">
            <th scope="row" class="py-1 pr-3 text-left font-normal">{n.path}</th>
            <td class="pr-3 font-sans" title={n.reason ?? ''}>
              {n.decision ? LABEL[n.decision] : 'Open'}
            </td>
            <td class="pr-3 font-sans">{n.class_name ?? '—'}</td>
            <td class="pr-3 text-right">{formatCount(n.n)}</td>
            <td class="pr-3 text-right"
              >{n.agree === null ? '—' : formatCount(n.agree)} / {formatCount(
                n.answered,
              )}</td
            >
            <td class="pr-3 text-right">{n.bound === null ? '—' : n.bound.toFixed(3)}</td>
            <td class="pr-3 text-right">{formatCount(n.items_propagated)}</td>
            <td class="whitespace-nowrap text-right font-sans">
              <button
                type="button"
                class="btn btn-sm"
                aria-expanded={open}
                aria-controls="loop-node-detail-{n.path}"
                aria-label="{open ? 'Hide' : 'Show'} members of {n.path}"
                onclick={() => void ctl.toggleNode(n.path)}
                data-testid="loop-node-toggle"
              >
                {open ? 'Hide' : 'Members'}
              </button>
              {#if n.items_propagated > 0 && !ctl.live}
                <button
                  type="button"
                  class="btn btn-sm"
                  disabled={ctl.busy}
                  aria-label="Roll back {n.path}"
                  onclick={() => void ctl.previewRollback([n.path])}
                  data-testid="loop-node-rollback"
                >
                  Roll back
                </button>
              {/if}
            </td>
          </tr>
          {#if open}
            <tr id="loop-node-detail-{n.path}" data-testid="loop-node-detail">
              <td colspan="8" class="bg-zinc-900/40 p-3 font-sans">
                {#if n.distribution.length > 0}
                  <p class="mb-1 text-zinc-300">
                    Votes:
                    {n.distribution.map((v) => `${v.class_name} ${v.votes}`).join(', ')}
                  </p>
                {/if}
                {#if n.new_class_proposals.length > 0}
                  <p class="mb-1 text-zinc-300">
                    Names the VLM proposed that are not in the registry:
                    {n.new_class_proposals.map((v) => `${v.name} ${v.votes}`).join(', ')}
                  </p>
                {/if}
                {#if n.reason}
                  <p class="mb-1 text-zinc-500">{n.reason}</p>
                {/if}
                {#if ctl.membersError}
                  <p class="text-red-300" role="alert">{ctl.membersError}</p>
                {/if}
                <ul class="flex flex-wrap gap-1" aria-label="Members of {n.path}">
                  {#each ctl.members.slice(0, THUMBS_SHOWN) as id (id)}
                    <li>
                      <img
                        src={getThumbUrl(id, 96)}
                        alt="Crop {id}"
                        loading="lazy"
                        width="64"
                        height="64"
                        class="h-16 w-16 rounded bg-zinc-800 object-cover"
                      />
                    </li>
                  {/each}
                </ul>
                <p class="mt-1 text-zinc-500" data-testid="loop-members-count">
                  {formatCount(ctl.members.length)} member ids loaded{ctl.membersNext
                    ? ' (more available)'
                    : ''}{ctl.members.length > THUMBS_SHOWN
                    ? `, first ${THUMBS_SHOWN} shown`
                    : ''}.
                </p>
                {#if ctl.membersNext}
                  <button
                    type="button"
                    class="btn btn-sm mt-1"
                    onclick={() => void ctl.moreMembers()}
                    data-testid="loop-members-more"
                  >
                    Load more members
                  </button>
                {/if}
              </td>
            </tr>
          {/if}
        {:else}
          <tr>
            <td colspan="8" class="py-2 font-sans text-zinc-500">
              {ctl.nodesLoading ? 'Loading…' : 'No clusters to show.'}
            </td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
  {#if ctl.nodesNext}
    <div>
      <button
        type="button"
        class="btn btn-sm"
        disabled={ctl.nodesLoading}
        onclick={() => void ctl.loadNodes(false)}
        data-testid="loop-nodes-more"
      >
        Load more
      </button>
    </div>
  {/if}
</div>
