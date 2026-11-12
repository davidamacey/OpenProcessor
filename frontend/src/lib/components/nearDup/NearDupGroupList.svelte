<!--
  The current run's groups of one level and purpose, largest first, with a
  thumbnail of the representative. Opening a group is the parent's call.
-->
<script lang="ts">
  import NearDupThumb from '$lib/components/nearDup/NearDupThumb.svelte';
  import type { NearDupPageController } from '$lib/nearDup/nearDupPageController.svelte';
  import type {
    NearDupGroupSummary,
    NearDupLevel,
    NearDupPurpose,
  } from '$lib/types_nearDup';

  let {
    ctl,
    onopen,
  }: { ctl: NearDupPageController; onopen: (g: NearDupGroupSummary) => void } = $props();

  const LEVELS: { id: NearDupLevel; label: string }[] = [
    { id: 'image', label: 'Images' },
    { id: 'item', label: 'Items' },
  ];
  const PURPOSES: { id: NearDupPurpose; label: string }[] = [
    { id: 'dedup', label: 'Duplicates' },
    { id: 'leak', label: 'Leak groups' },
  ];
  const PURPOSE_HINT: Record<NearDupPurpose, string> = {
    dedup: 'Members at or above the duplicate cut: keep one, drop the rest.',
    leak: 'Members at or above the leak cut: they must stay on one train/test side.',
  };

  function cos(v: number): string {
    return v.toFixed(3);
  }
</script>

<section class="surface flex flex-col gap-3 p-4" data-testid="near-dup-groups">
  <header class="flex flex-wrap items-center gap-3">
    <h2 class="text-base font-semibold">Groups</h2>
    <div class="flex gap-1" role="group" aria-label="Level">
      {#each LEVELS as l (l.id)}
        <button
          type="button"
          class="btn-sm"
          aria-pressed={ctl.level === l.id}
          data-testid="near-dup-level-tab-{l.id}"
          onclick={() => void ctl.showGroups(l.id, ctl.purpose)}>{l.label}</button
        >
      {/each}
    </div>
    <div class="flex gap-1" role="group" aria-label="Purpose">
      {#each PURPOSES as p (p.id)}
        <button
          type="button"
          class="btn-sm"
          aria-pressed={ctl.purpose === p.id}
          data-testid="near-dup-purpose-tab-{p.id}"
          onclick={() => void ctl.showGroups(ctl.level, p.id)}>{p.label}</button
        >
      {/each}
    </div>
  </header>
  <p class="text-xs text-fg-2">{PURPOSE_HINT[ctl.purpose]}</p>

  {#if ctl.groupsError}
    <div class="flex flex-col gap-2 text-sm" data-testid="near-dup-groups-error">
      <p class="text-danger" role="alert">{ctl.groupsError}</p>
      <button type="button" class="btn w-fit" onclick={() => void ctl.loadGroups()}
        >Retry</button
      >
    </div>
  {:else if !ctl.groupsLoaded}
    <p class="text-xs text-fg-3">Loading groups…</p>
  {:else if ctl.groups.length === 0}
    <p class="text-sm text-fg-2" data-testid="near-dup-groups-empty">
      {ctl.emptyReason}
    </p>
  {:else}
    <ul class="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
      {#each ctl.groups as g (g.group_id)}
        <li>
          <button
            type="button"
            class="flex w-full items-start gap-3 rounded border border-line p-3 text-left hover:border-line-strong"
            data-testid="near-dup-group"
            onclick={() => onopen(g)}
          >
            <NearDupThumb
              url={g.representative_thumbnail_url}
              label={`${ctl.level} ${g.representative_id}`}
              size="sm"
            />
            <span class="flex min-w-0 flex-col gap-1 text-xs text-fg-2">
              <span class="text-sm font-medium text-fg-1">{g.size} members</span>
              <span>Lowest edge cosine {cos(g.min_edge_cosine)}</span>
              <span>Lowest to representative {cos(g.min_cosine_to_rep)}</span>
              {#if g.has_holdout}
                <span class="chip w-fit border-warn-line text-warn">In test holdout</span>
              {/if}
            </span>
          </button>
        </li>
      {/each}
    </ul>
    {#if ctl.hasMore}
      <button
        type="button"
        class="btn w-fit"
        disabled={ctl.groupsLoading}
        data-testid="near-dup-groups-more"
        onclick={() => void ctl.loadGroups(true)}
        >{ctl.groupsLoading ? 'Loading…' : 'Load more'}</button
      >
    {/if}
  {/if}
</section>
