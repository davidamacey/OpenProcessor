<!--
  The pinned pairs ("Not a duplicate", "Keep together"), newest first, each
  with both members' pictures and an undo. Served by
  `GET /near_duplicates/decisions`; a pin applies from the next run.
-->
<script lang="ts">
  import NearDupThumb from '$lib/components/nearDup/NearDupThumb.svelte';
  import type { NearDupDecisionsController } from '$lib/nearDup/nearDupDecisionsController.svelte';

  let { ctl }: { ctl: NearDupDecisionsController } = $props();

  const KIND: Record<string, string> = {
    same: 'Keep together',
    different: 'Not a duplicate',
  };
</script>

<section class="surface flex flex-col gap-3 p-4" data-testid="near-dup-decisions">
  <h2 class="text-base font-semibold">Decisions</h2>
  {#if ctl.error}
    <p
      class="text-sm text-danger-text"
      role="alert"
      data-testid="near-dup-decisions-error"
    >
      {ctl.error}
    </p>
  {/if}
  {#if !ctl.loaded}
    <p class="text-xs text-fg-3">Loading decisions…</p>
  {:else if ctl.items.length === 0}
    <p class="text-sm text-fg-2" data-testid="near-dup-decisions-empty">
      No pair is pinned. Open a group and mark a member "Not a duplicate" or "Keep
      together".
    </p>
  {:else}
    <ul class="flex flex-col gap-2">
      {#each ctl.items as d (d.id)}
        <li
          class="flex flex-wrap items-center gap-3 rounded border border-line p-2"
          data-testid="near-dup-decision"
        >
          <NearDupThumb
            url={d.a_thumbnail_url}
            label={`${d.level} ${d.a_id}`}
            size="sm"
          />
          <NearDupThumb
            url={d.b_thumbnail_url}
            label={`${d.level} ${d.b_id}`}
            size="sm"
          />
          <span class="flex min-w-0 grow flex-col text-xs text-fg-2">
            <span class="text-sm font-medium text-fg-1"
              >{KIND[d.decision] ?? d.decision}</span
            >
            <span class="break-all font-mono">{d.a_id} / {d.b_id}</span>
            <span class="text-fg-3">{d.level} level, {d.decided_by}, {d.decided_at}</span>
          </span>
          <button
            type="button"
            class="btn-sm"
            disabled={ctl.busy}
            data-testid="near-dup-decision-undo"
            onclick={() => void ctl.undo(d.id)}>Undo</button
          >
        </li>
      {/each}
    </ul>
    {#if ctl.hasMore}
      <button
        type="button"
        class="btn w-fit"
        disabled={ctl.loading}
        data-testid="near-dup-decisions-more"
        onclick={() => void ctl.load(true)}
        >{ctl.loading ? 'Loading…' : 'Load more'}</button
      >
    {/if}
  {/if}
</section>
