<!--
  One group: members with their picture, class, holdout state and cosine to
  the representative, and the pair decisions. A decision applies at the next
  run; every served pin can be undone.
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import { resolve } from '$app/paths';
  import { projectHref } from '$lib/projectPaths';
  import Modal from '$components/Modal.svelte';
  import NearDupThumb from '$lib/components/nearDup/NearDupThumb.svelte';
  import { createNearDupGroupController } from '$lib/nearDup/nearDupGroupController.svelte';
  import type { NearDupGroupSummary, NearDupLevel } from '$lib/types_nearDup';

  let {
    level,
    group,
    onclose,
    ondecided,
  }: {
    level: NearDupLevel;
    /** The facts the title needs; a summary from the group list or an image's `near_dup`. */
    group: Pick<NearDupGroupSummary, 'group_id' | 'purpose'> & { size: number | null };
    onclose: () => void;
    /** A decision was made or undone: the status gains pending rows. */
    ondecided?: () => void;
  } = $props();

  // The modal is mounted per group (`{#key}` in the page), so level and group are fixed here.
  // svelte-ignore state_referenced_locally
  const ctl = createNearDupGroupController(level, group.group_id);

  onMount(() => {
    void ctl.load();
    return () => ctl.dispose();
  });

  const rep = $derived(ctl.detail?.members.find((m) => m.is_representative) ?? null);

  async function decide(memberId: string, kind: 'same' | 'different'): Promise<void> {
    if (!rep) return;
    await ctl.decide(rep.member_id, memberId, kind);
    ondecided?.();
  }

  async function undo(id: string): Promise<void> {
    await ctl.undo(id);
    ondecided?.();
  }
</script>

<Modal
  label="Near-duplicate group"
  {onclose}
  panelClass="w-full max-w-3xl max-h-[90vh] overflow-y-auto rounded-lg border border-line bg-surface-0 p-5 shadow-2xl"
  data-testid="near-dup-group-modal"
>
  <header class="mb-3 flex items-baseline gap-3">
    <h2 class="text-title font-semibold">
      {group.purpose === 'dedup' ? 'Duplicate group' : 'Leak group'}{group.size == null
        ? ''
        : `, ${group.size} members`}
    </h2>
    <span class="grow"></span>
    <a
      class="btn-sm"
      data-testid="near-dup-open-workspace"
      href={resolve(
        projectHref(
          `/workspace?view=gallery&scope=${encodeURIComponent(`near_dup:${group.group_id}`)}`,
        ),
      )}>Open in Workspace</a
    >
    <button type="button" class="btn-sm" onclick={onclose}>Close</button>
  </header>

  {#if ctl.loading}
    <p class="text-body text-fg-3">Loading the group…</p>
  {:else if ctl.error}
    <p class="text-body text-danger-text" role="alert" data-testid="near-dup-group-error">
      {ctl.error}
    </p>
  {:else if ctl.detail}
    <p class="mb-3 text-meta text-fg-2">
      A decision applies from the next run. "Not a duplicate" pins the member apart from
      the representative; "Keep together" pins it to the representative.
    </p>
    {#if ctl.actionError}
      <p
        class="mb-2 text-meta text-danger-text"
        role="alert"
        data-testid="near-dup-action-error"
      >
        {ctl.actionError}
      </p>
    {/if}
    <ul class="flex flex-col gap-3">
      {#each ctl.detail.members as m (m.member_id)}
        {@const pin = m.decision}
        <li
          class="flex items-start gap-3 rounded border border-line p-2"
          data-testid="near-dup-member"
        >
          <NearDupThumb url={m.thumbnail_url} label={`${level} ${m.member_id}`} />
          <div class="flex min-w-0 flex-col gap-1 text-meta text-fg-2">
            <code class="break-all text-fg-1">{m.member_id}</code>
            {#if m.class_name}<span>{m.class_name}</span>{/if}
            {#if m.in_holdout}
              <span class="chip w-fit border-warn-line text-warn-text"
                >In test holdout</span
              >
            {/if}
            {#if m.is_representative}
              <span class="chip w-fit">Representative</span>
            {:else}
              <span>Cosine to representative {m.cosine_to_rep.toFixed(3)}</span>
              <div class="flex flex-wrap gap-2">
                {#if pin?.kind === 'different'}
                  <button
                    type="button"
                    class="btn-sm"
                    disabled={ctl.busy}
                    data-testid="near-dup-undo"
                    onclick={() => void undo(pin.id)}>Undo "not a duplicate"</button
                  >
                {:else if pin?.kind === 'same'}
                  <button
                    type="button"
                    class="btn-sm"
                    disabled={ctl.busy}
                    data-testid="near-dup-undo"
                    onclick={() => void undo(pin.id)}>Undo "keep together"</button
                  >
                {:else}
                  <button
                    type="button"
                    class="btn-sm"
                    disabled={ctl.busy}
                    data-testid="near-dup-different"
                    onclick={() => void decide(m.member_id, 'different')}
                    >Not a duplicate</button
                  >
                  <button
                    type="button"
                    class="btn-sm"
                    disabled={ctl.busy}
                    data-testid="near-dup-same"
                    onclick={() => void decide(m.member_id, 'same')}>Keep together</button
                  >
                {/if}
              </div>
            {/if}
          </div>
        </li>
      {/each}
    </ul>
    {#if ctl.pinned > 0}
      <p class="mt-3 text-meta text-warn-text" data-testid="near-dup-decided-note">
        {ctl.pinned} decision{ctl.pinned === 1 ? '' : 's'} saved; run the level again to apply
        {ctl.pinned === 1 ? 'it' : 'them'}.
      </p>
    {/if}
  {/if}
</Modal>
