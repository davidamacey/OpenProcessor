<!--
  One group: members with their cosine to the representative, and the pair
  decisions. A decision applies at the next run; undo is offered for the ones
  made in this dialog.
-->
<script lang="ts">
  import { onMount } from 'svelte';
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
    group: NearDupGroupSummary;
    onclose: () => void;
    /** A decision was made or undone: the status gains pending rows. */
    ondecided: () => void;
  } = $props();

  // The modal is mounted per group (`{#key}` in the page), so level and group are fixed here.
  // svelte-ignore state_referenced_locally
  const ctl = createNearDupGroupController(level, group.group_id);

  onMount(() => {
    void ctl.load();
    return () => ctl.dispose();
  });

  const rep = $derived(ctl.detail?.members.find((m) => m.is_representative) ?? null);

  function decisionFor(memberId: string, kind: string) {
    return ctl.decisions.find(
      (d) => d.decision === kind && (d.a_id === memberId || d.b_id === memberId),
    );
  }

  async function decide(memberId: string, kind: 'same' | 'different'): Promise<void> {
    if (!rep) return;
    await ctl.decide(rep.member_id, memberId, kind);
    ondecided();
  }

  async function undo(id: string): Promise<void> {
    await ctl.undo(id);
    ondecided();
  }
</script>

<Modal
  label="Near-duplicate group"
  {onclose}
  panelClass="w-full max-w-3xl max-h-[90vh] overflow-y-auto rounded-lg border border-zinc-800 bg-zinc-950 p-5 shadow-2xl"
  data-testid="near-dup-group-modal"
>
  <header class="mb-3 flex items-baseline gap-3">
    <h2 class="text-lg font-semibold">
      {group.purpose === 'dedup' ? 'Duplicate group' : 'Leak group'}, {group.size} members
    </h2>
    <span class="grow"></span>
    <button type="button" class="btn-sm" onclick={onclose}>Close</button>
  </header>

  {#if ctl.loading}
    <p class="text-sm text-zinc-500">Loading the group…</p>
  {:else if ctl.error}
    <p class="text-sm text-red-300" role="alert" data-testid="near-dup-group-error">
      {ctl.error}
    </p>
  {:else if ctl.detail}
    <p class="mb-3 text-xs text-zinc-400">
      A decision applies from the next run. "Not a duplicate" pins the member apart from
      the representative; "Keep together" pins it to the representative.
    </p>
    {#if ctl.actionError}
      <p
        class="mb-2 text-xs text-red-300"
        role="alert"
        data-testid="near-dup-action-error"
      >
        {ctl.actionError}
      </p>
    {/if}
    <ul class="flex flex-col gap-3">
      {#each ctl.detail.members as m (m.member_id)}
        {@const different = decisionFor(m.member_id, 'different')}
        {@const same = decisionFor(m.member_id, 'same')}
        <li
          class="flex items-start gap-3 rounded border border-zinc-800 p-2"
          data-testid="near-dup-member"
        >
          <NearDupThumb {level} memberId={m.member_id} />
          <div class="flex min-w-0 flex-col gap-1 text-xs text-zinc-300">
            <span class="break-all font-mono text-zinc-100">{m.member_id}</span>
            {#if m.is_representative}
              <span class="chip w-fit">Representative</span>
            {:else}
              <span>Cosine to representative {m.cosine_to_rep.toFixed(3)}</span>
              <div class="flex flex-wrap gap-2">
                {#if different}
                  <button
                    type="button"
                    class="btn-sm"
                    disabled={ctl.busy}
                    data-testid="near-dup-undo"
                    onclick={() => void undo(different.id)}>Undo "not a duplicate"</button
                  >
                {:else if same}
                  <button
                    type="button"
                    class="btn-sm"
                    disabled={ctl.busy}
                    data-testid="near-dup-undo"
                    onclick={() => void undo(same.id)}>Undo "keep together"</button
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
    {#if ctl.decisions.length > 0}
      <p class="mt-3 text-xs text-amber-300" data-testid="near-dup-decided-note">
        {ctl.decisions.length} decision{ctl.decisions.length === 1 ? '' : 's'} saved; run the
        level again to apply {ctl.decisions.length === 1 ? 'it' : 'them'}.
      </p>
    {/if}
  {/if}
</Modal>
