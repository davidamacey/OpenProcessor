<script lang="ts">
  import Modal from '$components/Modal.svelte';
  import type { MergeClassesController } from '$lib/classes/mergeClassesController.svelte';
  import type { RegistryClass } from '$lib/types_classes';

  let {
    m,
    allClasses,
    busy,
  }: {
    m: MergeClassesController;
    allClasses: RegistryClass[];
    busy: boolean;
  } = $props();
</script>

<!-- Merge modal -->
{#if m.mergeOpen}
  <Modal label="Merge classes" onclose={m.closeMerge}>
    <h3 class="mb-3 text-base font-semibold">Merge classes</h3>
    <p class="mb-3 text-xs text-zinc-400">
      Source class is marked deprecated; all crops + label rows are relabeled to the
      target.
    </p>

    <label class="mb-3 block text-sm">
      <span class="mb-1 block text-zinc-400">Source (will be merged away)</span>
      <select
        bind:value={m.mergeSourceId}
        class="w-full rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm focus:border-blue-500 focus:outline-none"
      >
        <option value={null}>— pick source —</option>
        {#each allClasses.filter((c) => !c.deprecated) as c (c.id)}
          <option value={c.id}>{c.name} ({c.validated_count ?? 0})</option>
        {/each}
      </select>
    </label>

    <label class="mb-3 block text-sm">
      <span class="mb-1 block text-zinc-400">Target (kept)</span>
      <select
        bind:value={m.mergeTargetId}
        class="w-full rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm focus:border-blue-500 focus:outline-none"
      >
        <option value={null}>— pick target —</option>
        {#each allClasses.filter((c) => !c.deprecated && c.id !== m.mergeSourceId) as c (c.id)}
          <option value={c.id}>{c.name} ({c.validated_count ?? 0})</option>
        {/each}
      </select>
    </label>

    {#if m.mergeSource && m.mergeTarget}
      {#if m.mergePreviewBusy}
        <div class="mb-3 text-xs text-zinc-500">Checking impact…</div>
      {:else if m.mergePreviewError}
        <div
          class="mb-3 rounded border border-red-500/40 bg-red-500/10 px-3 py-2 text-xs text-red-200"
        >
          Preview failed: {m.mergePreviewError}
        </div>
      {:else if m.mergePreview}
        <div
          class="mb-3 rounded border px-3 py-2 text-xs {m.mergePreview.blocked
            ? 'border-red-500/40 bg-red-500/10 text-red-200'
            : 'border-orange-500/40 bg-orange-500/10 text-orange-200'}"
        >
          Will relabel <strong>{m.mergePreview.would_relabel}</strong> crops from
          <strong>{m.mergeSource.name}</strong> to <strong>{m.mergeTarget.name}</strong>
          {#if m.mergePreview.validations_carried_over > 0}
            <!-- 51b05d7: a merge keeps human validations. -->
            &middot; <strong>{m.mergePreview.validations_carried_over}</strong> human
            validation{m.mergePreview.validations_carried_over === 1 ? '' : 's'} will carry
            over
          {/if}
          {#if m.mergePreview.holdout_blocking > 0}
            &middot; <strong>{m.mergePreview.holdout_blocking}</strong> test-holdout crops block
            this merge
          {/if}
          {#if m.mergePreview.blocked}
            <div class="mt-1 font-semibold">
              Blocked — a real merge will 409 until the holdout conflict is resolved.
            </div>
          {/if}
        </div>
      {/if}
    {/if}

    <div class="flex justify-end gap-2">
      <button type="button" class="btn" onclick={m.closeMerge} disabled={busy}>
        Cancel
      </button>
      <button
        type="button"
        class="btn btn-primary"
        onclick={() => void m.submitMerge()}
        disabled={busy ||
          m.mergeSourceId == null ||
          m.mergeTargetId == null ||
          m.mergePreviewBusy ||
          m.mergePreview == null ||
          m.mergePreview?.blocked}
      >
        {busy ? 'Merging…' : 'Merge'}
      </button>
    </div>
  </Modal>
{/if}
