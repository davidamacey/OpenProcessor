<script lang="ts">
  /**
   * What the selection is and what an action will change. The sentence is the
   * server's (`POST /selection/summary`); every action opens the server's
   * dry run first and writes only on confirm. Ignore reasons are the served
   * list. A view adds its own buttons through `extra`.
   */
  import type { Snippet } from 'svelte';
  import ChevronDownIcon from '$components/ChevronDownIcon.svelte';
  import ClassPicker from '$components/ClassPicker.svelte';
  import SelectionActionDialog from '$components/itemFilter/SelectionActionDialog.svelte';
  import { createSelectionActionController } from '$lib/itemFilter/selectionActionController.svelte';
  import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';
  import { excludeReasonsStore } from '$stores/excludeReasons.svelte';
  import { keymapStore } from '$stores/keymap.svelte';
  import { undoStore } from '$stores/undo.svelte';

  interface Props {
    ws: WorkspaceController;
    /** Reload the view after a write landed. */
    onapplied?: () => void;
    extra?: Snippet;
  }

  let { ws, onapplied, extra }: Props = $props();

  const actions = createSelectionActionController(() => ws.itemSelection() ?? {});
  let reasonsOpen = $state(false);

  const hasSelection = $derived(ws.selection.kind !== 'none');
  const count = $derived.by((): number | null => {
    const sel = ws.selection;
    if (sel.kind === 'none') return null;
    if (sel.kind === 'items') return sel.cropIds.length;
    if (sel.kind === 'images') return sel.imageIds.length;
    if (sel.kind === 'clusters') return sel.clusterUids.length;
    return null;
  });
  const text = $derived.by(() => {
    if (!hasSelection) return 'Nothing selected';
    if (ws.summary.error) return ws.summary.error;
    return ws.summary.summary?.summary_text ?? 'Counting…';
  });

  $effect(() => {
    void excludeReasonsStore.init();
  });

  function assign(cls: { id: number }): void {
    ws.pickerOpen = false;
    actions.classId = cls.id;
    void actions.open('label');
  }

  function ignore(reason: string): void {
    reasonsOpen = false;
    actions.reason = reason;
    void actions.open('exclude');
  }
</script>

<div
  class="flex flex-wrap items-center gap-x-2 gap-y-1 border-b border-zinc-800 bg-zinc-900/60 px-4 py-2 text-xs"
  role="toolbar"
  aria-label="Bulk actions"
  data-testid="bulk-bar"
>
  <span
    class="min-w-0 text-zinc-200"
    class:text-red-300={!!ws.summary.error}
    aria-live="polite"
    data-testid="bulk-summary">{text}</span
  >
  {#if count !== null}
    <span class="font-mono text-zinc-500" data-testid="bulk-count">{count} selected</span>
  {/if}
  {#if ws.pendingSelectAll}
    <button
      type="button"
      class="btn"
      data-testid="select-all-matching"
      onclick={() => ws.confirmSelectAllMatching()}
    >
      Select all {ws.pendingSelectAll.total.toLocaleString()} matching
    </button>
  {/if}
  <span class="grow"></span>
  <button
    type="button"
    class="btn"
    disabled={!hasSelection}
    onclick={() => ws.clearSelection()}
  >
    Deselect
  </button>
  <button
    type="button"
    class="btn btn-primary"
    disabled={!hasSelection}
    data-testid="assign-selected"
    onclick={() => (ws.pickerOpen = true)}
  >
    Assign class…
  </button>
  <button
    type="button"
    class="btn"
    disabled={!hasSelection}
    title={keymapStore.glyph('cluster.move')}
    onclick={() => void actions.open('move')}
  >
    Move…
  </button>
  <div class="relative inline-flex">
    <button
      type="button"
      class="btn btn-join-start"
      disabled={!hasSelection}
      title={keymapStore.glyph('cluster.ignore')}
      data-testid="ignore-selected"
      onclick={() => ignore('ignore')}
    >
      Ignore
    </button>
    {#if excludeReasonsStore.reasons.length > 0}
      <button
        type="button"
        class="btn btn-icon btn-join-end"
        disabled={!hasSelection}
        aria-label="Choose ignore reason"
        aria-expanded={reasonsOpen}
        onclick={() => (reasonsOpen = !reasonsOpen)}
      >
        <ChevronDownIcon size={18} />
      </button>
    {/if}
    {#if reasonsOpen}
      <div
        class="absolute left-0 top-full z-20 mt-1 w-44 rounded sm:left-auto sm:right-0 border border-zinc-700 bg-zinc-900 py-1 shadow-lg"
        data-testid="ignore-reasons"
      >
        {#each excludeReasonsStore.reasons as r (r.id)}
          <button
            type="button"
            class="block w-full px-3 py-1 text-left text-xs text-zinc-200 hover:bg-zinc-800"
            onclick={() => ignore(r.id)}
          >
            {r.label}
          </button>
        {/each}
      </div>
    {/if}
  </div>
  <button
    type="button"
    class="btn"
    disabled={!hasSelection}
    onclick={() => void actions.open('unexclude')}
  >
    Restore…
  </button>
  <button type="button" class="btn" onclick={() => void undoStore.undoLast()}>
    Undo
  </button>
  {@render extra?.()}
</div>

<ClassPicker
  open={ws.pickerOpen}
  mode="assign"
  onpick={assign}
  onclose={() => (ws.pickerOpen = false)}
/>
<SelectionActionDialog controller={actions} {onapplied} />
