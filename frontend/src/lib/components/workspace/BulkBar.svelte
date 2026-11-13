<script lang="ts">
  import Icon from '$components/ui/Icon.svelte';
  /**
   * What the selection is and what an action will change. The sentence is the
   * server's (`POST /selection/summary`); every action opens the server's
   * dry run first and writes only on confirm. Ignore reasons are the served
   * list. A view adds its own buttons through `extra`.
   */
  import type { Snippet } from 'svelte';
  import ClassPicker from '$components/ClassPicker.svelte';
  import DiverseSample from '$components/workspace/DiverseSample.svelte';
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
  // The open near-duplicate group as a whole: the filter carries `near_dup_group`, so the label
  // reaches every member the server counts, not only the ones loaded.
  const group = createSelectionActionController(() => ({ filter: ws.itemBody() }));
  let groupPickerOpen = $state(false);
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

  // A label-by-key flow that ends without a write (picker or dialog dismissed) must not make a
  // later, ordinary label move to the next cluster.
  $effect(() => {
    if (ws.labelThenNext && !ws.pickerOpen && actions.action == null)
      ws.labelThenNext = false;
  });

  function assign(cls: { id: number }): void {
    ws.pickerOpen = false;
    actions.classId = cls.id;
    void actions.open('label');
  }

  function labelGroup(cls: { id: number }): void {
    groupPickerOpen = false;
    group.classId = cls.id;
    void group.open('label');
  }

  function ignore(reason: string): void {
    reasonsOpen = false;
    actions.reason = reason;
    void actions.open('exclude');
  }
</script>

<div
  class="relative flex flex-wrap items-center gap-x-2 gap-y-1 border-b border-line bg-surface-1/60 px-4 py-2 text-meta"
  role="toolbar"
  aria-label="Bulk actions"
  data-testid="bulk-bar"
>
  <span
    class="min-w-0 text-fg-1"
    class:text-danger-text={!!ws.summary.error}
    aria-live="polite"
    data-testid="bulk-summary">{text}</span
  >
  {#if count !== null}
    <span class="text-fg-3 tabular-nums" data-testid="bulk-count">{count} selected</span>
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
        <Icon name="chevron-down" size={18} />
      </button>
    {/if}
    {#if reasonsOpen}
      <div
        class="absolute left-0 top-full z-20 mt-1 w-44 rounded sm:left-auto sm:right-0 border border-line-strong bg-surface-1 py-1 shadow-lg"
        data-testid="ignore-reasons"
      >
        {#each excludeReasonsStore.reasons as r (r.id)}
          <button
            type="button"
            class="block w-full px-3 py-1 text-left text-meta text-fg-1 hover:bg-surface-2"
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
  {#if ws.inNearDupGroup}
    <button
      type="button"
      class="btn"
      data-testid="label-whole-group"
      onclick={() => (groupPickerOpen = true)}
    >
      Label whole duplicate group…
    </button>
  {/if}
  <DiverseSample {ws} />
  {@render extra?.()}
</div>

<ClassPicker
  open={ws.pickerOpen}
  mode="assign"
  onpick={assign}
  onclose={() => (ws.pickerOpen = false)}
/>
<ClassPicker
  open={groupPickerOpen}
  mode="assign"
  onpick={labelGroup}
  onclose={() => (groupPickerOpen = false)}
/>
<SelectionActionDialog controller={actions} {onapplied} subject={ws.actingOn} />
<SelectionActionDialog controller={group} {onapplied} />
