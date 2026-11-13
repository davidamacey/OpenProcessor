<script lang="ts">
  import Icon from '$components/ui/Icon.svelte';
  /**
   * What the selection is and what an action will change. The sentence is the
   * server's (`POST /selection/summary`); every action opens the server's
   * dry run first and writes only on confirm. Ignore reasons are the served
   * list. A view adds its own buttons through `extra`.
   */
  import ClassPicker from '$components/ClassPicker.svelte';
  import Menu, { type MenuItem } from '$components/ui/Menu.svelte';
  import SelectionActionDialog from '$components/itemFilter/SelectionActionDialog.svelte';
  import { createSelectionActionController } from '$lib/itemFilter/selectionActionController.svelte';
  import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';
  import { classesStore } from '$stores/classes.svelte';
  import { excludeReasonsStore } from '$stores/excludeReasons.svelte';
  import { keymapStore } from '$stores/keymap.svelte';
  import { undoStore } from '$stores/undo.svelte';

  interface Props {
    ws: WorkspaceController;
    /** Reload the view after a write landed. */
    onapplied?: () => void;
    /** The write was queued and the host moves on (the label-then-next flow). */
    ondetached?: () => void;
    /** The view's own rare actions, listed first under More. */
    more?: MenuItem[];
  }

  let { ws, onapplied, ondetached, more = [] }: Props = $props();

  const actions = createSelectionActionController(() => ws.itemSelection() ?? {}, {
    detachOnQueued: () => ws.labelThenNext && ws.openNextAfterLabel,
    ondetached: () => ondetached?.(),
  });
  // The open near-duplicate group as a whole: the filter carries `near_dup_group`, so the label
  // reaches every member the server counts, not only the ones loaded.
  const group = createSelectionActionController(() => ({ filter: ws.itemBody() }));
  let groupPickerOpen = $state(false);
  let reasonsOpen = $state(false);

  // A whole-cluster label of a cluster the detector split across classes says so before it writes.
  const mixedWarning = $derived.by((): string | null => {
    const card = ws.scopeCard;
    if (!ws.labelThenNext || !card?.flags?.includes('mixed_detector_classes'))
      return null;
    const cls = classesStore.classes.find((c) => c.id === actions.classId)?.name;
    const top = card.detector_top_class
      ? `the detector's most common class here is ${card.detector_top_class}${
          card.detector_top_share != null
            ? ` (${Math.round(card.detector_top_share * 100)}%)`
            : ''
        }`
      : 'the detector named more than one class here';
    return `This cluster is mixed: ${top}. Labelling all ${actions.selected ?? card.size} as ${cls ?? 'one class'} also labels the others. Tip: sort by Outliers first and label or move those first.`;
  });

  const moreItems = $derived<MenuItem[]>([
    ...more,
    {
      id: 'restore',
      label: 'Restore ignored\u2026',
      onselect: () => void actions.open('unexclude'),
    },
    ...(ws.inNearDupGroup
      ? [
          {
            id: 'label-group',
            label: 'Label whole duplicate group\u2026',
            testid: 'label-whole-group',
            onselect: () => (groupPickerOpen = true),
          },
        ]
      : []),
    {
      id: 'undo',
      label: 'Undo last change',
      icon: 'undo-2' as const,
      onselect: () => void undoStore.undoLast(),
    },
  ]);

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

{#if !hasSelection && !ws.pendingSelectAll && ws.inNearDupGroup}
  <!-- An open duplicate group is labelled whole, with nothing selected. -->
  <div
    class="flex flex-wrap items-center gap-2 border-b border-line bg-surface-1/60 px-4 py-2 text-meta"
    role="toolbar"
    aria-label="Duplicate group"
    data-testid="group-bar"
  >
    <span class="text-fg-1">This duplicate group</span>
    <button
      type="button"
      class="btn"
      data-testid="label-whole-group"
      onclick={() => (groupPickerOpen = true)}
    >
      Label whole group…
    </button>
  </div>
{:else if hasSelection || ws.pendingSelectAll}
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
      <span class="text-fg-3 tabular-nums" data-testid="bulk-count">{count} selected</span
      >
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
      class="btn btn-primary"
      disabled={!hasSelection}
      title={keymapStore.glyph('workspace.class_picker')}
      data-testid="assign-selected"
      onclick={() => (ws.pickerOpen = true)}
    >
      Label as…
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
          class="absolute right-0 top-full z-20 mt-1 w-44 rounded border border-line-strong bg-surface-1 py-1 shadow-lg"
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
    <Menu
      label="More"
      items={moreItems}
      align="end"
      triggerClass="btn"
      triggerTestid="bulk-more"
    />
    <button
      type="button"
      class="btn"
      title={keymapStore.glyph('workspace.escape')}
      onclick={() => ws.clearSelection()}
    >
      Clear
    </button>
  </div>
{/if}

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
<SelectionActionDialog
  controller={actions}
  {onapplied}
  subject={ws.actingOn}
  warning={mixedWarning}
  openNext={ws.labelThenNext
    ? { checked: ws.openNextAfterLabel, onchange: (v) => (ws.openNextAfterLabel = v) }
    : undefined}
/>
<SelectionActionDialog controller={group} {onapplied} />
