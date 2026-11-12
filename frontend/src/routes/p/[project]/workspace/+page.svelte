<script lang="ts">
  /**
   * The Workspace: one page for looking at a project's items. The filter row,
   * the view bar, the queues and classes rail, the active view and the
   * details panel share one controller, so the filter, queue, scope and
   * selection survive a view switch and Back restores the previous view.
   */
  import { onDestroy } from 'svelte';
  import DetailsPanel from '$components/workspace/DetailsPanel.svelte';
  import FilterBar from '$components/workspace/FilterBar.svelte';
  import ViewBar from '$components/workspace/ViewBar.svelte';
  import WorkspaceRail from '$components/workspace/WorkspaceRail.svelte';
  import { createWorkspaceController } from '$lib/workspace/workspaceController.svelte';
  import { VIEWS, viewEntry } from '$lib/workspace/viewRegistry';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { nextStepsStore } from '$stores/nextSteps.svelte';

  const ws = createWorkspaceController();
  onDestroy(() => ws.dispose());

  const entry = $derived(viewEntry(ws.view));
  const View = $derived(entry.component);

  $effect(() => {
    void nextStepsStore.init();
  });

  $effect(() => {
    keyboardStore.setScope(entry.keyContext);
  });

  function stepView(by: 1 | -1): void {
    const at = VIEWS.findIndex((v) => v.id === entry.id);
    ws.setView(VIEWS[(at + by + VIEWS.length) % VIEWS.length]!.id);
  }

  // Escape peels one layer: the selection, then the open scope, then the panel.
  function escape(): void {
    if (ws.selection.kind !== 'none') ws.clearSelection();
    else if (ws.url.scope) ws.setScope(null);
    else if (ws.detailsOpen) ws.toggleDetails();
  }

  $effect(() => {
    const offs = [
      keyboardStore.registerAction(
        'workspace.view_prev',
        () => stepView(-1),
        'workspace',
      ),
      keyboardStore.registerAction('workspace.view_next', () => stepView(1), 'workspace'),
      keyboardStore.registerAction(
        'workspace.toggle_details',
        () => ws.toggleDetails(),
        'workspace',
      ),
      keyboardStore.registerAction('workspace.escape', escape, 'workspace'),
    ];
    return () => offs.forEach((off) => off());
  });
</script>

<div class="flex h-full flex-col" data-testid="workspace-page">
  <FilterBar {ws} />
  <ViewBar {ws} />
  <div class="flex min-h-0 flex-1">
    <WorkspaceRail {ws} />
    <div class="min-w-0 flex-1 overflow-auto" data-testid="workspace-view">
      {#key entry.id}
        <View {ws} />
      {/key}
    </div>
    <DetailsPanel {ws} />
  </div>
</div>
