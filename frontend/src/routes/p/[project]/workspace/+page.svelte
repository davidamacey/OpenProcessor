<script lang="ts">
  /**
   * The Workspace: one page for looking at a project's items. The filter row,
   * the view bar, the queues and classes rail, the active view and the
   * details panel share one controller, so the filter, queue, scope and
   * selection survive a view switch and Back restores the previous view.
   */
  import { onDestroy } from 'svelte';
  import AdoptBanner from '$components/workspace/AdoptBanner.svelte';
  import DetailsPanel from '$components/workspace/DetailsPanel.svelte';
  import Lightbox from '$components/ui/Lightbox.svelte';
  import FilterBar from '$components/workspace/FilterBar.svelte';
  import ViewBar from '$components/workspace/ViewBar.svelte';
  import WorkspaceRail from '$components/workspace/WorkspaceRail.svelte';
  import { createWorkspaceController } from '$lib/workspace/workspaceController.svelte';
  import { availableViews, viewEntry } from '$lib/workspace/viewRegistry';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { nextStepsStore } from '$stores/nextSteps.svelte';

  // The default view is the served one of this visit: a cached one from before a clustering was
  // activated would open the Gallery on a project that now serves clusters.
  void nextStepsStore.enter();
  const ws = createWorkspaceController();
  onDestroy(() => ws.dispose());

  const entry = $derived(viewEntry(ws.view));
  const View = $derived(entry.component);

  $effect(() => {
    keyboardStore.setScope(entry.keyContext);
  });

  function stepView(by: 1 | -1): void {
    const views = availableViews(ws);
    const at = views.findIndex((v) => v.id === entry.id);
    ws.setView(views[(at + by + views.length) % views.length]!.id);
  }

  // Enter on a grid card opens Review at that card: the focused card, else the one the
  // details panel shows, else the single selected card.
  function openReview(): boolean {
    const card = document.activeElement?.closest<HTMLElement>('[data-crop-id]');
    const focused = ws.url.focus?.kind === 'item' ? ws.url.focus.id : null;
    const selected =
      ws.selection.kind === 'items' && ws.selection.cropIds.length === 1
        ? ws.selection.cropIds[0]!
        : null;
    const id = card?.dataset.cropId ?? focused ?? selected;
    if (!id) return false;
    ws.openReview(id);
    return true;
  }

  // Escape peels one layer: Review back to its grid (the selection is not shown there), the
  // selection, the open scope, then the panel.
  function escape(): void {
    if (ws.view === 'review') ws.closeReview();
    else if (ws.selection.kind !== 'none') ws.clearSelection();
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
      keyboardStore.registerAction(
        'workspace.open_editor',
        () => ws.openEditor(),
        'workspace',
      ),
      keyboardStore.registerAction(
        'workspace.lightbox',
        () => {
          // A card under the keyboard focus wins over the details focus.
          const card = document.activeElement?.closest<HTMLElement>('[data-image-id]');
          return void ws.openLightbox(card?.dataset.imageId);
        },
        'workspace',
      ),
      keyboardStore.registerAction(
        'workspace.class_picker',
        () => {
          if (ws.selection.kind !== 'none') ws.pickerOpen = true;
        },
        'workspace.items',
      ),
      keyboardStore.registerAction('workspace.open_review', openReview, 'cluster'),
      // False leaves Ctrl+A to the browser in a view with nothing to select.
      keyboardStore.registerAction(
        'workspace.select_all',
        () => ws.pressSelectAll(),
        'workspace',
      ),
      // Space on a focused card toggles it, like Ctrl+click.
      keyboardStore.registerAction(
        'workspace.select_toggle',
        () => {
          const el = document.activeElement;
          if (!(el instanceof HTMLButtonElement) || !el.hasAttribute('aria-pressed')) {
            return false;
          }
          el.dispatchEvent(new MouseEvent('click', { bubbles: true, ctrlKey: true }));
        },
        'workspace.items',
      ),
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
      {#if !ws.viewPending}
        {#if ['clusters', 'gallery', 'detections'].includes(ws.view)}
          <AdoptBanner />
        {/if}
        {#key entry.id}
          <View {ws} />
        {/key}
      {/if}
    </div>
    <DetailsPanel {ws} />
  </div>
  {#if ws.lightbox}
    <Lightbox
      imageId={ws.lightbox.imageId}
      label={ws.lightbox.label}
      onclose={() => ws.closeLightbox()}
      onprev={ws.lightboxNeighbour(-1) ? () => ws.stepLightbox(-1) : undefined}
      onnext={ws.lightboxNeighbour(1) ? () => ws.stepLightbox(1) : undefined}
    />
  {/if}
</div>
