<script lang="ts">
  import Icon from '$components/ui/Icon.svelte';
  /**
   * One image in the editor: the picture and its boxes, the tools, the objects, the filmstrip of the
   * list it was opened from, the class picker for a drawn box and the save state. Everything it
   * shows is what the server last served for this image.
   */
  import { onDestroy, untrack } from 'svelte';
  import { goto } from '$app/navigation';
  import { page } from '$app/state';
  import { resolve } from '$app/paths';
  import ClassPicker from '$components/ClassPicker.svelte';
  import EditorCanvas from '$components/editor/EditorCanvas.svelte';
  import EditorHeader from '$components/editor/EditorHeader.svelte';
  import Filmstrip from '$components/Filmstrip.svelte';
  import ObjectList from '$components/editor/ObjectList.svelte';
  import ToolStrip from '$components/editor/ToolStrip.svelte';
  import SlotBboxEditor from '$components/SlotBboxEditor.svelte';
  import ViewState from '$components/state/ViewState.svelte';
  import {
    getImageFileUrl,
    getImageThumbUrl,
    listImages,
    locateImage,
  } from '$lib/api_images';
  import { slotRegistry } from '$lib/annotations/registeredSlots';
  import { subBoxSlotFor } from '$lib/annotations/cropSlots';
  import { createEditorController } from '$lib/editor/editorController.svelte';
  import { registerEditorKeys } from '$lib/editor/editorKeys.svelte';
  import { editorHref, editorListQuery, workspaceBackHref } from '$lib/editor/editorUrl';
  import { setCrumbName } from '$stores/crumbName.svelte';

  interface Props {
    imageId: string;
    /** The Workspace query string the editor was opened with. */
    search: string;
  }

  let { imageId, search }: Props = $props();

  // The page keys this component on the image, so both props are fixed for its life.
  const id = untrack(() => imageId);
  const query = untrack(() => search);
  const listQuery = editorListQuery(new URLSearchParams(query));
  const c = createEditorController({
    imageId: id,
    locate: (signal) => locateImage(listQuery, id, signal),
    fetchPage: (cursor, signal) => listImages(listQuery, cursor, signal),
    navigate: (next) =>
      void goto(resolve(next ? editorHref(next, query) : workspaceBackHref(query))),
    regionSlot: (item) => subBoxSlotFor(item, slotRegistry.all),
  });
  const backHref = workspaceBackHref(query);
  onDestroy(() => c.dispose());

  // Registering reads and writes the keyboard scope, so it must not track what it reads.
  $effect(() => untrack(() => registerEditorKeys(c)));

  let canvas = $state<EditorCanvas | null>(null);
  $effect(() => {
    c.attachCanvas(canvas);
    return () => c.attachCanvas(null);
  });

  let objectsOpen = $state(false);
  const fileName = $derived(c.image?.image.file_name ?? id);
  // The breadcrumb's last crumb is the file name, not the id.
  $effect(() => {
    if (c.image) setCrumbName(page.url.pathname, fileName);
  });
  const status = $derived(c.loadError ? 'error' : c.image ? 'ready' : 'loading');
  const dropCount = $derived(c.drop?.regionBoxIds.length ?? 0);
</script>

<div class="flex h-full min-h-0 flex-col" data-testid="image-editor">
  <EditorHeader
    {backHref}
    {fileName}
    index={c.filmstrip?.index ?? null}
    total={c.filmstrip && c.filmstrip.index !== null ? c.filmstrip.total : null}
    status={c.stack.status}
    message={c.stack.message}
  />
  {#if c.filmstrip?.message}
    <p class="border-b border-line px-3 py-1.5 text-meta text-fg-2">
      {c.filmstrip.message}
    </p>
  {/if}

  {#if c.drop}
    <div
      class="flex flex-wrap items-center gap-2 border-b border-warn/40 bg-warn/10 px-3 py-1.5 text-body text-warn-text"
      role="alert"
      data-testid="drop-regions"
    >
      <span>
        This move leaves {dropCount} region {dropCount === 1 ? 'box' : 'boxes'} outside the
        item. Delete {dropCount === 1 ? 'it' : 'them'} with the move?
      </span>
      <button type="button" class="btn" onclick={() => c.confirmDrop()}>
        Move and delete
      </button>
      <button type="button" class="btn" onclick={() => c.dismissDrop()}>Cancel</button>
    </div>
  {/if}

  <ViewState {status} message={c.loadError ?? ''}>
    {#if c.image}
      <div class="flex min-h-0 flex-1 flex-col lg:flex-row">
        <div class="shrink-0 border-b border-line p-2 lg:w-14 lg:border-r lg:border-b-0">
          <ToolStrip
            tool={c.tool}
            sticky={c.sticky}
            labels={c.labels}
            zoomPercent={c.zoomPercent}
            canUndo={c.stack.canUndo}
            canRedo={c.stack.canRedo}
            undoLabel={c.stack.undoLabel}
            redoLabel={c.stack.redoLabel}
            ontool={(t, keep) => c.setTool(t, keep)}
            onzoomin={c.zoomIn}
            onzoomout={c.zoomOut}
            onfit={c.zoomFit}
            onactual={c.zoomActual}
            onundo={c.undo}
            onredo={c.redo}
            onlabels={c.toggleLabels}
          />
        </div>

        <div class="min-h-[40vh] min-w-0 flex-1">
          <EditorCanvas
            bind:this={canvas}
            imageUrl={(d) => getImageFileUrl(id, d)}
            width={c.image.image.width}
            height={c.image.image.height}
            boxes={c.boxes}
            selectedCropId={c.selectedCropId}
            locks={c.locks}
            tool={c.tool}
            showLabels={c.labels}
            busy={c.picker !== null}
            pending={c.pendingBox}
            ondraw={c.draw}
            onmove={c.move}
            onselect={c.select}
            ongesture={(g) => (c.gesture = g)}
            onzoom={(p) => (c.zoomPercent = p)}
          />
        </div>

        <aside
          class="shrink-0 border-t border-line lg:w-72 lg:overflow-y-auto lg:border-t-0 lg:border-l"
          aria-label="Objects"
        >
          <button
            type="button"
            class="flex w-full items-center justify-between px-3 py-2 text-body text-fg-1 lg:hidden"
            aria-expanded={objectsOpen}
            aria-controls="editor-objects"
            data-testid="objects-toggle"
            onclick={() => (objectsOpen = !objectsOpen)}
          >
            <span>Objects ({c.objects.length})</span>
            <Icon name={objectsOpen ? 'chevron-down' : 'chevron-right'} size={14} />
          </button>
          <div
            id="editor-objects"
            class="max-h-[40vh] overflow-y-auto p-2 lg:block lg:max-h-none {objectsOpen
              ? 'block'
              : 'hidden'}"
          >
            <ObjectList
              rows={c.objects}
              selectedCropId={c.selectedCropId}
              onselect={c.select}
              onclass={(id) => {
                c.select(id);
                c.openClassPicker();
              }}
              ondelete={(id) => {
                c.select(id);
                void c.deleteSelected();
              }}
            />
          </div>
        </aside>
      </div>

      {#if c.filmstrip && c.filmstrip.index !== null}
        <div class="shrink-0 border-t border-line p-2">
          <Filmstrip
            tiles={[
              ...(c.filmstrip.before ? [c.filmstrip.before] : []),
              id,
              ...c.filmstrip.after,
            ].map((other) => ({
              id: other,
              label: other === id ? 'This image' : `Open image ${other}`,
              thumbUrl: getImageThumbUrl(other, 96),
              href: editorHref(other, query),
            }))}
            focusedId={id}
            navLabel="Images in this list"
          />
        </div>
      {/if}
    {/if}
  </ViewState>

  <ClassPicker
    open={c.picker !== null}
    mode="assign"
    recent={c.recentClasses}
    onpick={(cls) => void c.pick(cls)}
    onclose={c.cancelPicker}
  />

  {#if c.regionsOpen && c.selected && c.regionSlot}
    <SlotBboxEditor
      crop={c.selected}
      slot={c.regionSlot}
      onclose={c.closeRegions}
      onsave={() => void c.regionsSaved()}
    />
  {/if}
</div>
