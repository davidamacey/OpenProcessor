<script lang="ts">
  /** The focused detection, as the server serves it. */
  import CropMetaPanel from '$components/CropMetaPanel.svelte';
  import ViewState from '$components/state/ViewState.svelte';
  import { getCrop } from '$lib/api_cropRead';
  import { apiErrorText } from '$lib/api_errors';
  import type { Crop } from '$lib/types';
  import { resolve } from '$app/paths';
  import { page } from '$app/state';
  import { keymapStore } from '$stores/keymap.svelte';
  import { editorHref } from '$lib/editor/editorUrl';
  import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';

  let {
    ws,
    id,
    embedded = false,
  }: { ws: WorkspaceController; id: string | null; embedded?: boolean } = $props();

  let crop = $state<Crop | null>(null);
  let error = $state<string | null>(null);

  $effect(() => {
    const cropId = id;
    crop = null;
    error = null;
    if (!cropId) return;
    const ctrl = new AbortController();
    getCrop(cropId, ctrl.signal)
      .then((c) => (crop = c))
      .catch((e: unknown) => {
        if (!ctrl.signal.aborted) error = apiErrorText(e);
      });
    return () => ctrl.abort();
  });
  const status = $derived(error ? 'error' : crop ? 'ready' : id ? 'loading' : 'empty');
</script>

<ViewState {status} message={error ?? 'Pick a detection to see its details.'}>
  {#if crop}
    {#if crop.image_id}
      <a
        class="btn mb-3 inline-block"
        title="Edit boxes ({keymapStore.glyph('workspace.open_editor')})"
        href={resolve(editorHref(crop.image_id, page.url.search))}
        data-testid="open-editor"
      >
        Edit boxes
      </a>
    {/if}
    <CropMetaPanel
      {crop}
      {embedded}
      onreprocessed={(c) => {
        crop = c;
        ws.adoptReviewItem(c);
      }}
    />
  {/if}
</ViewState>
