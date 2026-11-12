<script lang="ts">
  /** The focused detection, as the server serves it. */
  import CropMetaPanel from '$components/CropMetaPanel.svelte';
  import ViewState from '$components/state/ViewState.svelte';
  import { getCrop } from '$lib/api_cropRead';
  import { apiErrorText } from '$lib/api_errors';
  import type { Crop } from '$lib/types';
  import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';

  let { id }: { ws: WorkspaceController; id: string | null } = $props();

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
    <CropMetaPanel {crop} onreprocessed={(c) => (crop = c)} />
  {/if}
</ViewState>
