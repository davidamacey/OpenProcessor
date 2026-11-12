<script lang="ts">
  /** The Review item's details: the rest of what the server knows, beside the Review pane. */
  import ItemDetails from './ItemDetails.svelte';
  import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';

  let { ws, id }: { ws: WorkspaceController; id: string | null } = $props();
</script>

<ItemDetails {ws} {id} embedded />
