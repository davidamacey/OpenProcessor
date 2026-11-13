<script lang="ts">
  /**
   * The "More" menu of a view header: the view's own rare actions, then the ones every item
   * view shares (a diverse sample, undo of the last change). Opening "Select a diverse sample"
   * shows its dialog from here, so the actions sit in one place and take no row of their own.
   */
  import Menu, { type MenuItem } from '$components/ui/Menu.svelte';
  import DiverseSample from '$components/workspace/DiverseSample.svelte';
  import FindDisagreementsDialog from '$components/workspace/FindDisagreementsDialog.svelte';
  import { FIND_DISAGREEMENTS } from '$lib/probe';
  import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';
  import { undoStore } from '$stores/undo.svelte';

  let { ws, items = [] }: { ws: WorkspaceController; items?: MenuItem[] } = $props();

  let diverseOpen = $state(false);
  let disagreementsOpen = $state(false);
  const all = $derived<MenuItem[]>([
    ...items,
    {
      id: 'diverse-sample',
      label: 'Select a diverse sample\u2026',
      testid: 'diverse-sample-open',
      onselect: () => (diverseOpen = true),
    },
    {
      id: 'find-disagreements',
      label: `${FIND_DISAGREEMENTS}\u2026`,
      testid: 'find-disagreements-open',
      onselect: () => (disagreementsOpen = true),
    },
    {
      id: 'undo',
      label: 'Undo last change',
      testid: 'undo-last',
      icon: 'undo-2',
      onselect: () => void undoStore.undoLast(),
    },
  ]);
</script>

<Menu label="More" items={all} align="end" triggerClass="btn" triggerTestid="view-more" />
<DiverseSample {ws} bind:open={diverseOpen} />
{#if disagreementsOpen}
  <FindDisagreementsDialog onclose={() => (disagreementsOpen = false)} />
{/if}
