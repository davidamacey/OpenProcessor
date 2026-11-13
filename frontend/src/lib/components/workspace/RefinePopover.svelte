<script lang="ts">
  /**
   * The "Refine" control of a view: one button that opens the filters and orderings the view
   * keeps out of the way, with the number of refinements in force. The panel is a popover over
   * the view, so opening it moves nothing. A view puts its own groups inside.
   */
  import type { Snippet } from 'svelte';
  import Icon from '$components/ui/Icon.svelte';
  import Popover from '$components/ui/Popover.svelte';

  interface Props {
    /** Refinements in force; shown as "Refine (2)", nothing at zero. */
    count?: number;
    testid?: string;
    panelClass?: string;
    children: Snippet<[() => void]>;
  }
  let {
    count = 0,
    testid = 'refine-toggle',
    panelClass = 'w-[34rem] max-h-[70vh] overflow-y-auto p-3',
    children: body,
  }: Props = $props();
</script>

<Popover label="Refine" align="end" {panelClass}>
  {#snippet trigger({ open, attrs })}
    <button
      type="button"
      class="btn {count > 0 || open ? 'border-accent/60 text-accent-text' : ''}"
      data-testid={testid}
      {...attrs}
    >
      <Icon name="funnel" size={14} />
      Refine{count > 0 ? ` (${count})` : ''}
    </button>
  {/snippet}
  {#snippet children(close)}
    {@render body(close)}
  {/snippet}
</Popover>
