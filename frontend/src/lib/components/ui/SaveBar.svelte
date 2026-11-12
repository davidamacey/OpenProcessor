<script lang="ts">
  /**
   * The sticky Save / Discard footer of an editing page: it sticks to the bottom of the
   * scrolling pane, says whether there is anything to save, and keeps Save the page's one
   * primary action. Discard is hidden until something is dirty.
   */
  import Button from './Button.svelte';

  interface Props {
    dirty: boolean;
    saving?: boolean;
    onsave: () => void;
    ondiscard: () => void;
    /** What is dirty, in words (`3 shortcuts changed`); the default says unsaved changes. */
    message?: string;
  }
  let { dirty, saving = false, onsave, ondiscard, message }: Props = $props();
</script>

<div
  class="sticky bottom-0 z-10 -mx-4 mt-4 flex items-center gap-3 border-t border-line bg-surface-1/95 px-4 py-3 backdrop-blur md:-mx-gutter md:px-gutter"
  data-testid="save-bar"
>
  <span class="grow text-meta text-fg-2" role="status">
    {dirty ? (message ?? 'You have unsaved changes.') : 'No changes.'}
  </span>
  {#if dirty}
    <Button onclick={ondiscard} disabled={saving}>Discard</Button>
  {/if}
  <Button variant="primary" disabled={!dirty || saving} loading={saving} onclick={onsave}
    >Save</Button
  >
</div>
