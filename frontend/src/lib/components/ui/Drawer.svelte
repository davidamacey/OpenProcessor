<script lang="ts">
  /**
   * A panel flush to the left or right edge, for the mobile nav and for previews that should
   * not take the page. It is the shared `Modal` (scrim, focus trap, Escape, focus return) in
   * its edge alignment with an elevation-2 panel, so there is one dialog implementation.
   */
  import type { Snippet } from 'svelte';
  import Modal from '$components/Modal.svelte';
  import Button from './Button.svelte';

  interface Props {
    label: string;
    onclose: () => void;
    side?: 'left' | 'right';
    /** Panel width class; defaults to 20rem, capped at the viewport. */
    widthClass?: string;
    children: Snippet;
  }
  let { label, onclose, side = 'left', widthClass = 'w-80', children }: Props = $props();
</script>

<Modal
  {label}
  {onclose}
  layer="drawer"
  align={side === 'left' ? 'start' : 'end'}
  panelClass="flex h-full max-w-full flex-col bg-surface-1 shadow-e2 {widthClass} {side ===
  'left'
    ? 'border-r'
    : 'border-l'} border-line"
  data-testid="drawer"
>
  <div
    class="flex shrink-0 items-center justify-between gap-2 border-b border-line px-4 py-2"
  >
    <span class="text-title font-semibold text-fg-1">{label}</span>
    <Button variant="ghost" size="sm" icon="x" aria-label="Close" onclick={onclose} />
  </div>
  <div class="min-h-0 flex-1 overflow-y-auto">
    {@render children()}
  </div>
</Modal>
