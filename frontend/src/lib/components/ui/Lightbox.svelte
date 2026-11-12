<script lang="ts">
  /**
   * The full-size view of one image: the served pixels at 2048 px with the served boxes over
   * them, on the shared `Modal` (scrim, focus trap, Escape, focus return). Arrows step to the
   * neighbouring image when the caller has one; `+`, `-` and `0` zoom and reset.
   */
  import Modal from '$components/Modal.svelte';
  import SourceImageOverlay from '$components/SourceImageOverlay.svelte';
  import Button from './Button.svelte';

  interface Props {
    imageId: string;
    label?: string;
    onclose: () => void;
    /** Absent when there is no neighbour in that direction. */
    onprev?: () => void;
    onnext?: () => void;
  }
  let { imageId, label = 'Image', onclose, onprev, onnext }: Props = $props();

  const MIN = 1;
  const MAX = 4;
  const STEP = 0.5;
  let zoom = $state(1);

  function onkeydown(e: KeyboardEvent): void {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === 'ArrowLeft' && onprev) onprev();
    else if (e.key === 'ArrowRight' && onnext) onnext();
    else if (e.key === '+' || e.key === '=') zoom = Math.min(MAX, zoom + STEP);
    else if (e.key === '-') zoom = Math.max(MIN, zoom - STEP);
    else if (e.key === '0') zoom = 1;
    else return;
    e.preventDefault();
  }
</script>

<svelte:window {onkeydown} />

<Modal
  {label}
  {onclose}
  scrim="bg-black/95"
  panelClass="flex h-full w-full max-w-[96vw] flex-col"
  data-testid="lightbox"
>
  <div class="flex shrink-0 items-center gap-2 pb-2 text-fg-1">
    <span class="min-w-0 flex-1 truncate text-title font-semibold">{label}</span>
    <Button
      variant="ghost"
      size="sm"
      icon="chevron-left"
      aria-label="Previous image"
      disabled={!onprev}
      onclick={() => onprev?.()}
    />
    <Button
      variant="ghost"
      size="sm"
      icon="chevron-right"
      aria-label="Next image"
      disabled={!onnext}
      onclick={() => onnext?.()}
    />
    <Button
      variant="ghost"
      size="sm"
      icon="minus"
      aria-label="Zoom out"
      disabled={zoom <= MIN}
      onclick={() => (zoom = Math.max(MIN, zoom - STEP))}
    />
    <span class="w-12 text-center text-meta text-fg-2" data-testid="lightbox-zoom"
      >{Math.round(zoom * 100)}%</span
    >
    <Button
      variant="ghost"
      size="sm"
      icon="plus"
      aria-label="Zoom in"
      disabled={zoom >= MAX}
      onclick={() => (zoom = Math.min(MAX, zoom + STEP))}
    />
    <Button variant="ghost" size="sm" icon="x" aria-label="Close" onclick={onclose} />
  </div>
  <div class="min-h-0 flex-1 overflow-auto rounded-card bg-surface-0">
    <div style="width:{zoom * 100}%;height:{zoom * 100}%" data-testid="lightbox-stage">
      <SourceImageOverlay {imageId} maxDim={2048} />
    </div>
  </div>
</Modal>
