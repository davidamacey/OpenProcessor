<script lang="ts">
  /**
   * A modal confirm step for a destructive or bulk action: the caller
   * supplies the title, the body (served facts), and what Confirm does.
   * Esc and a backdrop click cancel. Confirm takes focus the first time it
   * can be pressed, so Enter applies.
   */
  import type { Snippet } from 'svelte';
  import Modal from '$components/Modal.svelte';

  interface Props {
    title: string;
    confirmLabel?: string;
    danger?: boolean;
    busy?: boolean;
    /** What the confirm button says while `busy` (the served state, say); "Working…" otherwise. */
    busyLabel?: string;
    /** Extra buttons before Cancel (a "Run in background"). */
    actions?: Snippet;
    confirmDisabled?: boolean;
    onconfirm: () => void;
    oncancel: () => void;
    children: Snippet;
  }

  let {
    title,
    confirmLabel = 'Confirm',
    danger = false,
    busy = false,
    busyLabel = 'Working…',
    actions,
    confirmDisabled = false,
    onconfirm,
    oncancel,
    children,
  }: Props = $props();

  let confirmButton = $state<HTMLButtonElement | null>(null);
  let focusedOnce = false;
  $effect(() => {
    if (focusedOnce || busy || confirmDisabled || !confirmButton) return;
    focusedOnce = true;
    confirmButton.focus();
  });
</script>

<Modal
  label={title}
  onclose={oncancel}
  panelClass="max-h-[90vh] w-full max-w-lg overflow-y-auto rounded-lg border border-line bg-surface-0 p-5 shadow-2xl"
>
  <h3 class="mb-3 text-title font-semibold text-fg-1">{title}</h3>
  <div class="mb-4 space-y-2 text-body text-fg-2">
    {@render children()}
  </div>
  <div class="flex justify-end gap-2">
    {@render actions?.()}
    <button type="button" class="btn" onclick={oncancel}>Cancel</button>
    <button
      bind:this={confirmButton}
      type="button"
      class="btn {danger ? 'btn-danger' : 'btn-primary'}"
      disabled={busy || confirmDisabled}
      onclick={onconfirm}
    >
      {busy ? busyLabel : confirmLabel}
    </button>
  </div>
</Modal>
