<script lang="ts">
  /**
   * A modal confirm step for a destructive or bulk action: the caller
   * supplies the title, the body (served facts), and what Confirm does.
   * Esc and a backdrop click cancel.
   */
  import type { Snippet } from 'svelte';
  import Modal from '$components/Modal.svelte';

  interface Props {
    title: string;
    confirmLabel?: string;
    danger?: boolean;
    busy?: boolean;
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
    confirmDisabled = false,
    onconfirm,
    oncancel,
    children,
  }: Props = $props();
</script>

<Modal
  label={title}
  onclose={oncancel}
  panelClass="max-h-[90vh] w-full max-w-lg overflow-y-auto rounded-lg border border-zinc-800 bg-zinc-950 p-5 shadow-2xl"
>
  <h3 class="mb-3 text-base font-semibold text-zinc-100">{title}</h3>
  <div class="mb-4 space-y-2 text-sm text-zinc-300">
    {@render children()}
  </div>
  <div class="flex justify-end gap-2">
    <button type="button" class="btn" onclick={oncancel}>Cancel</button>
    <button
      type="button"
      class="btn {danger ? 'btn-danger' : 'btn-primary'}"
      disabled={busy || confirmDisabled}
      onclick={onconfirm}
    >
      {busy ? 'Working…' : confirmLabel}
    </button>
  </div>
</Modal>
