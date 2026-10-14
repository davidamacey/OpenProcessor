<!--
  The /review in-app reject-reason prompt (replaces window.prompt()).
-->
<script lang="ts">
  import Modal from '$components/Modal.svelte';
  import { focusOnMount } from '$lib/actions/focusOnMount';
  import type { SlotSpec } from '$lib/annotations/types';

  interface Props {
    activeSlot: SlotSpec | null;
    rejectReasonPromptOpen: boolean;
    rejectReasonPromptValue: string;
    submitRejectReasonPrompt: () => void;
    cancelRejectReasonPrompt: () => void;
  }

  let {
    activeSlot,
    rejectReasonPromptOpen,
    rejectReasonPromptValue = $bindable(),
    submitRejectReasonPrompt,
    cancelRejectReasonPrompt,
  }: Props = $props();
</script>

{#if rejectReasonPromptOpen}
  <!-- DQ-m6: in-app replacement for the old window.prompt() — see
       promptForRejectionReason()'s doc comment for why. Backdrop click
       and Esc both cancel (reason stays null, matching the old
       "Cancel" prompt() behavior); Enter submits. -->
  <Modal
    label="Reject reason"
    onclose={cancelRejectReasonPrompt}
    layer="top"
    align="top"
    scrim="bg-black/50"
    panelClass="w-full max-w-sm overflow-hidden rounded-lg border border-zinc-700 bg-zinc-900 p-3 shadow-xl"
  >
    <label class="mb-2 block text-sm text-zinc-300" for="reject-reason-input">
      Reason for rejecting this {activeSlot?.label.singular ?? 'item'} (optional):
    </label>
    <input
      id="reject-reason-input"
      type="text"
      use:focusOnMount
      bind:value={rejectReasonPromptValue}
      onkeydown={(e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          submitRejectReasonPrompt();
        } else if (e.key === 'Escape') {
          e.preventDefault();
          cancelRejectReasonPrompt();
        }
      }}
      class="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm focus:border-blue-500 focus:outline-none"
      placeholder={`e.g. blurry, wrong angle, not a ${activeSlot?.label.singular ?? 'match'}…`}
    />
    <div class="mt-3 flex justify-end gap-2">
      <button type="button" class="btn" onclick={cancelRejectReasonPrompt}>
        Cancel
      </button>
      <button type="button" class="btn btn-primary" onclick={submitRejectReasonPrompt}>
        Reject
      </button>
    </div>
  </Modal>
{/if}
