<script lang="ts">
  /**
   * One binding's keys as removable chips, the "Change" button that starts a
   * capture (and the capture prompt while it runs), and the row's own extras.
   */
  import type { Snippet } from 'svelte';
  import { focusOnMount } from '$lib/actions/focusOnMount';
  import { formatShortcutKey } from '$lib/keyboardDisplay';
  import type { KeymapCardController } from '$components/settings/keymap/keymapCardController.svelte';

  interface Props {
    ctl: KeymapCardController;
    keys: string[];
    capturing: boolean;
    onremove: (key: string) => void;
    onstart: () => void;
    extra?: Snippet;
  }

  let { ctl, keys, capturing, onremove, onstart, extra }: Props = $props();
</script>

<div class="flex flex-wrap items-center gap-1">
  {#each keys as k (k)}
    <span
      class="flex items-center gap-1 rounded border border-zinc-700 bg-zinc-900 px-1.5 py-0.5 text-blue-300"
    >
      {formatShortcutKey(k)}
      <button
        class="text-zinc-500 hover:text-red-400"
        aria-label={`remove ${k}`}
        onclick={() => onremove(k)}
      >
        ×
      </button>
    </span>
  {/each}
  {#if capturing}
    <button
      data-capture
      use:focusOnMount
      class="rounded border border-blue-600 bg-blue-950/40 px-1.5 py-0.5 text-blue-300"
      onkeydown={ctl.onCaptureKeydown}
      onblur={ctl.cancelCapture}
    >
      press a key…
    </button>
  {:else}
    <button
      class="rounded border border-zinc-700 px-1.5 py-0.5 text-zinc-400 hover:text-white"
      onclick={onstart}
    >
      Change
    </button>
  {/if}
  {@render extra?.()}
</div>
