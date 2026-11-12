<script lang="ts">
  import Icon from '$components/ui/Icon.svelte';
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
      class="flex items-center gap-1 rounded border border-line-strong bg-surface-1 px-1.5 py-0.5 text-accent-text"
    >
      {formatShortcutKey(k)}
      <button
        class="text-fg-3 hover:text-danger-text"
        aria-label={`remove ${k}`}
        onclick={() => onremove(k)}
      >
        <Icon name="x" size={14} />
      </button>
    </span>
  {/each}
  {#if capturing}
    <button
      data-capture
      use:focusOnMount
      class="rounded border border-accent bg-accent-soft px-1.5 py-0.5 text-accent-text"
      onkeydown={ctl.onCaptureKeydown}
      onblur={ctl.cancelCapture}
    >
      press a key…
    </button>
  {:else}
    <button
      class="rounded border border-line-strong px-1.5 py-0.5 text-fg-2 hover:text-white"
      onclick={onstart}
    >
      Change
    </button>
  {/if}
  {@render extra?.()}
</div>
