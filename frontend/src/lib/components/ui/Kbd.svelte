<script lang="ts">
  /**
   * A key hint. `combo` is a normalized combo as the keymap serves it (`l`, `shift+space`,
   * `escape`); the component spells it with capitals (L, Shift+Space, Esc), so no call site
   * restyles or re-cases a key. Children are for text that is not a key (a symbol such as `/`).
   * Styling is the `kbd` rule in app.css.
   */
  import type { Snippet } from 'svelte';
  import { formatShortcutKey } from '$lib/keyboardDisplay';

  interface Props {
    combo?: string;
    class?: string;
    children?: Snippet;
  }
  let { combo, class: className, children }: Props = $props();
</script>

<kbd class={className}
  >{#if combo}{formatShortcutKey(combo)}{:else}{@render children?.()}{/if}</kbd
>
