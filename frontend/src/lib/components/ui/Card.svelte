<script lang="ts">
  /**
   * The card surface: surface-1, a hairline, radius 10 and elevation e1. `selected` adds the
   * accent ring and a check mark, so selection never rests on colour alone. `padded` is the
   * 16px card padding.
   */
  import type { Snippet } from 'svelte';
  import type { HTMLAttributes } from 'svelte/elements';
  import Icon from './Icon.svelte';

  interface Props extends Omit<HTMLAttributes<HTMLDivElement>, 'children' | 'class'> {
    padded?: boolean;
    selected?: boolean;
    class?: string;
    children: Snippet;
  }
  let {
    padded = true,
    selected = false,
    class: className = '',
    children,
    ...rest
  }: Props = $props();
</script>

<div
  {...rest}
  class="surface relative {padded ? 'p-4' : ''} {selected
    ? 'border-accent ring-2 ring-accent'
    : ''} {className}"
  data-selected={selected || undefined}
>
  {@render children()}
  {#if selected}
    <span
      class="absolute top-2 right-2 flex size-5 items-center justify-center rounded-full bg-accent-fill text-white"
    >
      <Icon name="check" size={12} label="Selected" />
    </span>
  {/if}
</div>
