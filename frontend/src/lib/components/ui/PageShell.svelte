<script lang="ts">
  /**
   * The page body wrapper: one width scale and one gutter for every page, so a page no
   * longer picks its own `max-w-*` (Quality and Settings used to jump width between tabs).
   * `default` is 6xl, `wide` is 7xl, `full` fills the pane (editors and the Workspace).
   */
  import type { Snippet } from 'svelte';
  import type { HTMLAttributes } from 'svelte/elements';

  interface Props extends Omit<HTMLAttributes<HTMLDivElement>, 'children' | 'class'> {
    width?: 'default' | 'wide' | 'full';
    class?: string;
    children: Snippet;
  }
  let { width = 'default', class: className = '', children, ...rest }: Props = $props();

  const MAX = { default: 'max-w-6xl', wide: 'max-w-7xl', full: '' } as const;
</script>

<div
  {...rest}
  class="mx-auto w-full px-4 py-6 md:px-gutter {MAX[width]} {className}"
  data-page-width={width}
>
  {@render children()}
</div>
