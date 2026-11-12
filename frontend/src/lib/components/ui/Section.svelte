<script lang="ts">
  /**
   * A titled block inside a page: an h2 (16px), an optional description, actions on the
   * right, then the content. Not a card; wrap the content in `Card` when it needs a surface.
   */
  import type { Snippet } from 'svelte';

  interface Props {
    title: string;
    description?: string;
    actions?: Snippet;
    children: Snippet;
  }
  let { title, description, actions, children }: Props = $props();
  const id = `section-${Math.random().toString(36).slice(2, 9)}`;
</script>

<section class="mb-6" aria-labelledby={id}>
  <div class="mb-2 flex flex-wrap items-end justify-between gap-x-4 gap-y-1">
    <div class="min-w-0">
      <h2 {id} class="text-title font-semibold text-fg-1">{title}</h2>
      {#if description}
        <p class="text-meta text-fg-3">{description}</p>
      {/if}
    </div>
    {#if actions}
      <div class="flex flex-wrap items-center gap-2">{@render actions()}</div>
    {/if}
  </div>
  {@render children()}
</section>
