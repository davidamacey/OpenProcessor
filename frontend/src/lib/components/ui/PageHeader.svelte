<script lang="ts">
  /**
   * The one page heading: the h1 (20px, semibold, on every page), an optional description,
   * chips next to the title (`meta`) and the page's actions on the right. A page has one
   * primary action; it goes in `actions`.
   */
  import type { Snippet } from 'svelte';

  interface Props {
    title: string;
    description?: string;
    meta?: Snippet;
    actions?: Snippet;
  }
  let { title, description, meta, actions }: Props = $props();
</script>

<header class="mb-4 flex flex-wrap items-start justify-between gap-x-4 gap-y-2">
  <div class="min-w-0">
    <div class="flex flex-wrap items-center gap-2">
      <h1 class="text-h1 font-semibold text-fg-1">{title}</h1>
      {@render meta?.()}
    </div>
    {#if description}
      <p class="mt-0.5 text-body text-fg-2">{description}</p>
    {/if}
  </div>
  {#if actions}
    <div class="flex flex-wrap items-center gap-2">{@render actions()}</div>
  {/if}
</header>
