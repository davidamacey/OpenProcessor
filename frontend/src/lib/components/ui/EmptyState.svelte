<script lang="ts">
  /**
   * An empty list or page: an icon, what is missing, why, and ONE next step (`action`).
   * Every list has an empty, a loading and an error state; this is the empty one.
   */
  import type { Snippet } from 'svelte';
  import Icon from './Icon.svelte';
  import type { IconName } from './iconRegistry';

  interface Props {
    title: string;
    description?: string;
    icon?: IconName;
    action?: Snippet;
  }
  let { title, description, icon = 'square-dashed', action }: Props = $props();
</script>

<div
  class="flex flex-col items-center gap-2 rounded-card border border-dashed border-line-strong px-6 py-10 text-center"
  data-testid="empty-state"
>
  <Icon name={icon} size={28} class="text-fg-3" />
  <p class="text-title font-semibold text-fg-1">{title}</p>
  {#if description}
    <p class="max-w-md text-body text-fg-2">{description}</p>
  {/if}
  {#if action}
    <div class="mt-2">{@render action()}</div>
  {/if}
</div>
