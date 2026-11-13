<script lang="ts">
  /**
   * The one loading / error / empty / ready frame of a Workspace view. It
   * renders the served empty message and one served action, a skeleton while
   * the first page loads, and never a spinner over content.
   */
  import type { Snippet } from 'svelte';
  import { resolve } from '$app/paths';
  import { projectHref, type ProjectSubPath } from '$lib/projectPaths';

  interface Props {
    status: 'loading' | 'error' | 'empty' | 'ready';
    /** The served sentence for `empty`, or the error text for `error`. */
    message?: string | null;
    /** One served way forward, shown with `empty` and `error`. */
    action?: { label: string; path: ProjectSubPath } | null;
    onretry?: () => void;
    children: Snippet;
  }

  let { status, message = null, action = null, onretry, children }: Props = $props();
</script>

{#if status === 'ready'}
  {@render children()}
{:else if status === 'loading'}
  <div
    class="grid grid-cols-2 gap-3 p-4 sm:grid-cols-3 lg:grid-cols-5"
    role="status"
    aria-label="Loading"
    data-testid="view-state-loading"
  >
    {#each Array.from({ length: 10 }, (_, i) => i) as i (i)}
      <div class="aspect-square animate-pulse rounded-lg bg-surface-2/60"></div>
    {/each}
  </div>
{:else}
  <div
    class="flex flex-col items-center gap-3 p-8 text-center text-body"
    role={status === 'error' ? 'alert' : 'status'}
    data-testid="view-state-{status}"
  >
    {#if message}
      <p class={status === 'error' ? 'text-danger-text' : 'text-fg-2'}>{message}</p>
    {/if}
    {#if status === 'error' && onretry}
      <button type="button" class="btn btn-sm" onclick={onretry}>Retry</button>
    {/if}
    {#if action}
      <a class="btn btn-sm btn-primary" href={resolve(projectHref(action.path))}
        >{action.label}</a
      >
    {/if}
  </div>
{/if}
