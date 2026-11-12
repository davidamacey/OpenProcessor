<script lang="ts">
  /**
   * The error page body, one for the app root (no project: full screen, a link to the project
   * list) and one for inside a project (`inShell`: the top bar stays, and the way out is the
   * dashboard). A 404 is not fixed by reloading; any other failure offers a reload and its
   * details.
   */
  import { page } from '$app/state';
  import { resolve } from '$app/paths';
  import Icon from '$components/ui/Icon.svelte';
  import { isChunkLoadError } from '$lib/chunkRecovery';
  import { projectHref } from '$lib/projectPaths';

  let { inShell = false }: { inShell?: boolean } = $props();

  const message = $derived(page.error?.message ?? '');
  const notFound = $derived(page.status === 404);
  const loadFailure = $derived(isChunkLoadError(page.error));
</script>

<div
  class="flex flex-col items-center justify-center gap-3 px-6 py-16 text-center {inShell
    ? 'min-h-full'
    : 'h-screen bg-surface-0 text-fg-1'}"
  data-testid="app-error"
>
  <Icon name={notFound ? 'search' : 'triangle-alert'} size={32} class="text-fg-3" />
  {#if notFound}
    <h1 class="text-h1 font-semibold">Page not found</h1>
    <p class="max-w-md text-body text-fg-2">
      There is nothing at this address. Check the link, or go back to
      {inShell ? 'the dashboard' : 'your projects'}.
    </p>
    <div class="mt-1 flex flex-wrap justify-center gap-2">
      {#if inShell}
        <a class="btn btn-primary" href={resolve(projectHref('/dashboard'))}>Dashboard</a>
        <a class="btn" href={resolve('/projects')}>All projects</a>
      {:else}
        <a class="btn btn-primary" href={resolve('/projects')}>All projects</a>
      {/if}
    </div>
  {:else}
    <h1 class="text-h1 font-semibold">
      {loadFailure ? "This page couldn't finish loading" : 'Something went wrong'}
    </h1>
    <p class="max-w-md text-body text-fg-2">
      {loadFailure
        ? 'The connection may have dropped, or the app was updated while this tab was open. Reloading usually fixes it.'
        : 'Reloading may help. If it keeps happening, check the details below.'}
    </p>
    <button
      type="button"
      class="btn btn-primary mt-1"
      onclick={() => window.location.reload()}
    >
      <Icon name="refresh-cw" size={14} />Reload
    </button>
  {/if}
  {#if message}
    <details class="max-w-xl text-left text-meta text-fg-3">
      <summary class="cursor-pointer">Details</summary>
      <p class="mt-1 break-words" data-testid="app-error-detail">
        {page.status}: {message}
      </p>
    </details>
  {/if}
</div>
