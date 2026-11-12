<script lang="ts">
  import { resolve } from '$app/paths';
  import { page } from '$app/state';
  import { isChunkLoadError } from '$lib/chunkRecovery';

  const message = $derived(page.error?.message ?? '');
  const notFound = $derived(page.status === 404);
  const loadFailure = $derived(isChunkLoadError(page.error));
</script>

<div
  class="flex h-screen flex-col items-center justify-center gap-3 bg-surface-0 px-6 text-center text-fg-1"
  data-testid="app-error"
>
  {#if notFound}
    <p class="text-lg font-semibold">Page not found</p>
    <p class="max-w-md text-sm text-fg-2">
      There is nothing at this address. Check the link, or go back to your projects.
    </p>
    <a
      href={resolve('/projects')}
      class="rounded border border-line-strong px-3 py-1.5 text-sm hover:border-fg-3"
      >All projects</a
    >
  {:else}
    <p class="text-lg font-semibold">
      {loadFailure ? "This page couldn't finish loading" : 'Something went wrong'}
    </p>
    <p class="max-w-md text-sm text-fg-2">
      {loadFailure
        ? 'The connection may have dropped, or the app was updated while this tab was open. Reloading usually fixes it.'
        : 'Reloading may help. If it keeps happening, check the details below.'}
    </p>
    <button
      type="button"
      class="rounded border border-line-strong px-3 py-1.5 text-sm hover:border-fg-3"
      onclick={() => window.location.reload()}
    >
      Reload
    </button>
  {/if}
  {#if message}
    <details class="max-w-xl text-left text-xs text-fg-3">
      <summary class="cursor-pointer">Details</summary>
      <p class="mt-1 break-words" data-testid="app-error-detail">
        {page.status}: {message}
      </p>
    </details>
  {/if}
</div>
