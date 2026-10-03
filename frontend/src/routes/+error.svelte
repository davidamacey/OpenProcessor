<script lang="ts">
  import { page } from '$app/state';
  import { isChunkLoadError } from '$lib/chunkRecovery';

  const message = $derived(page.error?.message ?? '');
  const loadFailure = $derived(isChunkLoadError(page.error));
</script>

<div
  class="flex h-screen flex-col items-center justify-center gap-3 bg-zinc-950 px-6 text-center text-zinc-100"
  data-testid="app-error"
>
  <p class="text-lg font-semibold">
    {loadFailure ? "This page couldn't finish loading" : 'Something went wrong'}
  </p>
  <p class="max-w-md text-sm text-zinc-400">
    {loadFailure
      ? 'The connection may have dropped, or the app was updated while this tab was open. Reloading usually fixes it.'
      : 'Reloading may help. If it keeps happening, check the details below.'}
  </p>
  <button
    type="button"
    class="rounded border border-zinc-700 px-3 py-1.5 text-sm hover:border-zinc-500"
    onclick={() => window.location.reload()}
  >
    Reload
  </button>
  {#if message}
    <details class="max-w-xl text-left text-xs text-zinc-500">
      <summary class="cursor-pointer">Details</summary>
      <p class="mt-1 break-words" data-testid="app-error-detail">
        {page.status}: {message}
      </p>
    </details>
  {/if}
</div>
