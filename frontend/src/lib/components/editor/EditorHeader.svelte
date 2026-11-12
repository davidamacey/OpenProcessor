<script lang="ts">
  /**
   * The editor's top bar: the way back, the file name and position in the list, and the save state,
   * which is always visible. A failed or conflicting save also shows the server's own words.
   */
  import { resolve } from '$app/paths';
  import type { SaveStatus } from '$lib/editor/editorCommands.svelte';

  interface Props {
    backHref: string;
    fileName: string;
    /** 0-based position in the list and its size; null when the image is not in it. */
    index: number | null;
    total: number | null;
    status: SaveStatus;
    /** The served message for the last failure. */
    message: string | null;
  }

  let { backHref, fileName, index, total, status, message }: Props = $props();

  const STATE: Record<SaveStatus, string> = {
    saved: 'Saved',
    saving: 'Saving…',
    conflict: 'Changed by someone else, reloaded',
    failed: 'Not saved',
  };
</script>

<header
  class="flex flex-wrap items-center gap-x-4 gap-y-1 border-b border-line px-3 py-2"
>
  <a
    href={resolve(backHref)}
    class="text-sm text-info-text hover:underline"
    data-testid="editor-back"
  >
    ‹ Workspace
  </a>
  <h1 class="min-w-0 truncate text-sm font-medium text-fg-1" title={fileName}>
    {fileName}
  </h1>
  {#if index !== null && total !== null}
    <span class="text-xs text-fg-2" data-testid="editor-position">
      {index + 1} of {total}
    </span>
  {/if}
  <span
    class="ml-auto text-xs {status === 'saved' || status === 'saving'
      ? 'text-fg-2'
      : 'text-warn-text'}"
    role="status"
    data-testid="save-state"
    data-status={status}
  >
    {STATE[status]}
  </span>
</header>
{#if message && (status === 'conflict' || status === 'failed')}
  <p
    class="border-b border-warn/40 bg-warn/10 px-3 py-1.5 text-sm text-warn-text"
    role="alert"
    data-testid="save-message"
  >
    {message}
  </p>
{/if}
