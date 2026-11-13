<script lang="ts">
  /** The title row of a config editor page: the name, what is served about the document,
   *  and the way back to the list. */
  import { resolve } from '$app/paths';
  import type { ConfigEditorView } from '$lib/config/configEditor.svelte';
  import { projectHref } from '$lib/projectPaths';

  interface Props {
    name: string;
    doc: ConfigEditorView['doc'];
    /** Project-relative list path, before `resolve`. */
    listPath: Parameters<typeof projectHref>[0];
    listLabel: string;
    /** Also say when the saved revision is not the active one, and where a clone came from. */
    lineage?: boolean;
  }

  let { name, doc: d, listPath, listLabel, lineage = false }: Props = $props();
</script>

<header class="flex flex-wrap items-baseline gap-3">
  <h1 class="text-h1 font-semibold tracking-tight">{name}</h1>
  {#if d}
    <span class="text-body text-fg-2" data-testid="config-meta">
      {d.source}{d.revision != null ? ` · revision ${d.revision}` : ''}{d.read_only
        ? ' · read-only'
        : ''}
    </span>
    {#if d.active}
      <span
        class="rounded border border-success/40 bg-success/10 px-1.5 text-meta text-success-text"
        >active{d.active_revision != null ? ` r${d.active_revision}` : ''}</span
      >
      {#if lineage && d.revision != null && d.active_revision != null && d.revision !== d.active_revision}
        <span class="text-meta text-warn-text" data-testid="saved-vs-active"
          >saved r{d.revision}, active r{d.active_revision}</span
        >
      {/if}
    {/if}
    {#if lineage && d.cloned_from}
      <span class="text-meta text-fg-3">cloned from <code>{d.cloned_from}</code></span>
    {/if}
  {/if}
  <span class="grow"></span>
  <a
    class="text-meta text-accent-text hover:underline"
    href={resolve(projectHref(listPath))}>{listLabel}</a
  >
</header>
