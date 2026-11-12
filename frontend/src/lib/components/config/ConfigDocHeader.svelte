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
  <h1 class="font-mono text-2xl font-semibold tracking-tight">{name}</h1>
  {#if d}
    <span class="text-sm text-zinc-400" data-testid="config-meta">
      {d.source}{d.revision != null ? ` · revision ${d.revision}` : ''}{d.read_only
        ? ' · read-only'
        : ''}
    </span>
    {#if d.active}
      <span
        class="rounded border border-emerald-500/40 bg-emerald-500/10 px-1.5 text-xs text-emerald-300"
        >active{d.active_revision != null ? ` r${d.active_revision}` : ''}</span
      >
      {#if lineage && d.revision != null && d.active_revision != null && d.revision !== d.active_revision}
        <span class="text-xs text-amber-300" data-testid="saved-vs-active"
          >saved r{d.revision}, active r{d.active_revision}</span
        >
      {/if}
    {/if}
    {#if lineage && d.cloned_from}
      <span class="font-mono text-xs text-zinc-500">cloned from {d.cloned_from}</span>
    {/if}
  {/if}
  <span class="grow"></span>
  <a class="text-xs text-blue-300 hover:underline" href={resolve(projectHref(listPath))}
    >{listLabel}</a
  >
</header>
